# Copyright (c) 2014-2025, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. See the top-level LICENSE file for details.
#
# LLNL-CODE-697807.
# All rights reserved.
#
# This file is part of LBANN: Livermore Big Artificial Neural Network
# Toolkit. For details, see http://software.llnl.gov/LBANN or
# https://github.com/LBANN and https://github.com/LLNL/LBANN.
#
# SPDX-License-Identifier: (Apache-2.0)
"""
Tests that the trampoline tears the process group down on every exit path.

``runpy`` used to be called with no ``try``/``finally``, so
``dist.destroy_process_group()`` ran only when the user's script returned
normally. Any exception -- or any ``sys.exit()`` with a non-zero status --
skipped it. A vestigial ``import atexit`` sat at the top of the file,
registering nothing, which is what an earlier attempt at this looks like.

Scope, deliberately narrow. This is *not* a hang: a rank that dies takes its
sockets with it, and a peer blocked in a gloo collective was measured raising
``Connection closed by peer`` within a third of a second, so the missing
teardown does not strand the job. What it does mean is that correctness rests
on one transport's socket-close behaviour. NCCL -- the primary path for this
tool -- enqueues on CUDA streams and polices itself with a watchdog thread on
its own timers, and neither that nor multi-node delivery can be exercised here.
Releasing the group explicitly is the cheap way not to depend on any of it.

This is also *not* an exit-code fix. An uncaught exception in the user's
script already exits non-zero by default, and the tests below pin that the
``finally`` preserves the status rather than swallowing or rewriting it.

The trampoline is executed as a subprocess so that the real exit status is
observable. Teardown is observed by having the user's script rebind
``torch.distributed.destroy_process_group`` to a function that drops a marker
file: the trampoline looks the attribute up on the shared module object at
call time, so the rebinding is visible to it, and this needs no real process
group and no rendezvous.
"""
import os
import shutil
import subprocess
import sys
import textwrap

from conftest import require_torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAMPOLINE_SOURCE = os.path.join(
    REPO_ROOT, "hpc_launcher", "torch", "torchrun_hpc_trampoline.py"
)

# Generous: the trampoline imports torch, which can be slow on a cold cache.
TIMEOUT = 300

# Prelude for every user script below. Fakes an initialized process group and
# swaps in a recording teardown, so the test observes whether the trampoline
# called it without needing a real one.
_INSTRUMENT = """\
import os
import sys
import torch.distributed as dist

_marker = os.environ["TRAMPOLINE_DESTROY_MARKER"]


def _record_destroy(*args, **kwargs):
    with open(_marker, "w") as fh:
        fh.write("destroyed")
    if os.environ.get("TRAMPOLINE_DESTROY_RAISES"):
        raise RuntimeError("destroy_process_group failed during teardown")


dist.is_initialized = lambda: True
dist.destroy_process_group = _record_destroy
"""


def _stage_launch_dir(path):
    """
    Build a stand-in for a launch folder: a directory whose only content is a
    copy of the trampoline, which is what ``cli/torchrun_hpc.py`` creates
    before handing the copy's path to the interpreter.
    """
    path.mkdir(parents=True, exist_ok=True)
    shutil.copy(TRAMPOLINE_SOURCE, str(path / "torchrun_hpc_trampoline.py"))
    return path


def _run_user_script(tmp_path, body, destroy_raises=False):
    """
    Run ``body`` (appended to the instrumentation prelude) under a staged
    trampoline and return ``(process, marker_path)``.

    ``TORCHRUN_HPC_SCHEDULER=local`` gives a world size of 1, so the trampoline
    never initializes a real process group and the test needs no rendezvous.
    The ``*_VISIBLE_DEVICES`` variables are blanked to force the CPU path.
    """
    launch_dir = _stage_launch_dir(tmp_path / "launch")
    script = tmp_path / "user_script.py"
    script.write_text(_INSTRUMENT + body)
    marker = tmp_path / "destroyed.marker"

    env = os.environ.copy()
    env["TORCHRUN_HPC_SCHEDULER"] = "local"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["ROCR_VISIBLE_DEVICES"] = ""
    env["HIP_VISIBLE_DEVICES"] = ""
    env["TRAMPOLINE_DESTROY_MARKER"] = str(marker)
    if destroy_raises:
        env["TRAMPOLINE_DESTROY_RAISES"] = "1"

    proc = subprocess.run(
        [
            sys.executable,
            "-u",
            str(launch_dir / "torchrun_hpc_trampoline.py"),
            str(script),
        ],
        env=env,
        cwd=str(launch_dir),
        capture_output=True,
        universal_newlines=True,
        timeout=TIMEOUT,
    )
    return proc, marker


def test_process_group_destroyed_when_user_script_raises(tmp_path):
    """
    An uncaught exception in the user's script must still release the process
    group, and must still exit non-zero with the traceback intact.

    Before the fix the exception propagated straight out of ``runpy`` past the
    unguarded ``destroy_process_group()`` call at the end of ``main()``, which
    therefore never ran.
    """
    require_torch()

    proc, marker = _run_user_script(
        tmp_path,
        textwrap.dedent(
            """\
            raise RuntimeError("user script blew up")
            """
        ),
    )

    assert marker.exists(), (
        "destroy_process_group() was not called when the user script raised\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    # The exception must still reach the user, unaltered, with the default
    # non-zero status. The teardown is additive, not a handler.
    assert proc.returncode == 1, f"{proc.stdout}\n{proc.stderr}"
    assert "RuntimeError: user script blew up" in proc.stderr, proc.stderr


def test_process_group_destroyed_on_nonzero_sys_exit(tmp_path):
    """
    A script that ends with ``sys.exit(7)`` -- the normal way a training job
    reports a failure it has already diagnosed -- must both release the group
    and keep its own status code.

    ``SystemExit`` is an exception like any other as far as ``runpy`` is
    concerned, so this took the same unguarded path out of ``main()``.
    """
    require_torch()

    proc, marker = _run_user_script(
        tmp_path,
        textwrap.dedent(
            """\
            sys.exit(7)
            """
        ),
    )

    assert marker.exists(), (
        "destroy_process_group() was not called on a non-zero sys.exit()\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    assert proc.returncode == 7, f"{proc.stdout}\n{proc.stderr}"


def test_process_group_destroyed_on_success(tmp_path):
    """
    Regression guard for the path that already worked: a script that returns
    normally still gets its process group destroyed, and still exits 0.
    """
    require_torch()

    proc, marker = _run_user_script(
        tmp_path,
        textwrap.dedent(
            """\
            print("USER_SCRIPT_OK", flush=True)
            """
        ),
    )

    assert marker.exists(), f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "USER_SCRIPT_OK" in proc.stdout, proc.stdout


def test_teardown_failure_does_not_mask_the_user_exception(tmp_path):
    """
    Teardown is best effort. If ``destroy_process_group()`` itself throws while
    unwinding -- plausible when the group is already half torn down by a dead
    peer -- the user's original exception is what must surface and what must
    set the exit status. An exception raised inside a ``finally`` block
    replaces the one in flight, so this has to be caught explicitly.
    """
    require_torch()

    proc, marker = _run_user_script(
        tmp_path,
        textwrap.dedent(
            """\
            raise RuntimeError("user script blew up")
            """
        ),
        destroy_raises=True,
    )

    assert marker.exists(), f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode == 1, f"{proc.stdout}\n{proc.stderr}"
    assert "RuntimeError: user script blew up" in proc.stderr, proc.stderr
    assert (
        "destroy_process_group failed during teardown" not in proc.stderr.splitlines()[-1]
    ), f"teardown error displaced the user's traceback:\n{proc.stderr}"


def test_no_vestigial_atexit_import(tmp_path):
    """
    The unused ``import atexit`` registered no hook and cleaned nothing up; it
    was a marker for the missing teardown, not an implementation of it. With
    the ``try``/``finally`` in place it must not come back, because a reader
    seeing it would reasonably assume cleanup is already handled.
    """
    require_torch()
    import hpc_launcher.torch.torchrun_hpc_trampoline as tramp

    assert not hasattr(tramp, "atexit"), (
        "torchrun_hpc_trampoline imports atexit but registers no hook; "
        "cleanup is done with try/finally around runpy"
    )
