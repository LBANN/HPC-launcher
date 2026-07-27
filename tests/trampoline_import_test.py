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
Tests that the trampoline puts the user's own code on ``sys.path``.

``runpy.run_path()`` does *not* add the target file's directory to
``sys.path`` the way ``python script.py`` does, and the program the schedulers
actually execute is ``python -u <launch_dir>/torchrun_hpc_trampoline.py
<script>``. Python's automatic ``sys.path[0]`` insertion therefore names the
*launch folder*, which holds nothing but the copied trampoline. A training
script that does ``import helper`` for a sibling module died with
``ModuleNotFoundError`` even though ``python train.py`` ran it fine.

The only thing that ever compensated was ``export PYTHONPATH=...`` in the
generated batch script, which puts the *invocation* directory on the path.
That rescues exactly one layout -- code sitting directly in the directory the
user launched from, under a batch scheduler. It does nothing for ``--local``
(``LocalScheduler`` emits no such export), and nothing for a script in a
subdirectory, because the invocation directory is not the script's directory.

The tests below cover both the end-to-end ``--local`` path and the trampoline
in isolation. The isolated cases stage a fake launch folder -- a copy of the
trampoline in a directory of its own -- and execute it by path, exactly as the
generated launch script does, so that ``sys.path[0]`` is the launch folder and
not the test's working directory. Running the trampoline with ``python -m``
instead would silently paper the bug over by putting the cwd on ``sys.path``.

These tests need a CPU-capable torch; the import is guarded with the shared
``require_torch()`` helper.
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


def _stage_launch_dir(path):
    """
    Build a stand-in for a launch folder: a directory whose only content is a
    copy of the trampoline, which is precisely what
    ``cli/torchrun_hpc.py`` creates before handing the copy's path to the
    interpreter.
    """
    path.mkdir(parents=True, exist_ok=True)
    shutil.copy(TRAMPOLINE_SOURCE, str(path / "torchrun_hpc_trampoline.py"))
    return path


def _run_trampoline(launch_dir, args, cwd, pythonpath=None):
    """
    Run the staged trampoline the way a generated launch script does: by
    absolute path, with the job's working directory set to ``cwd``.

    ``TORCHRUN_HPC_SCHEDULER=local`` gives a world size of 1, so no rendezvous
    is attempted and the test does not depend on any batch scheduler. The
    ``*_VISIBLE_DEVICES`` variables are blanked to force the CPU/gloo path so
    the test neither needs nor contends for a GPU.
    """
    env = os.environ.copy()
    env["TORCHRUN_HPC_SCHEDULER"] = "local"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["ROCR_VISIBLE_DEVICES"] = ""
    env["HIP_VISIBLE_DEVICES"] = ""
    # hpc_launcher is installed, so the trampoline can import it without help.
    # Control PYTHONPATH explicitly instead of inheriting whatever the test
    # runner happened to have, since PYTHONPATH is the mechanism under test.
    env.pop("PYTHONPATH", None)
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)

    cmd = [
        sys.executable,
        "-u",
        str(launch_dir / "torchrun_hpc_trampoline.py"),
    ] + [str(a) for a in args]
    return subprocess.run(
        cmd,
        env=env,
        cwd=str(cwd),
        capture_output=True,
        universal_newlines=True,
        timeout=TIMEOUT,
    )


def _write_sibling_pair(directory, value=8):
    """
    Write ``train.py`` plus the sibling ``helper.py`` it imports, the minimal
    shape of every multi-file training repository.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "helper.py").write_text(f"VALUE = {value}\n")
    (directory / "train.py").write_text(
        textwrap.dedent(
            """\
            import os
            import sys

            print("SYSPATH0=" + sys.path[0], flush=True)
            import helper
            print("RESULT=%d" % helper.VALUE, flush=True)
            """
        )
    )
    return directory / "train.py"


# ---------------------------------------------------------------------------
# End-to-end: --local, the mode this breaks hardest
# ---------------------------------------------------------------------------
def test_local_scheduler_sibling_import(tmp_path):
    """
    ``torchrun-hpc --local train.py`` from a directory holding ``train.py`` and
    its sibling ``helper.py``.

    This is the documented way to smoke-test a training script before scaling
    out, and it is the worst-affected configuration: ``LocalScheduler`` does not
    emit the ``PYTHONPATH`` export that saves the batch schedulers, so the
    variable reaches the job literally unset and nothing at all puts the
    user's code on the path. Before the fix this died with
    ``ModuleNotFoundError: No module named 'helper'`` while plain
    ``python train.py`` printed ``RESULT=8``.
    """
    require_torch()

    proj = tmp_path / "proj"
    _write_sibling_pair(proj)

    cmd = [
        sys.executable,
        "-m",
        "hpc_launcher.cli.torchrun_hpc",
        "--local",
        "-N1",
        "train.py",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(proj),
        capture_output=True,
        universal_newlines=True,
        timeout=TIMEOUT,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "RESULT=8" in proc.stdout, f"{proc.stdout}\n{proc.stderr}"


# ---------------------------------------------------------------------------
# The trampoline in isolation
# ---------------------------------------------------------------------------
def test_script_directory_is_first_on_sys_path(tmp_path):
    """
    The script's own directory must be ``sys.path[0]``, which is the entry
    ``python script.py`` creates and ``runpy.run_path`` does not.

    Asserting on position, not merely membership, is deliberate: the user's
    own module has to win over anything the launcher put on the path ahead of
    it, including the launch folder itself.
    """
    require_torch()

    proj = tmp_path / "proj"
    script = _write_sibling_pair(proj)
    launch_dir = _stage_launch_dir(tmp_path / "launch")

    proc = _run_trampoline(launch_dir, [script], cwd=launch_dir)

    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert f"SYSPATH0={proj}" in proc.stdout, f"{proc.stdout}\n{proc.stderr}"


def test_sibling_import_when_script_is_in_a_subdirectory(tmp_path):
    """
    ``torchrun-hpc src/train.py``, with ``src/helper.py`` alongside it.

    The batch schedulers' ``PYTHONPATH`` export names the *invocation*
    directory, so it is set here to model a healthy batch job -- and it still
    does not help, because the module lives in ``src/`` and the export points
    at ``src/``'s parent. This row of the breakage matrix bites Slurm, Flux and
    LSF as well as ``--local``.
    """
    require_torch()

    proj = tmp_path / "proj"
    proj.mkdir()
    script = _write_sibling_pair(proj / "src")
    launch_dir = _stage_launch_dir(proj / "torchrun_hpc-train.py_stamp")

    proc = _run_trampoline(launch_dir, [script], cwd=launch_dir, pythonpath=proj)

    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "RESULT=8" in proc.stdout, f"{proc.stdout}\n{proc.stderr}"


def test_sibling_import_with_launch_dir_outside_the_project(tmp_path):
    """
    ``torchrun-hpc -l /some/absolute/scratch/run1 -- train.py``.

    An absolute custom launch directory sends the generated ``PYTHONPATH``
    somewhere unrelated to the user's code (its own parent), which is modelled
    here. With no fallback in the trampoline the sibling import had no way to
    resolve; the script's own directory is what makes this work regardless of
    where the launch folder was placed.
    """
    require_torch()

    proj = tmp_path / "proj"
    script = _write_sibling_pair(proj)
    scratch = tmp_path / "scratch"
    launch_dir = _stage_launch_dir(scratch / "run1")

    proc = _run_trampoline(launch_dir, [script], cwd=launch_dir, pythonpath=scratch)

    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "RESULT=8" in proc.stdout, f"{proc.stdout}\n{proc.stderr}"


def test_sibling_import_when_launch_dir_is_the_script_dir(tmp_path):
    """
    Regression guard for the one configuration that already worked: ``-l .``,
    where the launch folder, the invocation directory and the script's
    directory all coincide, so the trampoline's own ``sys.path[0]`` happened to
    be right. It must keep working after the fix.
    """
    require_torch()

    proj = tmp_path / "proj"
    script = _write_sibling_pair(proj)
    launch_dir = _stage_launch_dir(proj)

    proc = _run_trampoline(launch_dir, [script], cwd=launch_dir)

    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "RESULT=8" in proc.stdout, f"{proc.stdout}\n{proc.stderr}"


def test_module_mode_uses_the_working_directory(tmp_path):
    """
    ``-m mypkg`` has no script whose directory could be used, so the trampoline
    follows what ``python -m`` itself does and prepends the process's working
    directory.

    Note what this does and does not buy under a real launch: the schedulers
    run the job *from the launch folder*, so the working directory the
    trampoline sees is the launch folder rather than the directory the user
    typed the command in. Module mode therefore still relies on the launch
    script exporting the invocation directory on ``PYTHONPATH``; the trampoline
    has no way to recover that directory on its own. What is fixed here is the
    trampoline's own contract -- run it with the working directory of the code
    and ``-m`` resolves, as it would under the interpreter.
    """
    require_torch()

    work = tmp_path / "work"
    pkg = work / "mypkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("VALUE = 8\n")
    (pkg / "__main__.py").write_text(
        "import mypkg\nprint('RESULT=%d' % mypkg.VALUE, flush=True)\n"
    )
    launch_dir = _stage_launch_dir(tmp_path / "launch")

    proc = _run_trampoline(launch_dir, ["-m", "mypkg"], cwd=work)

    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "RESULT=8" in proc.stdout, f"{proc.stdout}\n{proc.stderr}"
