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
CLI validation and usability regression tests.

- A missing command produces a clean validation error instead of a
  ``TypeError`` crash deep in the scheduler.
- The ``-x/--xargs`` override grammar (keys passed through verbatim, attached
  dashed forms, ``~key`` removal, values containing ``=``).
- ``--out/--err/-o/--save-hostlist`` are accepted when the run gets a launch
  directory (torchrun-hpc always does), but still rejected for a genuinely
  ephemeral blocking ``launch``.
- Re-running ``-l . -o name`` regenerates the script in place instead of
  raising ``shutil.SameFileError``.
- A relative-path argument that would not resolve from the launch directory
  produces a warning.
- ``--comm-backend`` validates and normalizes case-insensitively in one
  place shared by ``launch`` and ``torchrun-hpc``, so the two CLIs agree on
  what a given value means instead of ``launch`` silently forwarding an
  unrecognized value into a consumer that ignores it.
- ``torchrun-hpc --dry-run`` must not write (or clobber) the trampoline file.
- ``--out``/``--err`` with a directory component is a clean validation
  error, not an uncaught ``FileNotFoundError`` with a half-built launch
  directory left behind.
- ``--out`` and ``--err`` naming the same file is a clean validation error,
  instead of being accepted and then silently reduced to one of the two
  streams.
"""
import argparse
import importlib
import os
import socket
import subprocess
import sys

import pytest

from hpc_launcher.cli import common_args
from hpc_launcher.systems.lc import el_capitan_family

from conftest import require_torch


LAUNCH = [sys.executable, "-m", "hpc_launcher.cli.launch"]
TORCHRUN = [sys.executable, "-m", "hpc_launcher.cli.torchrun_hpc"]


def _override_parser() -> argparse.ArgumentParser:
    """A parser exposing just the common arguments, for exercising ``-x``."""
    parser = argparse.ArgumentParser()
    common_args.setup_arguments(parser)
    return parser


# ---------------------------------------------------------------------------
# Missing command is a clean error, not a TypeError crash
# ---------------------------------------------------------------------------
def test_missing_command_is_clean_error():
    """
    ``launch --local -N1 --dry-run`` with no command must fail with a clean,
    command-mentioning validation error rather than the ``TypeError:
    sequence item 0: expected str instance, NoneType found`` crash the dead
    guard used to allow through.
    """
    proc = subprocess.run(
        LAUNCH + ["--local", "-N1", "--dry-run"], capture_output=True
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode != 0, f"expected failure, got 0\n{stderr}"
    assert "TypeError" not in stderr, f"crashed with a TypeError:\n{stderr}"
    assert "command" in stderr.lower(), (
        f"error did not mention the missing command:\n{stderr}"
    )


# ---------------------------------------------------------------------------
# The -x/--xargs grammar
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tokens,expected",
    [
        # undashed key is kept verbatim (no dashes added)
        (["-x", "ntasks=8"], {"ntasks": "8"}),
        # multiple space-separated tokens after a single -x
        (["-x", "k1=v1", "k2=v2"], {"k1": "v1", "k2": "v2"}),
        # removal: ~key is kept verbatim (empty value)
        (["-x", "~--ntasks"], {"~--ntasks": ""}),
        # a dashed key can be removed space-separated (the ~ hides the dash
        # from argparse)
        (["-x", "~-nnodes"], {"~-nnodes": ""}),
        # attached short form is kept verbatim
        (["-x--ntasks=8"], {"--ntasks": "8"}),
        # attached long form via = is kept verbatim
        (["--xargs=--ntasks=8"], {"--ntasks": "8"}),
        # value may contain '=' (split on the first '=' only)
        (["-x", "foo=a=b"], {"foo": "a=b"}),
        # single-character key is also kept verbatim
        (["-x", "q=pbatch"], {"q": "pbatch"}),
    ],
)
def test_xargs_verbatim_key_forms(tokens, expected):
    """The parsed ``override_args`` dict matches the decided ``-x`` grammar."""
    args = _override_parser().parse_args(tokens)
    assert args.override_args == expected


def test_xargs_bad_token_is_clean_error():
    """A token that is neither ``key=value`` nor ``~key`` is a clean error."""
    with pytest.raises(SystemExit):
        _override_parser().parse_args(["-x", "notakeyvalue"])


@pytest.mark.parametrize(
    "override_tokens,expected",
    [
        # attached dashed form: the exact flag spelling lands in the script
        (["-x--myflag=myval"], "#SBATCH --myflag=myval"),
        # undashed key: passed through verbatim, no dashes added
        (["-x", "myflag=myval"], "#SBATCH myflag=myval"),
    ],
)
def test_xargs_flag_lands_in_generated_script(tmp_path, override_tokens, expected):
    """
    End-to-end: an ``-x`` override reaches the generated batch script with the
    exact spelling the user gave. Uses ``--scheduler slurm --bg --setup-only``
    so the override is emitted as a header directive and the internal run
    command, without submitting anything.
    """
    proc = subprocess.run(
        LAUNCH
        + [
            "--scheduler", "slurm",
            "-N1", "-n1",
            "--bg", "--setup-only",
            "-l", str(tmp_path),
        ]
        + override_tokens
        + ["--", "echo", "hi"],
        capture_output=True,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, stderr
    script = (tmp_path / "launch.sh").read_text()
    assert expected in script, (
        f"override did not land verbatim in the script:\n{script}"
    )


# ---------------------------------------------------------------------------
# out/err/etc. allowed when a launch directory is (auto-)provided, but
# still rejected for a genuinely ephemeral blocking launch.
# ---------------------------------------------------------------------------
def test_out_err_allowed_with_auto_launch_dir(tmp_path):
    """
    ``torchrun-hpc`` always runs from a launch directory (it auto-defaults
    ``-l``), so ``--out`` must be accepted -- not rejected as an ephemeral
    interactive job. The genuinely ephemeral rejection must still
    fire for a blocking ``launch`` with no ``-l``.
    """
    require_torch()

    driver = tmp_path / "trivial.py"
    driver.write_text("print('hello')\n")

    # torchrun-hpc auto-provides a launch dir: --out is accepted.
    proc = subprocess.run(
        TORCHRUN
        + [
            "--local", "-N1", "-n1",
            "--setup-only",
            "--out", "t.log",
            str(driver),
        ],
        cwd=str(tmp_path),
        capture_output=True,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, (
        f"torchrun-hpc --out with an auto launch dir should succeed:\n{stderr}"
    )

    # A blocking launch with no -l is genuinely ephemeral: --out is rejected.
    proc = subprocess.run(
        LAUNCH + ["--local", "-N1", "-n1", "--out", "t.log", "echo", "hi"],
        cwd=str(tmp_path),
        capture_output=True,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode != 0, "ephemeral --out should have been rejected"
    assert "ephemeral" in stderr.lower(), (
        f"expected an ephemeral-job rejection message:\n{stderr}"
    )


# ---------------------------------------------------------------------------
# Re-running -l . -o name must not crash with SameFileError
# ---------------------------------------------------------------------------
def test_rerun_same_output_script(tmp_path):
    """
    Running ``launch -l . -o run.sh --setup-only`` twice in the same directory
    must regenerate the script in place rather than aborting with
    ``shutil.SameFileError`` on the second run.
    """
    cmd = LAUNCH + [
        "--local", "-N1", "-n1",
        "-l", ".",
        "-o", "run.sh",
        "--setup-only",
        "echo", "hi",
    ]

    for i in range(2):
        proc = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True)
        stderr = proc.stderr.decode(errors="replace")
        assert proc.returncode == 0, f"run {i} failed:\n{stderr}"
        assert "SameFileError" not in stderr, f"run {i} hit SameFileError:\n{stderr}"

    assert (tmp_path / "run.sh").exists(), "the output script was not regenerated"


# ---------------------------------------------------------------------------
# Relative-path argument warning
# ---------------------------------------------------------------------------
def test_relative_arg_warning(tmp_path):
    """
    ``launch -l outdir python script.py`` where ``./script.py`` exists in the
    invocation directory but not under the launch directory must warn that the
    relative path will not resolve once the job cd's into the launch dir.
    """
    (tmp_path / "script.py").write_text("print(1)\n")

    proc = subprocess.run(
        LAUNCH
        + [
            "--local", "-N1", "-n1",
            "-l", "outdir",
            "--setup-only",
            "python", "script.py",
        ],
        cwd=str(tmp_path),
        capture_output=True,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, stderr
    assert "relative path" in stderr and "script.py" in stderr, (
        f"expected a relative-path warning for script.py:\n{stderr}"
    )


def test_no_relative_arg_warning_when_launch_dir_is_cwd(tmp_path):
    """
    The relative-path warning must not fire when the launch directory is
    the current directory (``-l .``): relative paths still resolve.
    """
    (tmp_path / "script.py").write_text("print(1)\n")

    proc = subprocess.run(
        LAUNCH
        + [
            "--local", "-N1", "-n1",
            "-l", ".",
            "--setup-only",
            "python", "script.py",
        ],
        cwd=str(tmp_path),
        capture_output=True,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, stderr
    assert "relative path" not in stderr, (
        f"unexpected relative-path warning when -l . :\n{stderr}"
    )


# ---------------------------------------------------------------------------
# --comm-backend must validate and normalize case-insensitively, in one
# place shared by `launch` and `torchrun-hpc`.
# ---------------------------------------------------------------------------
def _run_main_el_capitan(monkeypatch, tmp_path, module_name, argv,
                          hostname="tuolumne0001"):
    """
    Run a CLI entry point's ``main()`` in-process (not via ``subprocess``),
    with the hostname patched so ``autodetect_current_system()`` resolves to
    ``ElCapitan`` -- the same ``@patch("socket.gethostname", ...)`` pattern
    ``tests/system_autodetect_test.py`` uses for every El-Capitan-family
    test.

    In-process rather than ``subprocess`` is required specifically here: the
    ``--comm-backend`` bug only shows up once
    ``ElCapitan.environment_variables()`` runs (it is the sole consumer of
    ``job_comm_protocol``), and a *subprocess*'s
    hostname cannot be patched from the parent test process the way
    ``sys.argv`` can be -- ``socket.gethostname()`` is a real syscall
    wrapper that ignores both monkeypatching done in another process and the
    ``HOSTNAME`` environment variable (verified directly: ``env
    HOSTNAME=lassen0001 python3 -c "import socket;
    print(socket.gethostname())"`` still prints the real host). Every other
    test in this module drives the CLI via ``subprocess`` instead; this is
    the one exception, made only where the hostname patch requires it.

    :return: the code the entry point's ``sys.exit()`` was called with.
    """
    from hpc_launcher.systems import autodetect

    module = importlib.import_module(module_name)
    monkeypatch.setattr(sys, "argv", ["prog"] + list(argv))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: hostname)
    autodetect.clear_autodetected_system()
    try:
        with pytest.raises(SystemExit) as exc_info:
            module.main()
    finally:
        # Reset the module-level cache regardless of outcome, so a failure
        # here cannot leak a patched hostname's resolved system into a
        # later, unrelated test.
        autodetect.clear_autodetected_system()
    return exc_info.value.code


def _nccl_net_plugin_present(launch_dir) -> bool:
    """Whether the generated script enabled the RCCL/AWS-OFI env block."""
    script = launch_dir / "launch.sh"
    return script.exists() and "NCCL_NET_PLUGIN" in script.read_text()


@pytest.fixture
def el_capitan_rccl_env(monkeypatch, tmp_path):
    """
    Pin the two host-dependent inputs to the RCCL/AWS-OFI block so the
    ``--comm-backend`` tests below assert on the CLI's behaviour rather than
    on the machine running them.

    ``ElCapitan.environment_variables()`` emits ``NCCL_NET_PLUGIN`` only when
    a ROCm version resolves *and* an aws-ofi-rccl plugin tree is found for
    it. Both come from the host: the version from ``torch.version.hip`` or
    ``ROCM_PATH``, the plugin from ``/collab/usr/global/tools/rccl/$SYS_TYPE``.
    On an LC MI300A node both are present and these tests passed; on a
    CPU-only CI runner neither is, so the profile took its "could not
    determine the ROCm runtime version" path and the tests failed for a
    reason that has nothing to do with ``--comm-backend``. Point the probe at
    a scratch tree and pin the version, exactly as
    ``tests/el_capitan_env_test.py`` does for this same profile.

    The version is pinned by patching ``_rocm_runtime_version`` rather than
    by faking ``torch`` in ``sys.modules`` (el_capitan_env_test's approach):
    ``test_launch_and_torchrun_hpc_agree_on_comm_backend`` runs
    ``torchrun_hpc.main()`` in-process, which needs the real module. Version
    *resolution* is covered on its own in el_capitan_env_test.py; what these
    tests need is only that it lands somewhere determinate.
    """
    for var in ("NCCL_NET", "NCCL_NET_PLUGIN", "LBANN_USE_THIS_OFI_PLUGIN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SYS_TYPE", "test_sys_type")
    root = tmp_path / "rccl-plugins"
    (root / "test_sys_type" / "rocm-7.2.0" / "install" / "lib").mkdir(parents=True)
    monkeypatch.setattr(el_capitan_family, "_AWS_OFI_RCCL_ROOT", str(root))
    # 7.2 so the >=7.1 branch (the one that sets NCCL_NET_PLUGIN) is taken.
    monkeypatch.setattr(
        el_capitan_family,
        "_rocm_runtime_version",
        lambda: el_capitan_family._RocmRuntime((7, 2, 0), "test", False),
    )


@pytest.mark.parametrize("backend", ["NCCL", "nccl", "RCCL", "rccl"])
def test_launch_comm_backend_nccl_and_rccl_agree(
    monkeypatch, tmp_path, el_capitan_rccl_env, backend
):
    """
    ``launch --comm-backend NCCL`` (either case) on an El-Capitan-family
    system must enable the same RCCL/AWS-OFI environment block as
    ``--comm-backend RCCL``.

    Before the fix, ``--comm-backend`` was forwarded to
    ``system.job_comm_protocol`` completely unvalidated, and the sole
    consumer (``ElCapitan.environment_variables``) only recognizes the
    literal strings ``"RCCL"``/``"*CCL"`` case-insensitively -- so ``NCCL``
    was silently treated as "no protocol requested": the generated script
    was missing ``NCCL_NET_PLUGIN``, ``NCCL_NET=libfabric``, the AWS-OFI
    plugin on ``LD_LIBRARY_PATH``, and the ``FI_CXI_RDZV_*`` tuning -- nine
    fewer exports than ``--comm-backend RCCL`` -- with no warning that the
    requested backend did nothing. ``RCCL``/``rccl`` are included as the
    contrast case: the consumer already matched them (via its own
    ``.upper()``), so they must keep working exactly as before.
    """
    launch_dir = tmp_path / "r"
    code = _run_main_el_capitan(
        monkeypatch, tmp_path, "hpc_launcher.cli.launch",
        [
            "--scheduler", "slurm", "-N2", "--bg", "--setup-only",
            "-l", str(launch_dir), "--comm-backend", backend,
            "--", "echo", "hi",
        ],
    )
    assert code == 0
    assert _nccl_net_plugin_present(launch_dir), (
        f"--comm-backend {backend} did not enable the RCCL/AWS-OFI "
        f"environment block:\n{(launch_dir / 'launch.sh').read_text()}"
    )


def test_launch_and_torchrun_hpc_agree_on_comm_backend(
    monkeypatch, tmp_path, el_capitan_rccl_env
):
    """
    The core of the shared-validation fix: ``launch`` and ``torchrun-hpc``
    must not diverge for the same ``--comm-backend`` value.

    Before the fix, ``launch --comm-backend NCCL`` produced a script missing
    the RCCL/AWS-OFI block (see
    ``test_launch_comm_backend_nccl_and_rccl_agree``), while ``torchrun-hpc
    --comm-backend NCCL`` already included it -- not because torchrun-hpc
    validated the value, but because it unconditionally collapsed *every*
    non-"MPI" value (typos included) to ``"*CCL"``. The two entry points
    silently disagreed about identical input, for unrelated reasons on
    each side.
    """
    require_torch()

    launch_dir = tmp_path / "r_launch"
    code = _run_main_el_capitan(
        monkeypatch, tmp_path, "hpc_launcher.cli.launch",
        [
            "--scheduler", "slurm", "-N2", "--bg", "--setup-only",
            "-l", str(launch_dir), "--comm-backend", "NCCL",
            "--", "echo", "hi",
        ],
    )
    assert code == 0

    train = tmp_path / "train.py"
    train.write_text("print(1)\n")
    torchrun_dir = tmp_path / "r_torchrun"
    code = _run_main_el_capitan(
        monkeypatch, tmp_path, "hpc_launcher.cli.torchrun_hpc",
        [
            "--scheduler", "slurm", "-N2", "--bg", "--setup-only",
            "-l", str(torchrun_dir), "--comm-backend", "NCCL", str(train),
        ],
    )
    assert code == 0

    launch_script = (launch_dir / "launch.sh").read_text()
    torchrun_script = (torchrun_dir / "launch.sh").read_text()
    assert _nccl_net_plugin_present(launch_dir) == _nccl_net_plugin_present(torchrun_dir), (
        "launch and torchrun-hpc disagree on --comm-backend NCCL:\n"
        f"launch:\n{launch_script}\ntorchrun-hpc:\n{torchrun_script}"
    )
    assert _nccl_net_plugin_present(launch_dir), (
        "both entry points should have enabled the RCCL/AWS-OFI block"
    )


def test_comm_backend_invalid_value_rejected_by_launch(tmp_path):
    """
    An unrecognized ``--comm-backend`` value (a typo, e.g.) must be a clean
    argparse usage error for ``launch``, not a silent no-op that produces a
    script with an un-accelerated transport and no warning at all. This does
    not need the El-Capitan hostname patch: argparse validates ``choices``
    while parsing, before any system autodetection runs, so a plain
    ``subprocess`` invocation (like the rest of this module) is enough.
    """
    before = set(os.listdir(tmp_path))
    proc = subprocess.run(
        LAUNCH
        + [
            "--scheduler", "slurm", "-N1", "--bg", "--setup-only",
            "-l", str(tmp_path / "r"),
            "--comm-backend", "totally-bogus",
            "--", "echo", "hi",
        ],
        capture_output=True,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode != 0, f"invalid --comm-backend was accepted:\n{stderr}"
    assert "TOTALLY-BOGUS" in stderr, (
        f"rejection did not name the bad value:\n{stderr}"
    )
    assert set(os.listdir(tmp_path)) == before, (
        "an invalid --comm-backend must be rejected before any launch "
        f"directory is created; new entries: {set(os.listdir(tmp_path)) - before}"
    )


def test_comm_backend_invalid_value_rejected_by_torchrun_hpc(tmp_path):
    """Same as above, for torchrun-hpc -- the two entry points must agree."""
    require_torch()
    train = tmp_path / "train.py"
    train.write_text("print(1)\n")
    before = set(os.listdir(tmp_path))

    proc = subprocess.run(
        TORCHRUN
        + [
            "--local", "-N1", "-n1", "--setup-only",
            "--comm-backend", "totally-bogus",
            str(train),
        ],
        cwd=str(tmp_path),
        capture_output=True,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode != 0, f"invalid --comm-backend was accepted:\n{stderr}"
    assert "TOTALLY-BOGUS" in stderr, (
        f"rejection did not name the bad value:\n{stderr}"
    )


# ---------------------------------------------------------------------------
# torchrun-hpc --dry-run must not write (or clobber) the trampoline.
# ---------------------------------------------------------------------------
def test_dry_run_does_not_write_or_clobber_trampoline(tmp_path):
    """
    ``torchrun-hpc --dry-run -l .`` (or any already-existing launch
    directory) must not write ``torchrun_hpc_trampoline.py``. The
    ``shutil.copy`` that stages the trampoline was previously guarded only
    by ``os.path.exists(folder_name)``, not ``args.dry_run``, unlike every
    other launch-folder artifact (``create_launch_folder``'s
    ``os.makedirs``, and the script write/submission inside
    ``scheduler.launch``). The narrowest and most damaging trigger is a
    pre-existing user file of the same name in the launch directory
    (overwhelmingly ``-l .``): a plain ``--dry-run`` preview run must not
    clobber it, and must not write it at all.
    """
    require_torch()
    driver = tmp_path / "train.py"
    driver.write_text("print('hello')\n")
    trampoline = tmp_path / "torchrun_hpc_trampoline.py"
    trampoline.write_text("MY IMPORTANT USER FILE\n")

    proc = subprocess.run(
        TORCHRUN
        + [
            "--dry-run", "--scheduler", "slurm", "-l", ".",
            "-N", "1", str(driver),
        ],
        cwd=str(tmp_path),
        capture_output=True,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, stderr
    assert trampoline.read_text() == "MY IMPORTANT USER FILE\n", (
        "--dry-run must not clobber an existing torchrun_hpc_trampoline.py"
    )
    assert not (tmp_path / "launch.sh").exists(), (
        "--dry-run must not write launch.sh either (already covered "
        "elsewhere, asserted here as a sanity check on the same run)"
    )


def test_dry_run_does_not_create_trampoline_when_absent(tmp_path):
    """
    Contrast case: when no trampoline file exists yet, ``--dry-run`` must
    not create one either -- ruling out a fix that only special-cases
    overwriting and not creation.
    """
    require_torch()
    driver = tmp_path / "train.py"
    driver.write_text("print('hello')\n")
    trampoline = tmp_path / "torchrun_hpc_trampoline.py"
    assert not trampoline.exists()

    proc = subprocess.run(
        TORCHRUN
        + [
            "--dry-run", "--scheduler", "slurm", "-l", ".",
            "-N", "1", str(driver),
        ],
        cwd=str(tmp_path),
        capture_output=True,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, stderr
    assert not trampoline.exists(), (
        "--dry-run must not write torchrun_hpc_trampoline.py when it did "
        "not exist before"
    )


# ---------------------------------------------------------------------------
# --out/--err with a directory component must be a clean validation error,
# not an uncaught FileNotFoundError with a half-built launch dir left behind.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("flag", ["--out", "--err"])
def test_out_err_with_directory_component_is_clean_error(tmp_path, flag):
    """
    ``launch -N1 --local -l r3 --out logs/o.log -- /bin/echo hi`` must fail
    at ``validate_arguments`` with a ``ValueError`` naming the offending
    path -- the same, pre-existing failure mode ``-o`` already uses for a
    path-bearing output-script name (an uncaught exception, still reported
    as a Python traceback, but the *right* one: a deliberate validation
    raise, not an incidental crash) -- rather than creating ``r3/`` and
    ``r3/launch.sh`` and only then dying with an uncaught
    ``FileNotFoundError`` at ``open(self.out_log_file, "wb")``, deep inside
    ``scheduler.py``, well after the launch directory was already built. The
    contrasting ``-o`` case already failed this same way *before* any
    directory was created; ``--out``/``--err`` must now match it.
    """
    launch_dir = tmp_path / "r3"
    proc = subprocess.run(
        LAUNCH
        + [
            "--local", "-N1",
            "-l", str(launch_dir),
            flag, "logs/o.log",
            "--", "/bin/echo", "hi",
        ],
        capture_output=True,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode != 0, "a path-bearing --out/--err was accepted"
    assert "FileNotFoundError" not in stderr, (
        f"the uncaught FileNotFoundError still leaked through:\n{stderr}"
    )
    assert "ValueError" in stderr and "logs/o.log" in stderr, (
        f"expected a ValueError naming the offending path:\n{stderr}"
    )
    assert not launch_dir.exists(), (
        f"a half-built launch directory was left behind:\n{stderr}"
    )


# ---------------------------------------------------------------------------
# --out and --err resolving to the same path must be rejected, not silently
# reduced to one of the two streams.
# ---------------------------------------------------------------------------
def test_out_and_err_at_the_same_path_are_rejected(tmp_path):
    """
    ``--out both.log --err both.log`` reads as "give me one combined log",
    which is a natural thing to ask for -- and used to be accepted and then
    quietly not done. ``scheduler.py``'s blocking branch opens two
    independent ``"wb"`` handles on the one path and hands one to each of
    ``console_pipe``'s two replicator tasks; each writes from its own file
    offset, so the file ends up holding whichever stream wrote last (here:
    ``ERRLINE\n``, 8 bytes, with ``OUTLINE`` gone) and, once the streams are
    large enough to interleave, arbitrary mutual overwriting instead. The
    console showed both streams and the exit code was 0, so nothing about
    the run suggested the log was wrong.

    It is rejected rather than merged: sharing a single handle between the
    two replicators would still tear lines apart, because they write
    concurrently in ``buffer_size``-byte chunks with no line framing, so
    "one combined log" would become byte-interleaved rather than
    interleaved-by-line. An upfront error says so and leaves the user to ask
    for the merge they actually want (``2>&1``, or ``--out``/``--err`` at
    distinct paths). It also matches how the neighbouring ``-o`` and
    ``--out``/``--err`` path-component checks in ``validate_arguments``
    behave, and -- being a validation check rather than a fix to one
    branch of ``scheduler.py`` -- it covers ``--bg`` too, where the same
    two names become the scheduler's own ``--output``/``--error``
    directives.
    """
    launch_dir = tmp_path / "r2"
    proc = subprocess.run(
        LAUNCH
        + [
            "--local", "-N1",
            "-l", str(launch_dir),
            "--out", "both.log",
            "--err", "both.log",
            "--", "/bin/sh", "-c", "echo OUTLINE; echo ERRLINE 1>&2",
        ],
        capture_output=True,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode != 0, (
        "--out and --err at the same path were accepted; the log will "
        "silently contain only one of the two streams"
    )
    assert "ValueError" in stderr and "both.log" in stderr, (
        f"expected a ValueError naming the shared path:\n{stderr}"
    )
    assert not (launch_dir / "both.log").exists(), (
        f"a truncated combined log was written anyway:\n{stderr}"
    )
