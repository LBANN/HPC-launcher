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
``LocalScheduler`` regression tests (round 2: O1, O2, M1).

These three findings are one defect wearing three hats: ``LocalScheduler``
overrode the base class wholesale instead of extending it, so every
guarantee the base grew afterwards was silently dropped for the one backend
users reach for first when debugging -- and every guard written against
``args.local`` missed the equivalent ``--scheduler local`` spelling.

- **O1** -- ``--local -N 2 -n 2`` runs exactly *one* OS process. That is a
  spawn limitation, not a misreport: there is no second process anywhere, so
  nothing distributed is exercised. It cannot be fixed by reporting a
  different world size, so the launcher must at least say so out loud
  instead of producing a green single-rank run that looks like a passing
  two-rank one. The doc example that told users to do exactly this goes with
  it.
- **O2** -- the wholesale ``launcher_script`` override never emitted
  ``export HPC_LAUNCHER_MAX_GPU_MEM`` (so ``--fraction-max-gpu-mem`` was
  ignored under ``--local`` only) or ``export PYTHONPATH``, and never called
  ``build_command_string_and_batch_script``, which is where ``-x`` overrides
  are applied -- so ``-x`` was silently ignored under ``--local`` too.
- **M1** -- ``validate_arguments`` expressed "``--local`` jobs cannot be run
  in the background" in terms of ``args.local``, but ``--scheduler local``
  and ``--scheduler LocalScheduler`` select the *same* ``LocalScheduler``
  with ``args.local`` False. The guard was blind, the unsupported path ran,
  and because ``launch_command()`` returns ``[]`` the "submission" was a
  direct ``subprocess.run(..., capture_output=True)`` whose stdout was
  written nowhere: no ``out.log``, no job ID, exit 0, output gone.

``--local`` genuinely runs on a login node, so everything here is driven
through the real CLI entry points and asserts on generated scripts, real
process output and real exit codes.
"""
import argparse
import os
import subprocess
import sys

import pytest

from hpc_launcher.cli import common_args
from hpc_launcher.schedulers.local import LocalScheduler
from hpc_launcher.systems.system import GenericSystem, SystemParams

from conftest import require_torch


LAUNCH = [sys.executable, "-m", "hpc_launcher.cli.launch"]
TORCHRUN = [sys.executable, "-m", "hpc_launcher.cli.torchrun_hpc"]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TORCHRUN_DOC = os.path.join(REPO_ROOT, "torchrun-hpc_cli.md")

# The phrase the job-size warning must carry. Asserted rather than the whole
# sentence so wording can be improved without a test edit, but specific
# enough that a generic scheduler message cannot satisfy it.
WARNING_PHRASE = "single process"


def _local_scheduler(**kwargs) -> LocalScheduler:
    return LocalScheduler(nodes=1, procs_per_node=1, gpus_per_proc=0, **kwargs)


def _common_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    common_args.setup_arguments(parser)
    return parser


# ---------------------------------------------------------------------------
# O1 -- --local runs one process, whatever job size was requested
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "size_flags",
    [
        ["-N", "2", "-n", "2"],  # the exact invocation the doc advertised
        ["-N", "2"],             # two nodes cannot be one local process either
        ["-N", "1", "-n", "4"],  # four ranks on one node
        ["-g", "4"],             # four accelerators -> four processes
    ],
)
def test_local_warns_when_more_than_one_process_is_requested(size_flags, tmp_path):
    """
    A user who asks ``--local`` for more than one process gets exactly one,
    with no error and (before this fix) no warning at all -- so a smoke test
    of a DDP script "passes" without ever having created a second rank.

    Real multi-process local launching is a feature, not a fix; the launcher
    must instead say plainly that the requested job size is not being
    spawned. ``--dry-run`` keeps this to the diagnostic alone, with no side
    effects.
    """
    proc = subprocess.run(
        LAUNCH + ["--local"] + size_flags + ["--dry-run", "--", "/bin/echo", "hi"],
        capture_output=True,
        cwd=str(tmp_path),
    )
    stderr = proc.stderr.decode(errors="replace")

    assert proc.returncode == 0, f"expected a warning, not a failure:\n{stderr}"
    assert "local" in stderr.lower() and WARNING_PHRASE in stderr.lower(), (
        "requesting more than one process from --local produced no warning "
        f"that only one will run:\n{stderr!r}"
    )


def test_local_runs_exactly_one_process_and_says_so(tmp_path):
    """
    End-to-end companion to the test above: ``--local -N 1 -n 4`` really does
    produce a single line of output, and the warning that says so is the only
    thing standing between the user and a silently single-rank "distributed"
    run.
    """
    launch_dir = tmp_path / "one_proc"
    proc = subprocess.run(
        LAUNCH
        + [
            "--local", "-N", "1", "-n", "4",
            "-l", str(launch_dir),
            "--",
            sys.executable, "-c", "print('ran')",
        ],
        capture_output=True,
        cwd=str(tmp_path),
    )
    stderr = proc.stderr.decode(errors="replace")

    assert proc.returncode == 0, stderr
    assert (launch_dir / "out.log").read_text().splitlines() == ["ran"], (
        "the local run spawned more (or fewer) than one process"
    )
    assert WARNING_PHRASE in stderr.lower(), (
        f"four processes were requested and one ran, with no warning:\n{stderr!r}"
    )


@pytest.mark.parametrize(
    "size_flags",
    [
        ["-N", "1", "-n", "1"],  # explicitly one process
        ["-N", "1"],             # -n unspecified: filled in from the system
    ],
)
def test_local_does_not_warn_for_a_single_requested_process(size_flags, tmp_path):
    """
    Guard against a warning that cries wolf. ``configure_launch`` fills in
    ``procs_per_node`` from the *detected system* when ``-n`` is omitted (on
    a four-GPU node a bare ``-N 1`` becomes four processes per node), so a
    warning keyed on the resolved job size would fire on the most ordinary
    ``launch --local -N 1`` invocation there is. It has to key on what the
    user actually asked for.
    """
    proc = subprocess.run(
        LAUNCH + ["--local"] + size_flags + ["--dry-run", "--", "/bin/echo", "hi"],
        capture_output=True,
        cwd=str(tmp_path),
    )
    stderr = proc.stderr.decode(errors="replace")

    assert proc.returncode == 0, stderr
    assert WARNING_PHRASE not in stderr.lower(), (
        f"spurious job-size warning for {' '.join(size_flags)}:\n{stderr!r}"
    )


def test_doc_does_not_advertise_a_multiprocess_local_run():
    """
    ``torchrun-hpc_cli.md`` showed ``torchrun-hpc --local -N 2 -n 2
    test_script.py`` under "Local testing without scheduler", and the Tips
    section reinforces it ("Test locally first: Use ``--local`` flag for
    debugging"). Following that advice produces one rank-0 process and the
    impression that DDP/rendezvous code paths were exercised.

    No ``--local`` example may request more than one process, and the tip
    that recommends the flag has to carry the limitation -- that is where a
    user decides to trust a local run.
    """
    doc = open(TORCHRUN_DOC).read()

    size_flags = ("-N", "-n", "--nodes", "--procs-per-node", "-g", "--gpus-at-least")
    for line in doc.splitlines():
        if "--local" not in line:
            continue
        tokens = line.split()
        for flag in size_flags:
            if flag in tokens and tokens.index(flag) + 1 < len(tokens):
                value = tokens[tokens.index(flag) + 1]
                assert value == "1", (
                    f"doc example requests {flag} {value} with --local, which "
                    f"runs exactly one process:\n{line}"
                )

    lines = doc.splitlines()
    starts = [i for i, line in enumerate(lines) if "Test locally first" in line]
    assert len(starts) == 1, f"expected one 'Test locally first' tip, got {starts}"
    # The tip is a numbered list item that may wrap onto indented
    # continuation lines; take the whole item.
    tip = [lines[starts[0]]]
    for line in lines[starts[0] + 1:]:
        if not line.startswith((" ", "\t")) or not line.strip():
            break
        tip.append(line)
    tip = " ".join(tip)

    assert "one process" in tip or "single process" in tip, (
        "the tip recommends --local for local testing without stating that "
        f"it runs a single process regardless of the requested job size:\n{tip}"
    )


# ---------------------------------------------------------------------------
# O2 -- the local launcher script must extend the base, not replace it
# ---------------------------------------------------------------------------
def test_local_script_exports_max_gpu_mem(tmp_path):
    """
    ``--fraction-max-gpu-mem`` is delivered to the job as
    ``HPC_LAUNCHER_MAX_GPU_MEM``, which the trampoline reads. The base class
    emits it for every backend; the local override did not, so the cap was
    silently ignored under ``--local`` and the process could OOM the GPU it
    was meant to be limited on. Nothing is printed about it at default
    verbosity either, so the only symptom is the OOM.
    """
    system = GenericSystem()
    system.active_system_params = SystemParams(fraction_max_gpu_mem=0.5)

    script = _local_scheduler().launcher_script(
        system, "/bin/echo", ["hi"], blocking=True, launch_dir=str(tmp_path)
    )

    assert "export HPC_LAUNCHER_MAX_GPU_MEM=0.5" in script, (
        f"the local script drops the GPU memory cap:\n{script}"
    )


def test_local_script_exports_pythonpath(tmp_path):
    """
    The job runs from the launch directory, not the invocation directory, so
    the base class puts the caller's directory on ``PYTHONPATH`` -- which is
    what lets a script import a sibling module. The local override dropped
    the line, so ``torchrun-hpc --local train.py`` failed to import a sibling
    that the same run under Slurm/Flux/LSF imported fine.
    """
    script = _local_scheduler().launcher_script(
        GenericSystem(), "/bin/echo", ["hi"], blocking=True,
        launch_dir=str(tmp_path / "launchdir"),
    )

    assert "export PYTHONPATH=" in script, (
        f"the local script drops PYTHONPATH:\n{script}"
    )


def test_local_script_applies_x_overrides(tmp_path):
    """
    ``-x`` overrides are applied by ``build_command_string_and_batch_script``.
    The local override never called it, so the override pass did not run at
    all under ``--local``: the only mention of the key in the generated
    script was the recorded argv comment. There is no scheduler argv to
    override locally, so the outcome is inert either way -- but the pass has
    to run, because it is the same call that assembles the environment block.
    """
    scheduler = _local_scheduler()
    scheduler.override_launch_args = {"MY_OVERRIDE": "1"}

    script = scheduler.launcher_script(
        GenericSystem(), "/bin/echo", ["hi"], blocking=True,
        launch_dir=str(tmp_path),
    )

    assert scheduler.common_launch_args.get("MY_OVERRIDE") == "1", (
        "the -x override pass never ran for the local scheduler: "
        f"{scheduler.common_launch_args!r}"
    )
    assert "MY_OVERRIDE" in script, script


def test_local_script_keeps_what_it_legitimately_overrides(tmp_path):
    """
    Extending the base must not lose the two things a local run genuinely
    needs from its own implementation: there is no scheduler to place the job
    in its working directory (no ``--chdir``/``-D`` to set), so the script
    does the ``cd`` itself, and the command's arguments still have to be
    quoted so a value with a space stays one token.
    """
    scheduler = _local_scheduler(work_dir=str(tmp_path))

    script = scheduler.launcher_script(
        GenericSystem(), "/bin/echo", ["a b"], blocking=True,
        launch_dir=str(tmp_path),
    )

    assert f"cd {tmp_path}" in script, f"no cd into the working directory:\n{script}"
    assert "'a b'" in script, f"argument with a space was not quoted:\n{script}"


def test_local_end_to_end_delivers_the_launcher_environment(tmp_path):
    """
    The end-to-end form of O2, through the real CLI: the running process must
    see the GPU memory cap in its environment and a ``PYTHONPATH`` that lets
    it import a module sitting next to where the user invoked the launcher.

    The probe script deliberately lives *outside* the invocation directory,
    because Python puts a script's own directory on ``sys.path`` -- a probe
    stored next to the helper would import it whether or not the launcher
    contributed anything. ``-p fraction_max_gpu_mem=`` sets the same system
    parameter that ``torchrun-hpc --fraction-max-gpu-mem`` does, so this can
    run without torch.
    """
    invocation_dir = tmp_path / "project"
    invocation_dir.mkdir()
    (invocation_dir / "sibling_helper.py").write_text("VALUE = 'imported'\n")

    script_dir = tmp_path / "elsewhere"
    script_dir.mkdir()
    probe = script_dir / "probe.py"
    probe.write_text(
        "import os\n"
        "print('CAP', os.environ.get('HPC_LAUNCHER_MAX_GPU_MEM'))\n"
        "import sibling_helper\n"
        "print('SIBLING', sibling_helper.VALUE)\n"
    )

    launch_dir = invocation_dir / "run"
    proc = subprocess.run(
        LAUNCH
        + [
            "--local", "-N", "1", "-n", "1",
            "-p", "fraction_max_gpu_mem=0.5",
            "-l", str(launch_dir),
            "--",
            sys.executable, str(probe),
        ],
        capture_output=True,
        cwd=str(invocation_dir),
    )
    stderr = proc.stderr.decode(errors="replace")
    out = (launch_dir / "out.log").read_text()

    assert proc.returncode == 0, f"{stderr}\n{out}"
    assert "CAP 0.5" in out, (
        f"the GPU memory cap never reached the process:\n{out}\n{stderr}"
    )
    assert "SIBLING imported" in out, (
        f"the sibling module was not importable from the launch dir:\n{out}\n{stderr}"
    )


# ---------------------------------------------------------------------------
# M1 -- the background guard must follow the scheduler, not the flag
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "select_flags",
    [
        ["--local"],                      # already guarded before this fix
        ["--scheduler", "local"],         # same class, args.local is False
        ["--scheduler", "LocalScheduler"],
    ],
)
def test_local_background_submission_is_rejected(select_flags, tmp_path):
    """
    ``--scheduler local`` and ``--scheduler LocalScheduler`` select the same
    ``LocalScheduler`` as ``--local``, but leave ``args.local`` False, so the
    "cannot be run in the background" guard did not see them.

    What ran instead was the worst possible outcome, not a crash:
    ``launch_command()`` returns ``[]``, so the non-blocking "submission" was
    a direct ``subprocess.run(..., capture_output=True)`` of the launch
    script. On success its stdout is written nowhere (``out.log``/``err.log``
    are only opened on the blocking path), ``get_job_id()`` returns None, and
    the launcher exits 0 -- an apparently-successful background submission
    whose output is permanently lost. The canary proves the job really did
    run.
    """
    launch_dir = tmp_path / "bg"
    canary = tmp_path / "canary"

    proc = subprocess.run(
        LAUNCH
        + ["-N", "1"] + select_flags
        + ["--bg", "-l", str(launch_dir), "--", "/bin/touch", str(canary)],
        capture_output=True,
        cwd=str(tmp_path),
    )
    stderr = proc.stderr.decode(errors="replace")

    assert proc.returncode != 0, (
        "an unsupported local background submission reported success\n"
        f"stdout: {proc.stdout.decode(errors='replace')!r}\nstderr: {stderr!r}"
    )
    assert "background" in stderr, f"the failure did not explain itself:\n{stderr}"
    assert not canary.exists(), (
        "the job ran despite --bg being unsupported; its output goes nowhere"
    )
    assert not launch_dir.exists(), (
        "a launch directory was left behind for a job that was rejected"
    )


def test_local_background_guard_is_derived_from_the_scheduler_class():
    """
    The guard is derived from the resolved scheduler, so it holds for any
    spelling that reaches ``LocalScheduler`` -- including plain autodetection
    on a host with no batch system, which no flag-based check could ever see.
    """
    args = _common_args_parser().parse_args(
        ["-N", "1", "--scheduler", "local", "--bg"]
    )

    with pytest.raises(ValueError, match="background"):
        common_args.validate_scheduler_arguments(
            LocalScheduler(nodes=1, procs_per_node=1, gpus_per_proc=0), args
        )


def test_torchrun_hpc_local_background_submission_is_rejected(tmp_path):
    """
    ``torchrun-hpc`` resolves the scheduler through the same helper, so it
    inherits the same blind spot and needs the same guard; a ``torchrun-hpc
    --scheduler local --bg`` run that loses all of its output is exactly the
    debugging session this backend exists for.
    """
    require_torch()

    script = tmp_path / "t.py"
    script.write_text("print('hello')\n")
    launch_dir = tmp_path / "bg"

    proc = subprocess.run(
        TORCHRUN
        + ["-N", "1", "-n", "1", "--scheduler", "local", "--bg",
           "-l", str(launch_dir), str(script)],
        capture_output=True,
        cwd=str(tmp_path),
    )
    stderr = proc.stderr.decode(errors="replace")

    assert proc.returncode != 0, (
        f"stdout: {proc.stdout.decode(errors='replace')!r}\nstderr: {stderr!r}"
    )
    assert "background" in stderr, stderr
