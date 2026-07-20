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
Regression tests for shell quoting / injection in generated launch scripts
(review finding D1).

The generated ``launch.sh`` is executed by ``/bin/sh`` under the scheduler
(sbatch/bsub/flux) or directly for ``--local`` jobs. Every user-controlled
value that is interpolated into that script -- the command's arguments,
``-x/--xargs`` override values, and job names (on the scheduler directive
lines and on any internal run command) -- must be quoted so that:

1. an argument like ``my run`` survives as a single token instead of being
   split into two, and
2. a value containing ``$(...)``, backticks, ``;``, ``&`` or ``>`` is inert
   rather than interpreted by the shell.

These are pure Tier A tests: no torch, no scheduler binaries. Scripts are
generated directly via ``Scheduler.launcher_script`` into ``tmp_path``; the
``--local`` case is additionally executed end-to-end (local runs work in the
sandbox).
"""
import os
import shlex

import pytest

from hpc_launcher.systems.system import GenericSystem
from hpc_launcher.schedulers.local import LocalScheduler
from hpc_launcher.schedulers.slurm import SlurmScheduler
from hpc_launcher.schedulers.flux import FluxScheduler
from hpc_launcher.schedulers.lsf import LSFScheduler


BATCH_SCHEDULERS = (SlurmScheduler, FluxScheduler, LSFScheduler)


def _run_command_line(script: str, command: str) -> str:
    """Return the (single) line of the generated script that runs ``command``."""
    candidates = [
        line
        for line in script.splitlines()
        # The real run line invokes the command; skip shell comments (which
        # include the scheduler directive lines such as ``#SBATCH``) and the
        # bookkeeping comments appended after the script body.
        if command in line and not line.lstrip().startswith("#")
    ]
    assert len(candidates) == 1, (
        f"expected exactly one run line invoking {command!r}, "
        f"found {len(candidates)}:\n{script}"
    )
    return candidates[0]


def test_args_with_spaces_roundtrip(tmp_path):
    """
    An argument containing a space must survive as a single token. Before the
    fix, ``--name my run`` was emitted verbatim and the shell split it into two
    arguments (``my`` and ``run``).
    """
    system = GenericSystem()
    command = "printf"
    args = ["--name", "my run", "plain", "two words"]

    for cls in BATCH_SCHEDULERS + (LocalScheduler,):
        scheduler = cls(nodes=1, procs_per_node=1, gpus_per_proc=0)
        script = scheduler.launcher_script(
            system, command, args, blocking=False, launch_dir=str(tmp_path)
        )

        run_line = _run_command_line(script, command)
        tokens = shlex.split(run_line)

        # The command's own arguments appear at the end of the run line, in
        # order, each as exactly one token. "my run" is one token, not split
        # into "my" and "run".
        arg_tokens = tokens[-len(args):]
        assert arg_tokens == args, (
            f"{cls.__name__}: argv round-trip mismatch\n"
            f"run line: {run_line}\ntokens: {tokens}"
        )


def test_metacharacters_inert(tmp_path):
    """
    Args and a job name containing shell metacharacters must be quoted in the
    generated script (a), and executing the ``--local`` script must not run any
    embedded command substitution while delivering the args verbatim (b).
    """
    canary = tmp_path / "pwned"
    payload = f"$(touch {canary})"
    metas = [payload, f"`touch {canary}`", "a;b", "x&y", "z>w"]
    args = ["--data", "my run"] + metas
    job_name = f"job {payload}"
    system = GenericSystem()

    # (a) Every metacharacter-bearing value is quoted in the generated script.
    for cls in BATCH_SCHEDULERS:
        scheduler = cls(
            nodes=1, procs_per_node=1, gpus_per_proc=0, job_name=job_name
        )
        script = scheduler.launcher_script(
            system, "printf", args, blocking=False, launch_dir=str(tmp_path)
        )
        for value in metas + [job_name]:
            assert shlex.quote(value) in script, (
                f"{cls.__name__}: {value!r} was not quoted in:\n{script}"
            )
        # The run line, when tokenized, keeps each metacharacter arg as a
        # single inert token (shlex does not perform expansion).
        run_line = _run_command_line(script, "printf")
        run_tokens = shlex.split(run_line)
        for value in metas:
            assert value in run_tokens, (
                f"{cls.__name__}: {value!r} was split/interpreted\n"
                f"run line: {run_line}"
            )

    # (b) Execute the local script end-to-end: no canary is created and every
    # argument is echoed back verbatim.
    assert not canary.exists()
    scheduler = LocalScheduler(
        nodes=1, procs_per_node=1, gpus_per_proc=0, job_name=job_name
    )
    _, folder_name = scheduler.create_launch_folder_name(
        "printf", "launch", str(tmp_path)
    )
    filename = scheduler.create_launch_folder(folder_name, True)
    result = scheduler.launch(
        system,
        folder_name,
        filename,
        "printf",
        # ``printf '%s\n' a b c`` prints each argument on its own line.
        ["%s\n"] + metas,
        blocking=True,
    )

    assert result.returncode == 0
    assert not canary.exists(), "command substitution executed (shell injection)"

    out_log = os.path.join(folder_name, "out.log")
    printed = open(out_log).read().splitlines()
    assert printed == metas, (
        f"arguments were not delivered verbatim: {printed!r} != {metas!r}"
    )


def test_metacharacters_in_job_name_folder_inert(tmp_path, monkeypatch):
    """
    A malicious job name flows into the auto-generated launch-folder name and
    thence into the ``cd`` line of the local script; that path must be quoted
    so it cannot inject shell syntax when the script runs.
    """
    # Run from within tmp_path so the auto-generated folder is created there.
    # Use a slash-free canary name so the (malicious) job name -- which becomes
    # part of the folder name -- stays a single path component.
    monkeypatch.chdir(tmp_path)
    canary_name = "pwned_folder"
    canary = tmp_path / canary_name

    scheduler = LocalScheduler(
        nodes=1,
        procs_per_node=1,
        gpus_per_proc=0,
        job_name=f"job$(touch {canary_name})",
    )
    # launch_dir="" -> the folder name embeds the (malicious) job name.
    _, folder_name = scheduler.create_launch_folder_name("printf", "launch", "")
    filename = scheduler.create_launch_folder(folder_name, True)
    result = scheduler.launch(
        GenericSystem(),
        folder_name,
        filename,
        "printf",
        ["%s\n", "hello"],
        blocking=True,
    )

    assert result.returncode == 0
    assert not canary.exists(), "job-name command substitution executed via cd"
    printed = open(os.path.join(folder_name, "out.log")).read().splitlines()
    assert printed == ["hello"]


@pytest.mark.parametrize("scheduler_class", BATCH_SCHEDULERS)
def test_job_name_quoted_in_headers(scheduler_class, tmp_path):
    """
    In batch (non-blocking) mode the job name is emitted on a scheduler
    directive line (``#SBATCH``/``# FLUX:``/``#BSUB``). A job name containing a
    space and a command substitution must appear quoted/inert there.
    """
    job_name = "my job$(id)"
    system = GenericSystem()

    scheduler = scheduler_class(
        nodes=1, procs_per_node=1, gpus_per_proc=0, job_name=job_name
    )
    script = scheduler.launcher_script(
        system, "printf", ["hi"], blocking=False, launch_dir=str(tmp_path)
    )

    prefix = scheduler.batch_script_prefix()
    directive_lines = [
        line for line in script.splitlines() if line.startswith(prefix)
    ]
    # Exactly one directive line carries the job name, and it is quoted.
    job_name_lines = [line for line in directive_lines if "job" in line]
    assert job_name_lines, f"no job-name directive line in:\n{script}"
    assert any(shlex.quote(job_name) in line for line in job_name_lines), (
        f"job name not quoted on the directive line: {job_name_lines}"
    )
