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
How torchrun-hpc divides a command line between itself and the user's script.

torchrun-hpc owns the flags before the training script and the topology they
describe; everything after the script belongs to the script. Getting that
boundary wrong is quiet rather than loud -- the job still launches, just with
a topology the user did not ask for and a script missing an argument -- so
these are the cases worth pinning down.

Every test here drives the real entry point with real argv and inspects the
generated batch script. That is deliberate: the boundary is decided by the
interaction between the launcher's parser, the positional split, and what
reaches the run command, so a test that calls an internal helper with a
hand-tokenized list can pass while the assembled command line is wrong.
"""
import os
import subprocess
import sys

import pytest

from conftest import require_torch

TORCHRUN = [sys.executable, "-m", "hpc_launcher.cli.torchrun_hpc"]


def _generate(tmp_path, *cli_args, job_name="job"):
    """
    Run torchrun-hpc for a batch Slurm job, stopping after the script is
    written, and return ``(completed_process, script_text)``.

    ``--setup-only`` means nothing is submitted, so this needs no scheduler
    installed; Slurm is named explicitly so the emitted header states the
    allocation in a form the assertions can read.
    """
    launch_dir = tmp_path / job_name
    proc = subprocess.run(
        TORCHRUN
        + [
            "--scheduler", "slurm",
            "--bg", "--setup-only",
            "-l", str(launch_dir),
        ]
        + list(cli_args),
        cwd=str(tmp_path),
        capture_output=True,
        universal_newlines=True,
    )
    script_file = launch_dir / "launch.sh"
    script = script_file.read_text() if script_file.exists() else ""
    return proc, script


def _run_line(script):
    """The line of the generated script that runs the trampoline."""
    matches = [
        line for line in script.splitlines()
        if "torchrun_hpc_trampoline.py" in line and not line.startswith("#")
    ]
    assert len(matches) == 1, f"expected exactly one run line, got {matches}"
    return matches[0]


def _write_train_script(tmp_path):
    train = tmp_path / "train.py"
    train.write_text("print('hello')\n")
    return train


# ---------------------------------------------------------------------------
# Flags after the training script belong to the training script
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "script_args",
    [
        # Spellings of a launcher-owned flag, placed after the script where
        # they are the *script's* arguments and must not be reinterpreted.
        ["-N", "4"],
        ["--nodes", "4"],
        ["-n", "8"],
        # A flag the launcher does not define at all still belongs to the
        # script rather than being rejected as unknown.
        ["--epochs", "10"],
    ],
)
def test_script_arguments_are_not_consumed_by_the_launcher(
        tmp_path, script_args):
    """
    ``torchrun-hpc -N1 -n2 train.py -N 4`` requests one node and hands the
    script ``-N 4``.

    Both halves matter and they fail together: a parser that keeps scanning
    for its own options past the script both changes the allocation and
    swallows an argument the script needed, with no diagnostic either way.
    """
    require_torch()
    train = _write_train_script(tmp_path)

    proc, script = _generate(tmp_path, "-N1", "-n2", str(train), *script_args)

    assert proc.returncode == 0, proc.stderr

    # The launcher's own -N1 decides the allocation.
    assert "#SBATCH --nodes=1" in script, (
        f"the allocation did not come from the launcher's own -N1:\n{script}"
    )
    assert "#SBATCH --nodes=4" not in script, (
        f"an argument after the training script changed the allocation:\n{script}"
    )

    # ...and the script still receives its arguments, in order, after its path.
    run_line = _run_line(script)
    _, _, tail = run_line.partition(str(train))
    assert tail.split() == script_args, (
        f"the script's arguments {script_args} did not survive as its own; "
        f"run line was:\n{run_line}"
    )


def test_launcher_flags_before_the_script_still_apply(tmp_path):
    """
    The complement of the case above: before the script, the same flag is the
    launcher's own and must shape the allocation. Without this, a launcher
    that simply ignored the flag everywhere would satisfy the test above.
    """
    require_torch()
    train = _write_train_script(tmp_path)

    proc, script = _generate(tmp_path, "-N", "4", "-n", "2", str(train))

    assert proc.returncode == 0, proc.stderr
    assert "#SBATCH --nodes=4" in script, script
    assert "#SBATCH --ntasks-per-node=2" in script, script

    # With no arguments of its own, the script's path ends the run line.
    assert _run_line(script).rstrip().endswith(str(train)), _run_line(script)


# ---------------------------------------------------------------------------
# Abbreviated flags behave exactly like the spelling they abbreviate
# ---------------------------------------------------------------------------
def test_launcher_flag_abbreviations_are_accepted(tmp_path):
    """
    argparse accepts any unambiguous prefix of an option, so ``--nod 4`` is
    ``--nodes 4``. That is the premise the next two tests rest on: users can
    and do write abbreviations, so anything that inspects flags by exact
    spelling will disagree with what was actually parsed.
    """
    require_torch()
    train = _write_train_script(tmp_path)

    proc, script = _generate(tmp_path, "--nod", "4", "-n", "2", str(train))

    assert proc.returncode == 0, proc.stderr
    assert "#SBATCH --nodes=4" in script, script


def test_module_flag_spellings_are_equivalent(tmp_path):
    """
    ``-m``, ``--module`` and the abbreviation ``--mod`` must all produce the
    same invocation.

    Deciding module mode from one exact spelling while the parser accepts
    several is the failure this guards: the flag registers as "not module
    mode" for the launcher but still reaches the command line, and the job
    ends up trying to import the trampoline itself as a module by absolute
    path -- broken on every node, at run time rather than launch time.
    """
    require_torch()

    run_lines = {}
    for spelling in ("-m", "--module", "--mod"):
        job = "job" + spelling.strip("-")
        proc, script = _generate(tmp_path, "-N1", "-n2", spelling,
                                 "mypkg.train", job_name=job)
        assert proc.returncode == 0, f"{spelling}: {proc.stderr}"
        # The launch directory differs per spelling; normalize it away so the
        # comparison is about the shape of the invocation.
        run_lines[spelling] = _run_line(script).replace(
            str(tmp_path / job), "<LAUNCH_DIR>")

    assert len(set(run_lines.values())) == 1, (
        "module flag spellings produced different invocations:\n"
        + "\n".join(f"  {k}: {v}" for k, v in run_lines.items())
    )


def test_module_mode_runs_the_trampoline_by_path(tmp_path):
    """
    In module mode the trampoline is still executed as a script, and ``-m``
    applies to the user's module -- ``python <trampoline> -m mypkg.train``.

    The shape that must not appear is ``python -m <trampoline path>``, which
    asks Python to import a filesystem path as a module and fails everywhere.
    """
    require_torch()

    proc, script = _generate(tmp_path, "-N1", "-n2", "--mod", "mypkg.train")
    assert proc.returncode == 0, proc.stderr

    tokens = _run_line(script).split()
    trampoline = [t for t in tokens if t.endswith("torchrun_hpc_trampoline.py")]
    assert len(trampoline) == 1, tokens
    index = tokens.index(trampoline[0])

    assert tokens[index - 1] != "-m", (
        f"the trampoline is being imported as a module by path: {tokens}"
    )
    assert tokens[index + 1:index + 3] == ["-m", "mypkg.train"], (
        f"module mode did not reach the trampoline as '-m mypkg.train': {tokens}"
    )


@pytest.mark.parametrize(
    "conflicting_flag",
    [
        # torchrun spellings for the same topology the launcher already
        # allocated, including the abbreviation an exact-match check misses.
        "--nnode=2",
        "--nnodes=2",
        "--nproc_per_node=8",
    ],
)
def test_flag_cannot_silently_contradict_the_allocation(tmp_path,
                                                        conflicting_flag):
    """
    The scheduler allocation is authoritative: a job asked for one node must
    not end up being told to run on two.

    There are two defensible ways to honor that -- reject the flag, or accept
    it and reconcile it with the allocation -- so the assertion allows either
    and rules out the third outcome, where the flag is quietly passed along
    and the job runs on a topology the scheduler never allocated. Written as
    a rejection-only test this would start failing the day flag forwarding is
    added, even if the forwarding were correct.
    """
    require_torch()
    train = _write_train_script(tmp_path)

    proc, script = _generate(tmp_path, "-N1", "-n2", conflicting_flag,
                             str(train))

    if proc.returncode != 0:
        # Rejected. It must be a clean, actionable error naming the flag.
        assert "Traceback" not in proc.stderr, proc.stderr
        flag_name = conflicting_flag.split("=")[0]
        assert flag_name in proc.stderr, (
            f"rejection did not name {flag_name}:\n{proc.stderr}"
        )
        return

    # Accepted. The allocation must still be the one the launcher requested,
    # and the flag must not have been passed through to contradict it.
    assert "#SBATCH --nodes=1" in script, (
        f"{conflicting_flag} changed the allocation:\n{script}"
    )
    run_line = _run_line(script)
    assert conflicting_flag not in run_line, (
        f"{conflicting_flag} was forwarded verbatim and contradicts the "
        f"one-node allocation:\n{run_line}"
    )
