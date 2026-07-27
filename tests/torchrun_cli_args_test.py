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


def _generate(tmp_path, *cli_args):
    """
    Run torchrun-hpc for a batch Slurm job, stopping after the script is
    written, and return ``(completed_process, script_text)``.

    ``--setup-only`` means nothing is submitted, so this needs no scheduler
    installed; Slurm is named explicitly so the emitted header states the
    allocation in a form the assertions can read.
    """
    launch_dir = tmp_path / "job"
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
