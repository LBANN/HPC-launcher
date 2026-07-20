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
Regression tests for exit-code propagation (review finding C1).
"""
import subprocess
import sys

import pytest

from hpc_launcher.schedulers.local import LocalScheduler
from hpc_launcher.schedulers.scheduler import LaunchResult


@pytest.mark.parametrize("code", [0, 1, 3])
def test_blocking_exit_code_propagated(code, tmp_path):
    """
    A blocking ``launch --local`` run must exit with the child's exit code
    (C1). Previously the launcher discarded the child's return code and always
    exited 0, so failing jobs looked successful to shell pipelines / CI.
    """
    cmd = [
        sys.executable,
        "-m",
        "hpc_launcher.cli.launch",
        "--local",
        "-N1",
        "-n1",
        "-l",
        str(tmp_path),
        sys.executable,
        "-c",
        f"import sys; sys.exit({code})",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    assert proc.returncode == code, (
        f"launcher exited {proc.returncode}, expected {code}\n"
        f"stdout:\n{proc.stdout.decode(errors='replace')}\n"
        f"stderr:\n{proc.stderr.decode(errors='replace')}"
    )


def test_launch_result_unit(tmp_path, monkeypatch, stub_system):
    """
    Pure-unit form of C1: ``Scheduler.launch`` must surface the return code of
    ``run_process_with_live_output`` (the process that runs the generated
    launch script) inside the returned ``LaunchResult``.
    """
    # Do not actually execute the generated script; just assert the plumbing
    # carries whatever return code the runner produced.
    monkeypatch.setattr(
        "hpc_launcher.schedulers.scheduler.run_process_with_live_output",
        lambda *args, **kwargs: 3,
    )

    scheduler = LocalScheduler(nodes=1, procs_per_node=1, gpus_per_proc=0)
    _, folder_name = scheduler.create_launch_folder_name(
        sys.executable, "launch", str(tmp_path)
    )
    filename = scheduler.create_launch_folder(folder_name, True)

    result = scheduler.launch(
        stub_system,
        folder_name,
        filename,
        sys.executable,
        ["-c", "pass"],
        blocking=True,
    )

    assert isinstance(result, LaunchResult)
    assert result.returncode == 3
    assert result.job_id is None
