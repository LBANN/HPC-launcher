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
"""Unit tests for torchrun-hpc argument handling and per-node accounting."""
import pytest

from hpc_launcher.cli.torchrun_hpc import _detect_collision
from hpc_launcher.schedulers.slurm import SlurmScheduler
from hpc_launcher.schedulers.flux import FluxScheduler


@pytest.mark.parametrize(
    "flags",
    [
        ["--nnodes", "2"],
        ["--nnodes=2"],
        ["--nproc-per-node", "4"],
        ["--rdzv-endpoint", "host:1234"],
        ["--master-addr", "host"],
        ["--standalone"],
        ["--no-python"],
        ["--run-path"],
    ],
)
def test_collision_managed_flags_exit(flags):
    # A managed torchrun flag passed explicitly must abort the launch.
    with pytest.raises(SystemExit):
        _detect_collision(flags)


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--max-restarts", "3"],
        ["--monitor-interval", "1.0"],
        ["--tee", "1"],
        ["-r", "3"],  # torchrun --redirects short form; must pass through
        ["-m"],       # module mode is allowed to pass through
        ["--module"],
    ],
)
def test_collision_allowed_flags_pass(flags):
    # These flags are forwarded to torchrun untouched; no error.
    _detect_collision(flags)


def _clear(scheduler):
    scheduler.submit_only_args.clear()
    scheduler.run_only_args.clear()
    scheduler.common_launch_args.clear()
    scheduler.override_launch_args = None


def test_flux_torchrun_mode_one_task_per_node():
    scheduler = FluxScheduler(nodes=2, procs_per_node=4, gpus_per_proc=1)
    _clear(scheduler)
    scheduler.torchrun_mode = True
    scheduler.build_scheduler_specific_arguments(system=None, blocking=True)
    # One task per node (2), not nodes * procs_per_node (8).
    assert "-n2" in scheduler.common_launch_args
    assert "-n8" not in scheduler.common_launch_args
    # The single per-node task must see all of the node's GPUs.
    assert scheduler.run_only_args.get("--gpus-per-task") == "4"


def test_flux_default_mode_one_task_per_rank():
    scheduler = FluxScheduler(nodes=2, procs_per_node=4, gpus_per_proc=1)
    _clear(scheduler)
    scheduler.torchrun_mode = False
    scheduler.build_scheduler_specific_arguments(system=None, blocking=True)
    assert "-n8" in scheduler.common_launch_args
    assert scheduler.run_only_args.get("--gpus-per-task") == "1"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
