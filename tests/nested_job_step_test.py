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
Launching from inside an existing allocation must produce a nested job step,
not a request for a new allocation.

Two cooperating behaviors are covered:

- CLI argument processing: with no job-size flag given (--nodes /
  --gpus-at-least / --gpumem-at-least), the node count is inherited from the
  enclosing allocation instead of ``validate_arguments`` rejecting the
  invocation.
- Slurm command construction: for a blocking launch inside an allocation
  (``SLURM_JOB_ID`` set), allocation-selection flags
  (--partition/--account/--reservation) are dropped so srun runs a job step
  in the current allocation, and a node request exceeding the allocation is
  rejected upfront rather than silently pending as a second allocation.

No scheduler binaries are required: the allocation is simulated through the
environment variables the probes read (``SLURM_JOB_ID``,
``SLURM_JOB_NUM_NODES``).
"""
import argparse
import logging

import pytest
from unittest.mock import patch

from hpc_launcher.cli import common_args
from hpc_launcher.schedulers import num_nodes_in_current_allocation
from hpc_launcher.schedulers.slurm import SlurmScheduler
from hpc_launcher.systems.system import GenericSystem, SystemParams

_LOGGER = logging.getLogger(__name__)


def _generic_system() -> GenericSystem:
    """
    A GenericSystem shaped the way ``autodetect_current_system`` returns it
    on an unrecognized host: with a populated "auto" queue, which
    ``configure_launch`` requires (a bare GenericSystem has no system
    parameters and is never returned by autodetection).
    """
    sys = GenericSystem()
    sys.system_params = {
        "auto": SystemParams(
            cores_per_node=4,
            gpus_per_node=0,
            numa_domains=1,
            scheduler="slurm",
        )
    }
    sys.default_queue = "auto"
    return sys

_SLURM_ALLOC_ENV = {
    "SLURM_JOB_ID": "424242",
    "SLURM_JOB_NUM_NODES": "4",
}


def _clear_alloc_env(monkeypatch):
    """Make the test host look like it is not inside any allocation."""
    for var in (
        "SLURM_JOB_ID",
        "SLURM_JOB_NUM_NODES",
        "FLUX_URI",
        "LLNL_NUM_COMPUTE_NODES",
    ):
        monkeypatch.delenv(var, raising=False)


def _set_slurm_alloc_env(monkeypatch, nodes: int = 4):
    _clear_alloc_env(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", _SLURM_ALLOC_ENV["SLURM_JOB_ID"])
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", str(nodes))


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    common_args.setup_arguments(parser)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Scheduler-agnostic allocation probe
# ---------------------------------------------------------------------------
def test_probe_outside_allocation(monkeypatch):
    _clear_alloc_env(monkeypatch)
    assert num_nodes_in_current_allocation() is None


def test_probe_inside_slurm_allocation(monkeypatch):
    _set_slurm_alloc_env(monkeypatch, nodes=4)
    assert num_nodes_in_current_allocation() == 4


def test_probe_inside_lsf_allocation(monkeypatch):
    _clear_alloc_env(monkeypatch)
    monkeypatch.setenv("LLNL_NUM_COMPUTE_NODES", "2")
    assert num_nodes_in_current_allocation() == 2


# ---------------------------------------------------------------------------
# CLI: job size is inherited from the allocation
# ---------------------------------------------------------------------------
@patch(
    "hpc_launcher.systems.autodetect.autodetect_current_system",
    new=_generic_system,
)
def test_no_size_flags_inherits_allocation_nodes(monkeypatch):
    _set_slurm_alloc_env(monkeypatch, nodes=4)
    args = _parse([])
    common_args.process_arguments(args, _LOGGER)
    assert args.nodes == 4


@patch(
    "hpc_launcher.systems.autodetect.autodetect_current_system",
    new=_generic_system,
)
def test_explicit_size_flags_still_win_inside_allocation(monkeypatch):
    _set_slurm_alloc_env(monkeypatch, nodes=4)
    args = _parse(["--nodes", "2"])
    common_args.process_arguments(args, _LOGGER)
    assert args.nodes == 2


def test_no_size_flags_outside_allocation_still_rejected(monkeypatch):
    _clear_alloc_env(monkeypatch)
    args = _parse([])
    with pytest.raises(ValueError, match="--nodes"):
        common_args.process_arguments(args, _LOGGER)


@patch(
    "hpc_launcher.systems.autodetect.autodetect_current_system",
    new=_generic_system,
)
def test_local_does_not_inherit_allocation_nodes(monkeypatch):
    # --local runs a single process without a scheduler; it must not be
    # silently sized up to the enclosing allocation. With no size flags it
    # keeps failing validation exactly as outside an allocation.
    _set_slurm_alloc_env(monkeypatch, nodes=4)
    args = _parse(["--local"])
    with pytest.raises(ValueError, match="--nodes"):
        common_args.process_arguments(args, _LOGGER)


# ---------------------------------------------------------------------------
# Slurm: blocking launch inside an allocation is a nested job step
# ---------------------------------------------------------------------------
def test_blocking_in_allocation_drops_allocation_flags(
    stub_system, monkeypatch
):
    _set_slurm_alloc_env(monkeypatch, nodes=4)
    scheduler = SlurmScheduler(
        nodes=2,
        procs_per_node=1,
        gpus_per_proc=0,
        queue="pbatch",
        account="guests",
        reservation="dat123",
    )
    cmd = scheduler.launch_command(stub_system, blocking=True)
    assert cmd[0] == "srun"
    for flag in ("--partition", "--account", "--reservation"):
        assert not any(
            c.startswith(flag) for c in cmd
        ), f"{flag} must not be passed to a nested job step: {cmd}"
    # The step still carries its within-allocation shape.
    assert "--nodes=2" in cmd
    assert "--ntasks=2" in cmd
    # Since Slurm 20.11 job steps get exclusive CPUs by default, so a second
    # concurrent launch in the same allocation would hang on "step creation
    # temporarily disabled, retrying (Requested nodes are busy)" without
    # --overlap.
    assert "--overlap" in cmd


def test_blocking_outside_allocation_has_no_overlap(stub_system, monkeypatch):
    _clear_alloc_env(monkeypatch)
    scheduler = SlurmScheduler(nodes=2, procs_per_node=1, gpus_per_proc=0)
    cmd = scheduler.launch_command(stub_system, blocking=True)
    assert "--overlap" not in cmd


def test_overlap_can_be_removed_with_override(stub_system, monkeypatch):
    _set_slurm_alloc_env(monkeypatch, nodes=4)
    scheduler = SlurmScheduler(nodes=2, procs_per_node=1, gpus_per_proc=0)
    scheduler.override_launch_args = {"~--overlap": ""}
    cmd = scheduler.launch_command(stub_system, blocking=True)
    assert "--overlap" not in cmd


def test_cpus_per_task_emitted(stub_system, monkeypatch):
    _clear_alloc_env(monkeypatch)
    scheduler = SlurmScheduler(
        nodes=2, procs_per_node=1, gpus_per_proc=0, cpus_per_task=8
    )
    cmd = scheduler.launch_command(stub_system, blocking=True)
    assert "--cpus-per-task=8" in cmd


def test_cpus_per_task_step_gets_exact_footprint_not_overlap(
    stub_system, monkeypatch
):
    # With an explicit CPU footprint, srun's --cpus-per-task (which implies
    # --exact) lets Slurm pack disjoint concurrent steps side by side, so
    # --overlap must not be added: overlapping would put the steps on the
    # same CPUs instead of next to each other.
    _set_slurm_alloc_env(monkeypatch, nodes=4)
    scheduler = SlurmScheduler(
        nodes=2, procs_per_node=1, gpus_per_proc=0, cpus_per_task=8
    )
    cmd = scheduler.launch_command(stub_system, blocking=True)
    assert "--cpus-per-task=8" in cmd
    assert "--overlap" not in cmd


def test_exclusive_step_suppresses_overlap(stub_system, monkeypatch):
    # An explicitly exclusive step asks for dedicated resources, which
    # --overlap would contradict.
    _set_slurm_alloc_env(monkeypatch, nodes=4)
    scheduler = SlurmScheduler(
        nodes=2, procs_per_node=1, gpus_per_proc=0, exclusive=True
    )
    cmd = scheduler.launch_command(stub_system, blocking=True)
    assert "--overlap" not in cmd
    assert any(c.startswith("--exclusive") for c in cmd)


def test_blocking_outside_allocation_keeps_allocation_flags(
    stub_system, monkeypatch
):
    _clear_alloc_env(monkeypatch)
    scheduler = SlurmScheduler(
        nodes=2,
        procs_per_node=1,
        gpus_per_proc=0,
        queue="pbatch",
        account="guests",
    )
    cmd = scheduler.launch_command(stub_system, blocking=True)
    assert "--partition=pbatch" in cmd
    assert "--account=guests" in cmd


def test_nonblocking_in_allocation_keeps_allocation_flags(
    stub_system, monkeypatch
):
    # sbatch from inside an allocation is a deliberate new job submission;
    # the partition/account selection must survive.
    _set_slurm_alloc_env(monkeypatch, nodes=4)
    scheduler = SlurmScheduler(
        nodes=2,
        procs_per_node=1,
        gpus_per_proc=0,
        queue="pbatch",
        account="guests",
    )
    cmd = scheduler.launch_command(stub_system, blocking=False)
    assert cmd[0] == "sbatch"
    assert "--partition=pbatch" in cmd
    assert "--account=guests" in cmd


def test_oversized_step_in_allocation_rejected(stub_system, monkeypatch):
    _set_slurm_alloc_env(monkeypatch, nodes=4)
    scheduler = SlurmScheduler(nodes=8, procs_per_node=1, gpus_per_proc=0)
    with pytest.raises(ValueError, match="new .*allocation|allocation of"):
        scheduler.launch_command(stub_system, blocking=True)


def test_step_filling_allocation_accepted(stub_system, monkeypatch):
    _set_slurm_alloc_env(monkeypatch, nodes=4)
    scheduler = SlurmScheduler(nodes=4, procs_per_node=1, gpus_per_proc=0)
    cmd = scheduler.launch_command(stub_system, blocking=True)
    assert "--nodes=4" in cmd
