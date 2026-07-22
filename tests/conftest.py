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
Shared pytest fixtures for the hpc_launcher test suite.
"""
import os

import pytest

from hpc_launcher.schedulers.slurm import SlurmScheduler
from hpc_launcher.systems.system import GenericSystem


def require_torch():
    """
    Import and return the ``torch`` module, or skip the calling test if it
    is not installed.

    This is the shared replacement for the ``try: import torch / except
    (ImportError, ModuleNotFoundError): pytest.skip(...)`` pattern used by
    the torch-guarded tests. When the environment variable
    ``HPC_LAUNCHER_CI_REQUIRE_TORCH`` is set (CI sets it on every matrix
    leg once torch is installed everywhere), a missing torch fails the
    test instead of silently skipping it. Without this, a CI leg that
    fails to install torch could report fully green while every
    torch-dependent test was quietly skipped.
    """
    try:
        import torch
    except (ImportError, ModuleNotFoundError):
        if os.environ.get("HPC_LAUNCHER_CI_REQUIRE_TORCH"):
            pytest.fail(
                "torch is not importable, but HPC_LAUNCHER_CI_REQUIRE_TORCH "
                "is set: torch is required to be installed in this "
                "environment, so this cannot be silently skipped."
            )
        pytest.skip("torch not found")
    return torch


@pytest.fixture
def stub_system() -> GenericSystem:
    """
    Returns a directly-constructed system object with fixed, explicit
    parameters and no host autodetection, for use by command-construction
    tests that must not depend on the machine running the test.
    """
    return GenericSystem()


@pytest.fixture(autouse=True)
def _no_shared_scheduler_or_system_state():
    """
    Autouse tripwire for shared-mutable-state regressions.

    Before every test, verify that a freshly constructed ``Scheduler`` starts
    with empty argument dictionaries and that a freshly constructed
    ``System`` starts with an empty auxiliary environment-variable list. If
    either of these were ever reverted to class-level (shared) state, one of
    these constructions would come back non-empty as soon as any earlier test
    in the process had populated it, and this fixture fails loudly and
    immediately instead of letting the pollution silently corrupt later
    tests.

    Deliberately cheap: no subprocesses, no torch import, no autodetection.
    """
    scheduler = SlurmScheduler(nodes=1, procs_per_node=1, gpus_per_proc=0)
    assert scheduler.submit_only_args == {}
    assert scheduler.run_only_args == {}
    assert scheduler.common_launch_args == {}

    system = GenericSystem()
    assert system._aux_env_list == []

    yield
