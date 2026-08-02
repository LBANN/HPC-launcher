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
Coverage for ``get_parallel_configuration()`` on the Flux and LSF schedulers.

Before this file, only ``SlurmScheduler.get_parallel_configuration`` was
exercised anywhere (``trampoline_device_test.py::test_cpu_gloo_two_ranks_init``
drives it through a real two-rank gloo job). Flux's and LSF's implementations
had zero coverage, including in CI: the only tests that would reach them
(``test_torchrun_hpc.py::test_launcher_multinode[flux-tcp-1-2]``,
``[lsf-tcp-1-2]``) are gated on ``shutil.which("flux")``/``shutil.which
("jsrun")``, both false on the CI runner, and even on a host that does have
Flux installed, running that end-to-end test never reaches this classmethod
(confirmed by an assertion-probe mutation during review: identical pass/fail
counts with and without the probe).

That absence matters more than usual right now: rank identity was just
reworked so that ``RANK``/``LOCAL_RANK``/``NODE_RANK``/``WORLD_SIZE`` are
computed by the trampoline from this exact function's return value (see
``tests/rank_identity_test.py``). A typo'd env-var name, a swapped tuple
field, or a division edge case here produces a wrong rank or world size for
every worker, silently, with no crash -- feeding straight into
``init_process_group``.

Scope, to avoid duplicating ``rank_identity_test.py``: that file drives the
trampoline end to end (always through the Slurm backend, per its own
comment) to check what a *task* ends up publishing. This file stays one
level down -- it calls ``FluxScheduler.get_parallel_configuration()`` and
``LSFScheduler.get_parallel_configuration()`` directly, with the scheduler's
own environment variables monkeypatched, the same technique
``trampoline_device_test.py`` uses for Slurm's ``SLURM_*`` variables. These
are plain classmethods over ``os.getenv`` with no scheduler binary
dependency, so nothing here needs Flux or LSF installed.

Expected values below are derived from what each scheduler's environment
variables mean (Flux's and OpenMPI's own documentation of
``FLUX_JOB_SIZE``/``FLUX_TASK_RANK``/``FLUX_TASK_LOCAL_ID``/
``FLUX_JOB_NNODES`` and ``OMPI_COMM_WORLD_SIZE``/``_RANK``/
``_LOCAL_RANK``/``_LOCAL_SIZE``), not from what the code currently returns.
"""
import pytest

from hpc_launcher.schedulers.flux import FluxScheduler
from hpc_launcher.schedulers.lsf import LSFScheduler


# ---------------------------------------------------------------------------
# Flux: (FLUX_JOB_SIZE, FLUX_TASK_RANK, FLUX_TASK_LOCAL_ID, FLUX_JOB_NNODES)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "job_size,task_rank,task_local_id,job_nnodes,expected",
    [
        # Single-rank job: the degenerate case every other shape reduces to.
        (1, 0, 0, 1, (1, 0, 1, 0)),
        # 4 tasks over 2 nodes, evenly split. Rank 3 is the second task
        # (local id 1) on its node -- exactly the shape rank identity work
        # cares about, since it is where a rank/local-rank mixup would show.
        (4, 3, 1, 2, (4, 3, 2, 1)),
        # 8 tasks over 4 nodes (2 per node): local_world_size must come out
        # to 2, not the job-wide task count or the node count.
        (8, 5, 1, 4, (8, 5, 2, 1)),
    ],
)
def test_flux_get_parallel_configuration(monkeypatch, job_size, task_rank,
                                         task_local_id, job_nnodes,
                                         expected):
    """
    ``FluxScheduler.get_parallel_configuration()`` reads
    ``FLUX_JOB_SIZE``/``FLUX_TASK_RANK``/``FLUX_TASK_LOCAL_ID``/
    ``FLUX_JOB_NNODES`` directly (``FLUX_TASK_LOCAL_ID`` *is* the node-local
    rank already -- no derivation needed there) and derives
    ``local_world_size`` as ``world_size // nodes_per_job``.
    """
    monkeypatch.setenv("FLUX_JOB_SIZE", str(job_size))
    monkeypatch.setenv("FLUX_TASK_RANK", str(task_rank))
    monkeypatch.setenv("FLUX_TASK_LOCAL_ID", str(task_local_id))
    monkeypatch.setenv("FLUX_JOB_NNODES", str(job_nnodes))

    assert FluxScheduler.get_parallel_configuration() == expected


def test_flux_local_world_size_truncates_on_uneven_distribution(monkeypatch):
    """
    Documents current behavior rather than asserting a fix: 5 tasks over 2
    nodes cannot split evenly (e.g. 3 on one node, 2 on the other), but
    ``local_world_size`` is computed as ``world_size // nodes_per_job`` with
    no way to see the actual per-node count, so it truncates to 2 for every
    rank regardless of which node it is actually sharing with two others.

    This is not a Flux-specific defect: ``SlurmScheduler.get_parallel_
    configuration`` (``slurm.py:170-188``) does the exact same integer
    division for the identical reason, and its only existing test
    (``trampoline_device_test.py::test_cpu_gloo_two_ranks_init``) uses 2
    tasks on 1 node, which divides evenly and never exercises this edge.
    Recorded here, not fixed here: whether an uneven split
    should raise instead of truncating is a design decision for the
    schedulers generally, not something to change under one backend as a
    side effect of adding coverage.
    """
    monkeypatch.setenv("FLUX_JOB_SIZE", "5")
    monkeypatch.setenv("FLUX_TASK_RANK", "4")
    monkeypatch.setenv("FLUX_TASK_LOCAL_ID", "0")
    monkeypatch.setenv("FLUX_JOB_NNODES", "2")

    world_size, rank, local_world_size, local_rank = (
        FluxScheduler.get_parallel_configuration())
    assert (world_size, rank, local_world_size, local_rank) == (5, 4, 2, 0)


@pytest.mark.parametrize(
    "missing_var",
    ["FLUX_JOB_SIZE", "FLUX_TASK_RANK", "FLUX_TASK_LOCAL_ID",
     "FLUX_JOB_NNODES"],
)
def test_flux_missing_variable_raises_naming_it(monkeypatch, missing_var):
    """
    Each of the four variables is required individually; a job launched
    outside of a real Flux task (or under a Flux version that renamed one)
    must fail with a message naming the specific variable that is missing,
    not a generic KeyError/TypeError from the ``int()`` conversion below it.
    """
    env_vars = {
        "FLUX_JOB_SIZE": "4",
        "FLUX_TASK_RANK": "1",
        "FLUX_TASK_LOCAL_ID": "1",
        "FLUX_JOB_NNODES": "2",
    }
    for name, value in env_vars.items():
        if name == missing_var:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    with pytest.raises(Exception, match=missing_var):
        FluxScheduler.get_parallel_configuration()


# ---------------------------------------------------------------------------
# LSF: (OMPI_COMM_WORLD_SIZE, _RANK, _LOCAL_RANK, _LOCAL_SIZE)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "world_size,rank,local_rank,local_size,expected",
    [
        # Single-rank job.
        (1, 0, 0, 1, (1, 0, 1, 0)),
        # The configuration verified directly against a real jsrun-style
        # environment during review: 9 ranks total, local size 3.
        (9, 5, 1, 3, (9, 5, 3, 1)),
        # Uneven distribution across nodes (5 ranks, e.g. 3 + 2): unlike
        # Flux/Slurm, LSF reads OMPI_COMM_WORLD_LOCAL_SIZE directly from the
        # environment instead of deriving it by dividing the job-wide task
        # count by the node count, so this is reported correctly rather than
        # truncated -- the case the Flux test above documents as wrong.
        (5, 4, 0, 2, (5, 4, 2, 0)),
    ],
)
def test_lsf_get_parallel_configuration(monkeypatch, world_size, rank,
                                        local_rank, local_size, expected):
    """
    ``LSFScheduler.get_parallel_configuration()`` reads all four values
    directly from OpenMPI's ``OMPI_COMM_WORLD_*`` variables -- notably
    ``OMPI_COMM_WORLD_LOCAL_SIZE`` for ``local_world_size``, with no
    ``world_size // nodes_per_job`` division at all.
    """
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", str(world_size))
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", str(rank))
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", str(local_rank))
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_SIZE", str(local_size))

    assert LSFScheduler.get_parallel_configuration() == expected


@pytest.mark.parametrize(
    "missing_var",
    ["OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_RANK",
     "OMPI_COMM_WORLD_LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_SIZE"],
)
def test_lsf_missing_variable_raises_naming_it(monkeypatch, missing_var):
    """
    Same contract as the Flux case above: jsrun sets all four
    ``OMPI_COMM_WORLD_*`` variables together, but if one is absent (wrong
    launcher, unset by a wrapper script, renamed by an OpenMPI version) the
    error must name it rather than failing generically.
    """
    env_vars = {
        "OMPI_COMM_WORLD_SIZE": "4",
        "OMPI_COMM_WORLD_RANK": "1",
        "OMPI_COMM_WORLD_LOCAL_RANK": "1",
        "OMPI_COMM_WORLD_LOCAL_SIZE": "2",
    }
    for name, value in env_vars.items():
        if name == missing_var:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    with pytest.raises(Exception, match=missing_var):
        LSFScheduler.get_parallel_configuration()
