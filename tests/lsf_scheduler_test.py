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
Regression tests for the LSF backend (review findings E1, E2, E3):

- E1: bsub flags were built as single dict keys with an embedded space
  (``f"-nnodes {n}"``), which both launch paths execute as argv *without* a
  shell, so bsub saw the single literal token ``-nnodes 2`` instead of
  ``-nnodes`` and ``2`` as separate arguments. The ``-W`` value also had a
  trailing newline baked into the (buggy) embedded key.
- E2: bsub-only flags lived in ``common_launch_args``, which leaked them
  into the internal ``jsrun`` line written into the batch script.
- E3: ``LSFScheduler.get_job_id`` raised ``NotImplementedError`` instead of
  parsing bsub's "Job <N> is submitted..." output.

These are pure Tier A tests: no torch, no scheduler binaries. Commands and
scripts are constructed directly against a ``GenericSystem`` stub.
"""
from hpc_launcher.schedulers.lsf import LSFScheduler


def _make_scheduler():
    return LSFScheduler(
        nodes=2,
        procs_per_node=4,
        gpus_per_proc=1,
        job_name="myjob",
        queue="pbatch",
        time_limit=90,
    )


def test_bsub_argv_tokens_are_split(stub_system, monkeypatch):
    """
    ``launch_command`` must return the flag and its value as adjacent,
    separate argv elements (e.g. ``..., "-nnodes", "2", ...``), and no
    element may contain a space or a newline (the ``-W`` value used to have
    a trailing ``\\n`` baked in).
    """
    # Force the "not already inside an LSF allocation" code paths so the
    # command construction is deterministic regardless of the host running
    # the test.
    monkeypatch.delenv("LSB_HOSTS", raising=False)

    for blocking in (True, False):
        scheduler = _make_scheduler()
        cmd = scheduler.launch_command(stub_system, blocking=blocking)

        for token in cmd:
            assert " " not in token, f"argv token contains a space: {token!r} in {cmd}"
            assert "\n" not in token, f"argv token contains a newline: {token!r} in {cmd}"

        expected_pairs = {
            "-nnodes": "2",
            "-J": "myjob",
            "-q": "pbatch",
            "-W": "1:30",
        }
        for flag, value in expected_pairs.items():
            assert flag in cmd, f"{flag} missing from {cmd} (blocking={blocking})"
            idx = cmd.index(flag)
            assert cmd[idx + 1] == value, (
                f"{flag} not immediately followed by {value!r} in {cmd} "
                f"(blocking={blocking})"
            )

        # --shared-launch is a bare flag (no value).
        assert "--shared-launch" in cmd
