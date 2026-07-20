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

# bsub-only flags that must never reach jsrun (they live in
# submit_only_args, not common_launch_args/run_only_args).
BSUB_ONLY_FLAGS = ("-nnodes", "-q", "-J", "--shared-launch", "-W", "-G")

# jsrun flags the scheduler is expected to still emit on the internal run
# line (unaffected by the E1/E2 restructuring).
JSRUN_FLAGS = (
    "--nrs",
    "--rs_per_host",
    "--tasks_per_rs",
    "--launch_distribution",
    "--cpu_per_rs",
    "--gpu_per_rs",
)


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


def test_jsrun_line_has_no_bsub_flags(stub_system, tmp_path, monkeypatch):
    """
    The internal ``jsrun`` line written into the batch script by
    ``launcher_script`` must not contain any bsub-only flag, and must
    contain the genuine jsrun flags.
    """
    monkeypatch.delenv("LSB_HOSTS", raising=False)
    scheduler = _make_scheduler()

    script = scheduler.launcher_script(
        stub_system,
        "python",
        ["train.py"],
        blocking=False,
        launch_dir=str(tmp_path),
    )

    jsrun_lines = [
        line
        for line in script.splitlines()
        if "jsrun" in line and not line.lstrip().startswith("#")
    ]
    assert len(jsrun_lines) == 1, (
        f"expected exactly one internal jsrun line, found {len(jsrun_lines)}:\n{script}"
    )
    jsrun_line = jsrun_lines[0]

    for flag in BSUB_ONLY_FLAGS:
        assert (
            f" {flag} " not in f" {jsrun_line} "
        ), f"bsub-only flag {flag!r} leaked into the jsrun line: {jsrun_line}"

    for flag in JSRUN_FLAGS:
        assert flag in jsrun_line, f"expected jsrun flag {flag!r} missing from: {jsrun_line}"

    # The job name (a submit-only value) must not have leaked into the run
    # line either -- it only belongs on the #BSUB header line.
    assert "myjob" not in jsrun_line

    # Sanity check: the job name *is* present, quoted, on a #BSUB directive.
    directive_lines = [
        line for line in script.splitlines() if line.startswith("#BSUB")
    ]
    assert any("myjob" in line for line in directive_lines)
