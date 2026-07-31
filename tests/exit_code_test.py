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
Regression tests for exit-code propagation and Ctrl-C /
SIGINT handling.
"""
import signal
import subprocess
import sys
import threading
import time
import types

import pytest

from hpc_launcher.schedulers import scheduler as scheduler_mod
from hpc_launcher.schedulers.local import LocalScheduler
from hpc_launcher.schedulers.slurm import SlurmScheduler
from hpc_launcher.schedulers.scheduler import LaunchResult


@pytest.mark.parametrize("code", [0, 1, 3])
def test_blocking_exit_code_propagated(code, tmp_path):
    """
    A blocking ``launch --local`` run must exit with the child's exit code.
    Previously the launcher discarded the child's return code and always
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
    Pure-unit form: ``Scheduler.launch`` must surface the return code of
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


# ---------------------------------------------------------------------------
# Non-blocking (batch) submission: only the exit status decides success
# ---------------------------------------------------------------------------

# A routine, *non-fatal* sbatch diagnostic. Slurm prints this on an otherwise
# successful submission whenever --nodes/--ntasks/--ntasks-per-node are
# mutually inconsistent; the launcher always passes all three, and ``-x`` lets
# a user desync them (e.g. ``-N2 --procs-per-node 3 -x--ntasks=8``). Site
# ``job_submit`` Lua plugins print bank/QoS notices the same way.
_SBATCH_WARNING = (
    b"sbatch: Warning: can't honor --ntasks-per-node set to 3 which doesn't "
    b"match the requested tasks 8 with the number of requested nodes 2. "
    b"Ignoring --ntasks-per-node.\n"
)
_SBATCH_SUCCESS = b"Submitted batch job 987654\n"


def _submit_batch(monkeypatch, tmp_path, stub_system, returncode, stdout, stderr):
    """
    Drive ``Scheduler.launch`` down its non-blocking (batch submission)
    branch with a stubbed submit command that produces exactly the given exit
    status and streams, and return the resulting ``LaunchResult``.

    Only ``subprocess.run`` as seen by the scheduler module is replaced, so
    everything up to and including the submit-command construction is the
    real code path.
    """

    def _fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

    monkeypatch.setattr(
        scheduler_mod, "subprocess", types.SimpleNamespace(run=_fake_run)
    )

    scheduler = SlurmScheduler(nodes=1, procs_per_node=1, gpus_per_proc=0)
    _, folder_name = scheduler.create_launch_folder_name(
        "hostname", "launch", str(tmp_path)
    )
    filename = scheduler.create_launch_folder(folder_name, False)

    return scheduler.launch(
        stub_system,
        folder_name,
        filename,
        "hostname",
        [],
        blocking=False,
    )


def test_batch_submission_stderr_is_not_a_failure(
    tmp_path, monkeypatch, stub_system, capfd
):
    """
    A submit command that exits 0 has *succeeded*, whatever it wrote to
    stderr. The non-blocking path used to treat any non-empty stderr as an
    error (``if process.returncode or process.stderr``), which produced the
    self-contradictory "exited with error code 0", skipped ``get_job_id``,
    discarded the submit command's stdout entirely, and -- once the
    exit-code fix added ``returncode or 1`` -- made ``launch --bg`` exit 1.

    The damage is not cosmetic: the job really is queued and consuming
    allocation, but its ID is never reported, so it cannot be cancelled,
    monitored, or chained with ``--dependency``.
    """
    result = _submit_batch(
        monkeypatch,
        tmp_path,
        stub_system,
        returncode=0,
        stdout=_SBATCH_SUCCESS,
        stderr=_SBATCH_WARNING,
    )

    assert isinstance(result, LaunchResult)
    assert result.job_id == "987654", (
        "the job ID was discarded because the successful submission printed "
        "a warning to stderr"
    )
    # No exit code yet: the job is still running. ``launch`` exits 0 on this.
    assert result.returncode is None, result.returncode

    # The warning is still shown to the user -- forwarded, just not construed
    # as failure -- and no bogus error line accompanies it.
    captured = capfd.readouterr()
    assert "can't honor --ntasks-per-node" in captured.err
    assert "error code 0" not in captured.err


def test_batch_submission_failure_reports_failure(
    tmp_path, monkeypatch, stub_system, capfd
):
    """
    The exit-code invariant, which the stderr fix must not undo: a submit
    command that exits non-zero must report failure, with a non-``None`` non-zero
    return code (``LaunchResult.returncode`` of ``None`` means "submitted,
    still running" and makes the CLI exit 0), and no job ID.

    Both streams reach the user: a genuine failure often explains itself on
    stdout, which the old error path threw away.
    """
    result = _submit_batch(
        monkeypatch,
        tmp_path,
        stub_system,
        returncode=1,
        stdout=b"sbatch: some context on stdout\n",
        stderr=b"sbatch: error: Batch job submission failed: Invalid account\n",
    )

    assert result.job_id is None
    assert result.returncode is not None, "a failed submission must not exit 0"
    assert result.returncode == 1

    captured = capfd.readouterr()
    assert "Batch job submission failed" in captured.err
    assert "some context on stdout" in captured.out + captured.err


def test_batch_submission_quiet_success(tmp_path, monkeypatch, stub_system):
    """
    The plain case, for symmetry with the two above: exit 0 and nothing on
    stderr yields the job ID and no exit code.
    """
    result = _submit_batch(
        monkeypatch,
        tmp_path,
        stub_system,
        returncode=0,
        stdout=_SBATCH_SUCCESS,
        stderr=b"",
    )

    assert result.job_id == "987654"
    assert result.returncode is None


def test_sigint_kills_child(tmp_path):
    """
    After the launcher receives SIGINT, its scheduler child (and any
    grandchildren it spawned) must be killed within a bounded time and the
    launcher must exit non-zero (130 = 128 + SIGINT).

    Sending SIGINT only to the launcher pid (not the whole process group) is
    deliberate: it verifies the launcher actively *forwards* the signal to the
    child rather than the child receiving it directly.
    """
    psutil = pytest.importorskip("psutil")

    # The child prints "ready" then sleeps; if the launcher does not forward
    # the signal, this sleeper is orphaned and keeps running.
    child_code = (
        "import sys, time; print('ready'); sys.stdout.flush(); time.sleep(60)"
    )
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
        child_code,
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,  # launcher gets its own process group
    )

    # Drain stdout on a background thread and signal when the child is ready.
    ready = threading.Event()

    def _reader():
        try:
            for line in proc.stdout:
                if b"ready" in line:
                    ready.set()
        except Exception:
            pass

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    try:
        assert ready.wait(timeout=30), "child never reported ready"

        # Enumerate descendants BEFORE signaling so we can confirm they are
        # gone afterwards (the sleeping python grandchild in particular).
        parent = psutil.Process(proc.pid)
        descendants = []
        deadline = time.time() + 10
        while time.time() < deadline:
            descendants = parent.children(recursive=True)
            if descendants:
                break
            time.sleep(0.1)
        assert descendants, "expected launcher to have spawned a child process"

        # SIGINT to the launcher pid only.
        proc.send_signal(signal.SIGINT)

        rc = proc.wait(timeout=15)
        assert rc == 130, f"launcher exited {rc}, expected 130"

        # Every descendant must be gone (dead or reaped); tolerate a brief
        # window while the OS reaps them.
        alive = descendants
        gone_deadline = time.time() + 10
        while time.time() < gone_deadline:
            alive = [
                d
                for d in descendants
                if d.is_running() and d.status() != psutil.STATUS_ZOMBIE
            ]
            if not alive:
                break
            time.sleep(0.2)
        assert not alive, f"orphaned descendants still running after SIGINT: {alive}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        reader.join(timeout=5)
