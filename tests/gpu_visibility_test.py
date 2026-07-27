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
Every GPU the scheduler grants a rank must still be usable by that rank.

With ``--gpus-per-proc N`` the scheduler asks for N GPUs per task, so a rank
running per-process model or pipeline parallelism expects to address all N of
them. Nothing between the allocation and the user's script may quietly shrink
that set: a worker narrowed to a single device reports "invalid device
ordinal" for cuda:1 and leaves the rest of the allocation idle but still
charged to the job.

Two halves are checked here, because the guarantee needs both:

- the scheduler really does request ``gpus_per_proc`` GPUs per task, so there
  is something to preserve; and
- the trampoline hands the user's script the same visible-device set it was
  given, rather than picking one device and hiding the others.

The trampoline chooses a *primary* device for the rank (round-robin over the
visible list, exposed as LOCAL_RANK). Choosing a primary is not the same as
restricting the process to it, and this file is about the difference.
"""
import os
import socket
import subprocess
import sys

import pytest

from hpc_launcher.schedulers.flux import FluxScheduler
from hpc_launcher.schedulers.slurm import SlurmScheduler
from hpc_launcher.systems.system import GenericSystem

from conftest import require_torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_VISIBILITY_VARS = ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                    "HIP_VISIBLE_DEVICES")


# ---------------------------------------------------------------------------
# The allocation really does grant gpus_per_proc GPUs per task
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "scheduler_cls,expected",
    [
        (SlurmScheduler, ["--gpus-per-task=2"]),
        (FluxScheduler, ["--gpus-per-slot=2", "--gpus-per-task=2"]),
    ],
)
def test_scheduler_requests_gpus_per_proc_per_task(scheduler_cls, expected,
                                                   stub_system):
    """
    ``-n 2 --gpus-per-proc 2`` must ask the scheduler for 2 GPUs per task (4
    on the node), not 1. This is the half of the guarantee that makes the
    visibility test below meaningful.
    """
    scheduler = scheduler_cls(nodes=1, procs_per_node=2, gpus_per_proc=2)

    script = scheduler.launcher_script(stub_system, "python", ["train.py"],
                                       blocking=False)

    for token in expected:
        assert token in script, (
            f"{scheduler_cls.__name__} did not request 2 GPUs per task "
            f"({token} missing):\n{script}"
        )


# ---------------------------------------------------------------------------
# The trampoline preserves the granted visible-device set
# ---------------------------------------------------------------------------
def _visibility_seen_by_user_script(granted_var, granted_value, tmp_path):
    """
    Run the real trampoline on a script that reports the visibility it was
    handed, and return that report as a dict.

    A single-rank (``local``) configuration is used so no rendezvous is
    needed: the visible-device set is a per-process property and does not
    depend on the world size.
    """
    user_script = tmp_path / "report_visibility.py"
    user_script.write_text(
        "import json\n"
        "import os\n"
        "state = {v: os.environ.get(v) for v in "
        f"{list(_VISIBILITY_VARS)!r}"
        "}\n"
        "state['LOCAL_RANK'] = os.environ.get('LOCAL_RANK')\n"
        "with open(os.environ['VISIBILITY_REPORT'], 'w') as fh:\n"
        "    json.dump(state, fh)\n"
    )
    report = tmp_path / "visibility.json"

    env = os.environ.copy()
    env["TORCHRUN_HPC_SCHEDULER"] = "local"
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["VISIBILITY_REPORT"] = str(report)
    for var in _VISIBILITY_VARS:
        env.pop(var, None)
    env[granted_var] = granted_value

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "hpc_launcher.torch.torchrun_hpc_trampoline",
            str(user_script),
        ],
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        universal_newlines=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"

    import json

    return json.loads(report.read_text())


@pytest.mark.parametrize(
    "granted_var",
    ["CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"],
)
@pytest.mark.parametrize(
    "granted_value",
    [
        # --gpus-per-proc 2: the shape where dropping devices actually costs
        # the user half their allocation.
        "0,1",
        "0,1,2,3",
    ],
)
def test_trampoline_preserves_granted_devices(granted_var, granted_value,
                                              tmp_path):
    """
    Whatever device list the rank was granted must still be visible to the
    user's script.

    ROCR_VISIBLE_DEVICES is checked alongside CUDA_VISIBLE_DEVICES because on
    ROCm the launcher deliberately moves ROCR_VISIBLE_DEVICES into
    HIP_VISIBLE_DEVICES on import; that rename is expected, but the *set* of
    devices must survive it intact.
    """
    require_torch()

    seen = _visibility_seen_by_user_script(granted_var, granted_value,
                                           tmp_path)

    granted = set(granted_value.split(","))
    still_visible = set()
    for var in _VISIBILITY_VARS:
        if seen.get(var):
            still_visible.update(seen[var].split(","))

    assert still_visible == granted, (
        f"the rank was granted devices {sorted(granted)} via {granted_var} "
        f"but its script sees {sorted(still_visible)} (raw: {seen}); the "
        "devices the scheduler allocated must not be hidden from the rank"
    )

    # The primary device is selected by index into the granted list, so it is
    # always a valid index -- but selecting one must not have hidden the rest.
    assert seen["LOCAL_RANK"] is not None, seen
    assert 0 <= int(seen["LOCAL_RANK"]) < len(granted), seen


# ---------------------------------------------------------------------------
# Primary-device selection: in range, and free of side effects
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "visible,local_rank,expected",
    [
        # One GPU per rank: rank N takes the Nth visible device.
        ("0,1,2,3", 0, 0),
        ("0,1,2,3", 3, 3),
        # --gpus-per-proc 2 with 2 ranks per node: each rank-task is granted
        # its own pair, so within the process the index is rank-relative.
        ("0,1", 0, 0),
        ("0,1", 1, 1),
        # More ranks than visible devices wraps rather than going out of range.
        ("0,1", 2, 0),
    ],
)
def test_primary_device_index_in_range(visible, local_rank, expected,
                                       monkeypatch):
    require_torch()
    import hpc_launcher.torch.torchrun_hpc_trampoline as tramp

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)

    assert tramp._select_local_device_id(local_rank) == expected


def test_local_rank_is_not_the_primary_device_index(tmp_path):
    """
    The primary device index and ``LOCAL_RANK`` are different quantities and
    must not collapse into one (round-2 review K2).

    Round-robin selection over the visible list is correct -- that is what
    ``test_primary_device_index_in_range`` above pins -- but with the
    launcher's default ``--gpus-per-proc 1`` each task is confined to a single
    device, so the selected index is ``0`` for *every* rank on the node. If
    ``LOCAL_RANK`` is that index, every rank claims to be local rank 0 and
    local-leader election, per-node-rank sharding and per-local-rank log names
    all quietly break, while each rank is nonetheless holding a distinct GPU.

    Four real ranks are launched (a rendezvous over loopback with the gloo
    backend, as in ``trampoline_device_test.py``). The visibility environment
    is set so that the trampoline is handed a one-device list -- the real
    per-task view -- while torch itself sees no accelerator, which is the only
    way to have both properties in an environment with no working GPU
    collective: ``ROCR_VISIBLE_DEVICES`` carries the granted device with the
    ROCR->HIP rename disabled, and both ``CUDA_VISIBLE_DEVICES`` and
    ``HIP_VISIBLE_DEVICES`` are emptied so every torch build takes the CPU
    path.
    """
    require_torch()

    user_script = tmp_path / "report_local_rank.py"
    user_script.write_text(
        "import os\n"
        "with open(os.environ['LOCAL_RANK_REPORT'], 'w') as fh:\n"
        "    fh.write(os.environ.get('LOCAL_RANK', '<UNSET>'))\n"
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    base_env = os.environ.copy()
    base_env["TORCHRUN_HPC_SCHEDULER"] = "slurm"
    base_env["SLURM_NTASKS"] = "4"
    base_env["SLURM_NNODES"] = "1"
    base_env["TORCHRUN_HPC_RDV_PROTOCOL"] = f"tcp://127.0.0.1:{port}"
    base_env["PYTHONPATH"] = REPO_ROOT + os.pathsep + base_env.get("PYTHONPATH", "")
    # One granted device, as --gpus-per-proc 1 produces, but no accelerator
    # for torch: see the docstring.
    base_env["TORCHRUN_HPC_UNSWAP_ROCR_HIP_VIS_DEV"] = "TRUE"
    base_env["ROCR_VISIBLE_DEVICES"] = "0"
    base_env["CUDA_VISIBLE_DEVICES"] = ""
    base_env["HIP_VISIBLE_DEVICES"] = ""

    procs, reports = [], []
    for rank in range(4):
        env = base_env.copy()
        env["SLURM_PROCID"] = str(rank)
        env["SLURM_LOCALID"] = str(rank)
        report = tmp_path / f"local_rank_{rank}.txt"
        env["LOCAL_RANK_REPORT"] = str(report)
        reports.append(report)
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "hpc_launcher.torch.torchrun_hpc_trampoline",
                    str(user_script),
                ],
                env=env,
                cwd=str(tmp_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
        )

    outputs = []
    try:
        for proc in procs:
            outputs.append(proc.communicate(timeout=300))
    except subprocess.TimeoutExpired:
        for proc in procs:
            proc.kill()
        pytest.fail("the four ranks did not complete their rendezvous")

    for rank, proc in enumerate(procs):
        assert proc.returncode == 0, f"rank {rank}: {outputs[rank][0]}"

    seen = [r.read_text() for r in reports]
    assert seen == ["0", "1", "2", "3"], (
        f"four ranks on one node reported LOCAL_RANK {seen}; each rank holds "
        "a distinct GPU but they cannot all be local rank 0 -- LOCAL_RANK is "
        "the rank's place on the node, not the index of the device it picked"
    )


def test_primary_device_selection_does_not_restrict_visibility(monkeypatch):
    """
    Choosing this rank's primary device must be a pure computation: it may
    not rewrite the visibility variables to the chosen device, which is what
    would strand the rest of a multi-GPU allocation.
    """
    require_torch()
    import hpc_launcher.torch.torchrun_hpc_trampoline as tramp

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)

    tramp._select_local_device_id(2)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    for var in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        assert var not in os.environ, (
            f"selecting a primary device introduced {var}, which would "
            "override the granted device list"
        )
