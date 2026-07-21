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
Tests for trampoline device handling (review finding E5).

E5: ``torchrun_hpc_trampoline.main()`` used to pass
``device_id=torch.device("cpu", ...)`` to ``dist.init_process_group`` even on
the CPU/gloo path, which torch >= 2.x rejects, crashing every multi-rank
CPU/gloo job at initialization.

These are Tier B tests: they need a CPU-capable torch. The import is guarded
with the shared ``require_torch()`` helper.
"""
import os
import socket
import subprocess
import sys
from unittest import mock

import pytest

from conftest import require_torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _free_port() -> int:
    """Bind an ephemeral loopback port, then release it for reuse."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ---------------------------------------------------------------------------
# E5 - _process_group_kwargs helper
# ---------------------------------------------------------------------------
def test_pg_kwargs_cpu_omits_device_id():
    require_torch()
    import hpc_launcher.torch.torchrun_hpc_trampoline as tramp

    kwargs = tramp._process_group_kwargs(
        "gloo", "tcp://127.0.0.1:12345", 2, 0, "cpu", 0
    )
    # torch >= 2.x rejects a CPU device_id, so it must never be present for
    # the CPU/gloo path regardless of whether an accelerator is installed on
    # the machine running the test.
    assert "device_id" not in kwargs
    assert kwargs == {
        "backend": "gloo",
        "init_method": "tcp://127.0.0.1:12345",
        "world_size": 2,
        "rank": 0,
    }


def test_pg_kwargs_cuda_includes_device_id():
    torch = require_torch()
    import hpc_launcher.torch.torchrun_hpc_trampoline as tramp

    # Force the accelerator branch even on a CPU-only host.
    with mock.patch(
        "hpc_launcher.torch.torchrun_hpc_trampoline.torch.cuda.is_available",
        return_value=True,
    ):
        kwargs = tramp._process_group_kwargs(
            "nccl", "tcp://127.0.0.1:12345", 2, 1, "cuda", 3
        )

    assert "device_id" in kwargs
    assert kwargs["device_id"] == torch.device("cuda", 3)
    assert kwargs["backend"] == "nccl"
    assert kwargs["rank"] == 1
    assert kwargs["world_size"] == 2
    assert kwargs["init_method"] == "tcp://127.0.0.1:12345"


# ---------------------------------------------------------------------------
# E5 - end-to-end CPU/gloo two-rank regression
# ---------------------------------------------------------------------------
def test_cpu_gloo_two_ranks_init(tmp_path):
    """
    Launch the real trampoline in two subprocesses over loopback TCP with a
    stubbed 2-rank scheduler configuration and a CPU-only visibility env. Each
    rank runs a tiny user script that asserts ``dist.is_initialized()``,
    all-reduces a CPU tensor and writes a success marker.

    Before the E5 fix this crashed both ranks with
    ``ValueError: init_process_group device_id parameter must be an
    accelerator with an index``. Loopback TCP rendezvous is expected to work
    in CI and in the sandbox.
    """
    require_torch()

    user_script = tmp_path / "user_script.py"
    user_script.write_text(
        "import os\n"
        "import torch\n"
        "import torch.distributed as dist\n"
        "assert dist.is_initialized(), 'torch.distributed was not initialized'\n"
        "rank = dist.get_rank()\n"
        "world_size = dist.get_world_size()\n"
        "t = torch.ones(1)\n"
        "dist.all_reduce(t)\n"
        "assert t.item() == world_size, f'all_reduce {t.item()} != {world_size}'\n"
        "with open(os.environ['TRAMPOLINE_TEST_MARKER'], 'w') as fh:\n"
        "    fh.write(f'OK rank={rank} world_size={world_size} sum={t.item()}')\n"
        "print(f'TRAMPOLINE_TEST_SUCCESS rank={rank} of {world_size}', flush=True)\n"
    )

    port = _free_port()

    base_env = os.environ.copy()
    # The `local` scheduler's get_parallel_configuration() is hardcoded to a
    # world size of 1, so a real multi-rank world is stubbed via the `slurm`
    # backend, whose get_parallel_configuration() reads these SLURM_* vars.
    base_env["TORCHRUN_HPC_SCHEDULER"] = "slurm"
    base_env["SLURM_NTASKS"] = "2"
    base_env["SLURM_NNODES"] = "1"
    base_env["TORCHRUN_HPC_RDV_PROTOCOL"] = f"tcp://127.0.0.1:{port}"
    # Force the CPU/gloo path deterministically (this is the E5 regression
    # path). Empty *_VISIBLE_DEVICES makes torch.cuda.is_available() False on
    # both CUDA and ROCm builds.
    base_env["HIP_VISIBLE_DEVICES"] = ""
    base_env["CUDA_VISIBLE_DEVICES"] = ""
    base_env["PYTHONPATH"] = REPO_ROOT + os.pathsep + base_env.get("PYTHONPATH", "")

    procs = []
    markers = []
    for rank in range(2):
        env = base_env.copy()
        env["SLURM_PROCID"] = str(rank)
        env["SLURM_LOCALID"] = str(rank)
        marker = tmp_path / f"marker_{rank}.txt"
        env["TRAMPOLINE_TEST_MARKER"] = str(marker)
        markers.append(marker)
        cmd = [
            sys.executable,
            "-m",
            "hpc_launcher.torch.torchrun_hpc_trampoline",
            str(user_script),
        ]
        procs.append(
            subprocess.Popen(
                cmd,
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
            out, _ = proc.communicate(timeout=180)
            outputs.append((proc.returncode, out))
    except subprocess.TimeoutExpired:
        for proc in procs:
            proc.kill()
        pytest.fail("Trampoline ranks did not finish rendezvous within timeout")

    for rank, (rc, out) in enumerate(outputs):
        assert rc == 0, f"rank {rank} exited with {rc}\n{out}"
        assert f"TRAMPOLINE_TEST_SUCCESS rank={rank} of 2" in out, out

    for rank, marker in enumerate(markers):
        assert marker.exists(), f"rank {rank} did not write its success marker"
        assert marker.read_text().startswith("OK "), marker.read_text()
