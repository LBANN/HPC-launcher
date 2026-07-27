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
import hpc_launcher.torch

import torch
import torch.distributed as dist
import runpy
import atexit
import sys
import os

from hpc_launcher.schedulers import get_schedulers


def _process_group_kwargs(backend, init_method, world_size, rank, device,
                          local_device_id):
    """
    Build the keyword arguments for ``torch.distributed.init_process_group``.

    The ``device_id`` argument is only meaningful for accelerator devices:
    torch >= 2.x rejects a CPU ``device_id`` with ``ValueError:
    init_process_group device_id parameter must be an accelerator with an
    index``. Passing it unconditionally crashed every multi-rank CPU/gloo job
    at initialization. Include ``device_id`` only when an
    accelerator is actually in use.
    """
    kwargs = dict(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    if device != "cpu" and torch.cuda.is_available():
        kwargs["device_id"] = torch.device(device, local_device_id)
    return kwargs


def _select_local_device_id(local_rank):
    """
    Round-robin the visible GPUs to choose the device this rank will use.

    Reads the first populated ``*_VISIBLE_DEVICES`` variable and, when GPUs
    are visible, assigns ``local_rank % len(visible)``; otherwise falls back
    to ``local_rank``.
    """
    avail_gpus = []
    for e in [
            "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES"
    ]:
        if os.getenv(e):
            avail_gpus = os.getenv(e).split(",")
            break
    if avail_gpus:
        return local_rank % len(avail_gpus)
    return local_rank


def _rank_identity(world_size, rank, local_world_size, local_rank):
    """
    Build the rank-identity environment this process publishes to the user's
    script.

    Rank identity is published *here*, by the component that is told what it
    is, rather than by the generated launch script. ``export
    RANK=${SLURM_PROCID}`` in a script is only correct when that script is
    itself executed once per task. For a ``--bg`` submission it is the batch
    script, which runs once at allocation scope before any task exists, so
    every task inherited a single frozen value -- ``0`` under Slurm (whose
    batch step is a real one-task step), empty under Flux (``FLUX_TASK_RANK``
    is actively unset for an initial program) and empty under LSF. The
    trampoline is executed once per task by construction and already computes
    this rank for ``init_process_group``.

    ``LOCAL_RANK`` is the rank's node-local rank, which is *not* the index of
    the GPU it selected. The two agree only when the process can see at least
    ``local_world_size`` devices; the launcher passes ``--gpus-per-task``
    (default 1) and the scheduler confines each task to its own GPU, so the
    selected device index is 0 for every rank on the node while their
    node-local ranks run 0..N-1. Local-leader checks, per-node-rank sharding
    and per-local-rank log names all need the latter.
    :func:`_select_local_device_id` remains the device selector and its
    round-robin is deliberately unchanged.

    ``NODE_RANK`` is derived from the two under the same uniform-distribution
    assumption the schedulers' ``get_parallel_configuration`` already makes
    when it computes ``local_world_size`` as ``world_size // nodes``.

    :param world_size: Total number of ranks in the job.
    :param rank: This process's global rank.
    :param local_world_size: Number of ranks on this node.
    :param local_rank: This process's rank within its node.
    :return: A mapping of environment variable name to value.
    """
    node_rank = rank // local_world_size if local_world_size else 0
    return {
        "WORLD_SIZE": f"{world_size}",
        "RANK": f"{rank}",
        "LOCAL_RANK": f"{local_rank}",
        "NODE_RANK": f"{node_rank}",
    }


def _apply_memory_fraction(local_device_id):
    """
    Apply the optional GPU memory-fraction cap to the device this rank has
    actually selected.

    This previously ran at ``import hpc_launcher.torch`` time with no device
    argument, which capped device 0 regardless of which GPU the worker went
    on to use. Applying it here, after ``local_device_id``
    has been chosen, caps the correct device. It is a no-op on CPU-only ranks
    and when the fraction is left at the default 1.0.
    """
    if not torch.cuda.is_available():
        return
    fraction_max_gpu_mem = float(os.getenv("HPC_LAUNCHER_MAX_GPU_MEM", 1.0))
    if fraction_max_gpu_mem != 1.0:
        torch.cuda.set_per_process_memory_fraction(
            fraction_max_gpu_mem, device=local_device_id)


def main():
    # Strip off the name of this script and pass the rest to runpy
    args = sys.argv[1:]
    if args[0] == "-m":
        is_module = True
        args = args[1:]
    else:
        is_module = False

    scheduler_type = os.getenv("TORCHRUN_HPC_SCHEDULER")
    scheduler = get_schedulers()[scheduler_type]
    (world_size, rank, local_world_size,
     local_rank) = (scheduler.get_parallel_configuration())

    # Check on the backend and report if the memory size was set
    backend = None
    device = None
    if torch.cuda.is_available():
        backend = "nccl"
        device = "cuda"
        fraction_max_gpu_mem = float(os.getenv("HPC_LAUNCHER_MAX_GPU_MEM",
                                               1.0))
        if fraction_max_gpu_mem != 1.0 and rank == 0:
            print(
                f"[Rank {rank} of {world_size}] TORCHRUN-HPC set the max GPU memory fraction to {fraction_max_gpu_mem}"
            )
    else:
        backend = "gloo"
        device = "cpu"

    # Standard operating mode assumes that there is one rank per GPU.
    # Round-robin the visible GPUs to select this rank's device. This is a
    # device index, not this rank's identity -- see _rank_identity.
    local_device_id = _select_local_device_id(local_rank)

    # Publish this rank's identity, overwriting anything the launch script may
    # have left in the environment. Even when the world size is 1 the called
    # application may set torch distributed up itself, and some codes (e.g.
    # Huggingface accelerate) look for these fields.
    os.environ.update(
        _rank_identity(world_size, rank, local_world_size, local_rank))

    # Apply the optional GPU memory-fraction cap to the selected device. This
    # used to run at import time against device 0 regardless of the device the
    # worker ends up using.
    _apply_memory_fraction(local_device_id)

    torch_dist_initialized = dist.is_initialized()
    rdv_protocol = os.getenv("TORCHRUN_HPC_RDV_PROTOCOL")
    if world_size > 1 or rdv_protocol == "mpi://":
        if rdv_protocol == "mpi://":
            try:
                import mpi4py

                # This will automatically register MPI for initialization.
                import mpi_rdv
                from mpi4py import MPI

                mpi = True
                if backend == "gloo" and torch.distributed.is_mpi_available():
                    backend = "mpi"
            except (ImportError, ModuleNotFoundError):
                mpi = None
                raise Exception(
                    f"MPI rendezvous protocol selected without installing mpi_rndv library."
                )

        if not torch_dist_initialized:
            if not backend:
                raise Exception(
                    f"torchrun-hpc is unable to find a valid backend for torch distributed."
                )

            if rank == 0:
                print(
                    f"[Rank {rank} of {world_size}]: Initializing distributed PyTorch using protocol: {rdv_protocol}"
                )
            # TODO(later): Fix how we handle CUDA visible devices and MPI bind
            pg_kwargs = _process_group_kwargs(backend, rdv_protocol,
                                              world_size, rank, device,
                                              local_device_id)
            dist.init_process_group(**pg_kwargs)

            if rdv_protocol == "mpi://" and rank == 0:
                print("[Rank {} of {}]: MPI Version: {}".format(
                    rank, world_size, MPI.Get_version()))
                print("[Rank {} of {}]: MPI Implementation: {}".format(
                    rank, world_size, MPI.Get_library_version()))

    # The rendezvous coordinates go alongside the identity published above, so
    # an application that sets torch distributed up itself finds a complete
    # environment.
    if os.getenv("TORCHRUN_HPC_MASTER_ADDR"):
        os.environ["MASTER_ADDR"] = os.getenv("TORCHRUN_HPC_MASTER_ADDR")
    else:
        # If the mpi rendezvous protocol is set, this should be necessary but some packages still look for it
        os.environ["MASTER_ADDR"] = "127.0.0.1"

    if os.getenv("TORCHRUN_HPC_MASTER_PORT"):
        os.environ["MASTER_PORT"] = os.getenv("TORCHRUN_HPC_MASTER_PORT")
    else:
        # If the mpi rendezvous protocol is set, this should be necessary but some packages still look for it
        os.environ["MASTER_PORT"] = "23456"

    # Note that run_path will prepend the args[0] back onto the sys.argv so it needs to be stripped off first
    sys.argv = sys.argv[1:] if not is_module else sys.argv[2:]

    # Run underlying script
    if is_module:
        runpy.run_module(args[0], run_name="__main__", alter_sys=True)
    else:
        runpy.run_path(args[0], run_name="__main__")

    if dist.is_initialized():
        # Deal with destroying the process group here
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
