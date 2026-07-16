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
#
# This trampoline runs in one of two modes, selected by the
# ``TORCHRUN_HPC_MODE`` environment variable that torchrun-hpc sets:
#
# * "torchrun" (default): the script is invoked as a torchrun *worker*. torchrun
#   has already performed rendezvous and set RANK/LOCAL_RANK/WORLD_SIZE/
#   MASTER_ADDR/MASTER_PORT. This trampoline only applies the HPC-specific GPU
#   visibility tweaks and then runs the user's script/module. Distributed
#   initialization is left to the user's code (via the standard "env://").
#
# * "mpi": the legacy path where the scheduler launches one process per rank and
#   this trampoline performs its own MPI/TCP rendezvous. Used for
#   ``--rdv-protocol mpi``.
import hpc_launcher.torch

import torch
import torch.distributed as dist
import runpy
import sys
import os

from hpc_launcher.schedulers import get_schedulers


def _discover_visible_gpus():
    """Return ``(env_var_name, [device_ids])`` for whichever vendor GPU
    visibility variable is populated, or ``(None, [])`` if none are set."""
    for e in [
            "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES"
    ]:
        val = os.getenv(e)
        if val:
            return e, val.split(",")
    return None, []


def _run_user_target(args, is_module):
    """Execute the user's script or module, restoring its sys.argv first."""
    # Note that run_path/run_module will prepend args[0] back onto sys.argv, so
    # it needs to be stripped off first.
    sys.argv = args
    if is_module:
        runpy.run_module(args[0], run_name="__main__", alter_sys=True)
    else:
        runpy.run_path(args[0], run_name="__main__")


def torchrun_worker_main():
    """Default mode: run as a torchrun worker.

    torchrun owns rendezvous and has already exported the standard distributed
    environment variables. Here we only reconcile GPU visibility with the
    LOCAL_RANK that torchrun assigned and then hand off to the user's code.
    """
    # sys.argv == [trampoline, <user_target>, <user_args...>]
    args = sys.argv[1:]
    if not args:
        raise Exception(
            "torchrun-hpc trampoline received no training target to execute.")

    is_module = os.getenv("TORCHRUN_HPC_IS_MODULE", "0") == "1"

    # torchrun sets LOCAL_RANK for each worker it spawns.
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    # The single per-node task can see all of the node's GPUs. Find the visible
    # device list from whichever vendor variable is populated.
    gpu_var, avail_gpus = _discover_visible_gpus()

    visibility = os.getenv("TORCHRUN_HPC_GPU_VISIBILITY", "single")
    if avail_gpus:
        local_device_id = local_rank % len(avail_gpus)
        if visibility == "single":
            # Narrow this worker to only its assigned GPU and reset LOCAL_RANK to
            # 0, so user code that assumes a single visible device (index 0)
            # behaves correctly.
            os.environ[gpu_var] = avail_gpus[local_device_id]
            os.environ["LOCAL_RANK"] = "0"
        else:
            # Keep all GPUs visible; point LOCAL_RANK at the round-robin device.
            os.environ["LOCAL_RANK"] = f"{local_device_id}"

    # Distributed initialization is intentionally left to the user's script,
    # which should use the default "env://" init method populated by torchrun.
    _run_user_target(args, is_module)


def mpi_legacy_main():
    """Legacy mode: one process per rank, this trampoline does the rendezvous."""
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

    # Standard operating mode assumes that there is one rank per GPU
    # Check to see how many GPUS are actually available to this rank
    _, avail_gpus = _discover_visible_gpus()

    # Round-robin assign the visibile GPUs
    if avail_gpus:
        local_device_id = local_rank % len(avail_gpus)
    else:
        local_device_id = local_rank
    os.environ["LOCAL_RANK"] = f"{local_device_id}"

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
            dist.init_process_group(backend,
                                    init_method=rdv_protocol,
                                    world_size=world_size,
                                    rank=rank,
                                    device_id=torch.device(
                                        device, local_device_id))

            if rdv_protocol == "mpi://" and rank == 0:
                print("[Rank {} of {}]: MPI Version: {}".format(
                    rank, world_size, MPI.Get_version()))
                print("[Rank {} of {}]: MPI Implementation: {}".format(
                    rank, world_size, MPI.Get_library_version()))

    # If the world size is only 1, torch distributed doesn't have to be initialized
    # however, the called application may try to setup torch distributed -- provide env variables
    # Additionally, some codes (e.g. Huggingface accelerate) will look for these fields
    os.environ["WORLD_SIZE"] = f"{world_size}"
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

    # ``args`` already has this trampoline's name (and any leading "-m")
    # stripped, so args[0] is the user target -- exactly what _run_user_target
    # expects as sys.argv.
    _run_user_target(args, is_module)

    if dist.is_initialized():
        # Deal with destroying the process group here
        dist.destroy_process_group()


def main():
    mode = os.getenv("TORCHRUN_HPC_MODE", "torchrun")
    if mode == "mpi":
        mpi_legacy_main()
    else:
        torchrun_worker_main()


if __name__ == "__main__":
    main()
