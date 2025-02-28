# Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
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


def main():
    # Strip off the name of this script and pass the rest to runpy
    args = sys.argv[1:]

    scheduler_type = os.getenv("TORCHRUN_HPC_SCHEDULER")
    scheduler = get_schedulers()[scheduler_type]
    (world_size, rank, local_world_size, local_rank) = (
        scheduler.get_parallel_configuration()
    )

    # Report if the memory size was set
    if torch.cuda.is_available():
        fraction_max_gpu_mem = float(os.getenv("TORCHRUN_HPC_MAX_GPU_MEM"))
        if fraction_max_gpu_mem != 1.0 and rank == 0:
            print(f"[Rank {rank} of {world_size}] TORCHRUN-HPC set the max GPU memory fraction to {fraction_max_gpu_mem}")

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
            except (ImportError, ModuleNotFoundError):
                mpi = None
                raise Exception(
                    f"MPI rendezvous protocol selected without installing mpi_rndv library."
                )

        if not torch_dist_initialized:
            if rank == 0:
                print(f"[Rank {rank} of {world_size}]: Initializing distributed PyTorch using protocol: {rdv_protocol}")
            # TODO(later): Fix how we handle CUDA visible devices and MPI bind
            dist.init_process_group(
                "nccl", init_method=rdv_protocol, world_size=world_size, rank=rank
            )

            if rdv_protocol == "mpi://" and rank == 0:
                print("[Rank {} of {}]: MPI Version: {}".format(rank, world_size, MPI.Get_version()))
                print("[Rank {} of {}]: MPI Implementation: {}".format(rank, world_size, MPI.Get_library_version()))
    else:
        # If the world size is only 1, torch distributed doesn't have to be initialized
        # however, the called application may try to setup torch distributed -- provide env variables
        os.environ["WORLD_SIZE"] = f"{world_size}"
        if os.getenv("TORCHRUN_HPC_MASTER_ADDR"):
            os.environ["MASTER_ADDR"] = os.getenv("TORCHRUN_HPC_MASTER_ADDR")
        if os.getenv("TORCHRUN_HPC_MASTER_PORT"):
            os.environ["MASTER_PORT"] = os.getenv("TORCHRUN_HPC_MASTER_PORT")


    # Note that run_path will prepend the args[0] back onto the sys.argv so it needs to be stripped off first
    sys.argv = sys.argv[1:]
    # Run underlying script
    runpy.run_path(args[0], run_name="__main__")

    if dist.is_initialized():
        # Deal with destroying the process group here
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
