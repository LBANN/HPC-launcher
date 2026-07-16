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
# from psutil import Process

import torch
import torch.distributed as dist

import os
import sys
import socket


def main():
    args = sys.argv[1:]

    # The legacy MPI path initializes the process group inside the trampoline.
    # The default (torchrun) path leaves initialization to us: torchrun exports
    # RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT so the standard "env://" init works.
    if not dist.is_initialized() and os.getenv("RANK") is not None and os.getenv("WORLD_SIZE") is not None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    torch_dist_initialized = dist.is_initialized()
    if torch_dist_initialized:
        print(
            f"Device mesh: rank={dist.get_rank()},",
        )

        print(f"{socket.gethostname()} reporting it is rank {dist.get_rank()} of {dist.get_world_size()}")
    else:
        print(f"{socket.gethostname()} reporting it is rank 0 of 1")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
