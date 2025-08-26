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

import sys
import socket
import os


def main():
    args = sys.argv[1:]
    torch_dist_initialized = dist.is_initialized()
    for e in ["CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"]:
        if os.getenv(e):
            gpus = os.getenv(e)

    if gpus:
        avail_gpus = gpus.split(",")
        
    local_rank = int(os.environ['LOCAL_RANK'])
    print(f"Local Rank: {local_rank}")
    
    if torch_dist_initialized:
        print(
            f"Device mesh: rank={dist.get_rank()} and local rank is {local_rank} and avail_gpus = {avail_gpus},",
        )

        print(f"{socket.gethostname()} reporting it is rank {dist.get_rank()} of {dist.get_world_size()}")
    else:
        print(f"{socket.gethostname()} reporting it is rank 0 of 1")


if __name__ == "__main__":
    main()
