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
# from psutil import Process

# # Save affinity
# affinity = Process().cpu_affinity()

import torch
import torch.distributed as dist

# # Restore affinity
# process().cpu_affinity(affinity)
import sys

def main():
    args = sys.argv[1:]
#    args = get_args()

    # device = torch.device("cuda:0")
    # dist.init_process_group("nccl")
    # device_mesh = LlamaDeviceMesh(
    #     tensor_parallel=dist.get_world_size() // args.pp, pipeline_parallel=args.pp
    # )
    # if args.debug:
    print(
        f"Device mesh: rank={dist.get_rank()},",
            # f"TP={device_mesh.tp_rank()}/{device_mesh.tp_size()},",
            # f"PP={device_mesh.pp_rank()}/{device_mesh.pp_size()}",
    )

    # Choose the number of I/O threads automatically
    # io_threads = args.io_threads if args.io_threads > 0 else device_mesh.tp_size()

    print(f'BVE I think that I am rank {dist.get_rank()}')

    # dist.destroy_process_group()


if __name__ == "__main__":
    main()
