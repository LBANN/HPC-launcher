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
from psutil import Process

# Save affinity before importing torch.distributed
affinity = Process().cpu_affinity()

import torch
import torch.distributed as dist
import runpy
import atexit
import sys
import os

# Restore affinity after importing torch.distributed
Process().cpu_affinity(affinity)
import sys


def main():
    # args = get_args()
    args = sys.argv[1:]

    # Fix how we handle CUDA visible devices and MPI bind
    device = torch.device("cuda:0")
    # Do we need to pass env
    dist.init_process_group("nccl")

    # Needs to figure out how to handle exceptions
    # atexit.register(dist.destroy_process_group)
    # runfile
    script = sys.argv[0]
    run_path = os.path.dirname(script)
    # print(f'BVE I have the args {args} and an old argv[0] {sys.argv[0]} and run_path = {run_path}')
    # Strip off the name of this script and pass the rest to runpy
    # sys.argv[1] = run_path + "/" + sys.argv[1]
    # sys.argv[0] = run_path + sys.argv[1]
    sys.argv = sys.argv[1:]
    # print(f'BVE NOW Here are the new args {sys.argv}')
    runpy.run_path(sys.argv[0], run_name="__main__")

    # Deal with destroying the process group here
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
