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

# Save affinity before importing torch
affinity = Process().cpu_affinity()

import torch
import torch.distributed as dist
import runpy
import atexit
import sys
import os

# Restore affinity after importing torch
Process().cpu_affinity(affinity)
import sys


def main():
    # Strip off the name of this script and pass the rest to runpy
    args = sys.argv[1:]

    # TODO(later): Fix how we handle CUDA visible devices and MPI bind
    dist.init_process_group("nccl")

    # Run underlying script
    runpy.run_path(args[0], run_name="__main__")

    # Deal with destroying the process group here
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
