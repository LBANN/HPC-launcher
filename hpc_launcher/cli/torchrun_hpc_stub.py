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

from hpc_launcher.schedulers import get_schedulers

def main():
    # Strip off the name of this script and pass the rest to runpy
    args = sys.argv[1:]

    scheduler_type = os.getenv('TORCHRUN_HPC_SCHEDULER')
    scheduler = get_schedulers()[scheduler_type]
    (world_size, rank, local_world_size, local_rank) = scheduler.get_parallel_configuration()

    rdv_protocol = os.getenv('TORCHRUN_HPC_RDV_PROTOCOL')
    init_method = scheduler.dynamically_configure_rendezvous_protocol(rdv_protocol)

    if dist.is_initialized():
        raise Exception('PyTorch Distributed is already initialized')

    # TODO(later): Fix how we handle CUDA visible devices and MPI bind
    dist.init_process_group("nccl", init_method=init_method,
                            world_size=world_size, rank=rank)

    # Run underlying script
    runpy.run_path(args[0], run_name="__main__")

    # Deal with destroying the process group here
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
