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

import torch
import torch.distributed as dist

import sys
import socket


def main():
    args = sys.argv[1:]
    print(
        f"Device mesh: rank={dist.get_rank()},",
    )

    print(f"{socket.gethostname()} reporting it is rank {dist.get_rank()}")


if __name__ == "__main__":
    main()
