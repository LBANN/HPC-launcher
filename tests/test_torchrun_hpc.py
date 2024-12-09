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
import pytest
import hpc_launcher
# import torch.distributed as dist

# import torch
import subprocess


#@pytest.fixture(scope="module")
def test_launcher():
    subprocess.run(["ls", "-l"]) 
    return True


# torchrun-hpc -v -N2 -n1 hpc_launcher/cli/test_main.py --pp 2 --debug --io-threads 4 --compile 
