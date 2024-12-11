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
import os
import re

def test_launcher():
    cmd = ["torchrun-hpc", "-v",  "-N2", "-n1",  "torch_dist_test.py", "--pp 2", "--debug", "--io-threads 4", "--compile"]
    proc = subprocess.run(cmd,
                          universal_newlines = True,
                          capture_output=True)
    # print('Here is stdout')
    # print(proc.stdout)
    # print('Here is stderr')
    # print(proc.stderr)
    m = re.search('^.*Script filename: (\S+)$', proc.stderr, re.MULTILINE | re.DOTALL)
    if m:
        script = m.group(1)
        exp_dir = os.path.dirname(script)
        hostlist = exp_dir + "/hpc_launcher_hostlist.txt"
        with open(hostlist) as f:
            s = f.read()
            s = s.strip("]\n")
            (hostname, inst_array) = s.split("[")
            instances = inst_array.split(',')
            hosts = []
            for i in instances:
                hosts += [hostname + i]

            # print(f'I have hosts {hosts}')

            i = 0
            for h in hosts:
                regex = re.compile('.*({}) reporting it is rank ({}).*'.format(h, i), re.MULTILINE | re.DOTALL)
                match = regex.match(proc.stdout)
                if match:
                    assert match.group(2) != i, f'{match.group(1)} has the incorrect rank'
                    i += 1
                else:
                    assert False, f'{h} not found in output'

    else:
        assert False, f'Unable to find expected hostlist {hostlist}'

    assert(proc.returncode == 0)
