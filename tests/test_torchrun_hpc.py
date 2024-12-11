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

#@pytest.fixture(scope="module")
def test_launcher():
#     proc = subprocess.run(["ls", "-l"],
# #                          capture_output=True,
#                           stdout = subprocess.PIPE)
# torchrun-hpc -v -N2 -n1  hpc_launcher/cli/test_main.py --pp 2 --debug --io-threads 4 --compile
    cmd = ["torchrun-hpc", "-v",  "-N2", "-n1",  "torch_dist_test.py", "--pp 2", "--debug", "--io-threads 4", "--compile"]
#    cmd = ["torchrun-hpc", "-v",  "-N2", "-n1",  "../hpc_launcher/cli/test_main.py", "--pp 2", "--debug", "--io-threads 4", "--compile"]
    proc = subprocess.run(cmd,
                          universal_newlines = True,
                          capture_output=True)
    # proc = subprocess.run(["ls", "-l"],
    #                       universal_newlines = True,
    #                       capture_output=True)
    print('Here is stdout')
    print(proc.stdout)
    print('Here is stderr')
    print(proc.stderr)
    m = re.search('^.*Script filename: (\S+)$', proc.stderr, re.MULTILINE | re.DOTALL)
    if m:
        script = m.group(1)
        exp_dir = os.path.dirname(script)
        # print(f'I found match >>>{script}<<\n and dir >>> {exp_dir}<<<')
        hostlist = exp_dir + "/hpc_launcher_hostlist.txt"
        # print(f'I am going to read {hostlist}')
        with open(hostlist) as f:
            s = f.read()
            # print(f'I am reading the file list {s}')
            s = s.strip("]\n")
            (hostname, inst_array) = s.split("[")
            instances = inst_array.split(',')
            # print(f'I have {hostname} and instances {inst_array} and instance list {instances}')
            hosts = []
            for i in instances:
                hosts += [hostname + i]

            # print(f'I have hosts {hosts}')

            i = 0
            for h in hosts:
                regex = re.compile('.*({}) reporting it is rank ({}).*'.format(h, i), re.MULTILINE | re.DOTALL)
#                regex = re.compile('^({}) reporting it is rank ({})$'.format(h, i), re.MULTILINE | re.DOTALL)
                match = regex.match(proc.stdout)
                if match:
                    # print(f'BVE ASSERT {match.group(1)} checking in as rank {match.group(2)}')
                    assert match.group(2) != i, f'{match.group(1)} has the incorrect rank'
                    # print(f'I found the string {match.group(0)}')
                    i += 1
                else:
                    assert False, f'{h} not found in output'

    else:
        assert False, f'Unable to find expected hostlist {hostlist}'

    assert(proc.returncode == 0)


# torchrun-hpc -v -N2 -n1 hpc_launcher/cli/test_main.py --pp 2 --debug --io-threads 4 --compile 
