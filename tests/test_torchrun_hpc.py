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

import subprocess
import shutil
import os
import re
import sys
import shutil

from hpc_launcher.systems import autodetect
from hpc_launcher.systems.lc.sierra_family import Sierra

def check_hostlist_file(exp_dir: str, stdout_buffer, num_ranks):
    hostlist = os.path.join(exp_dir, "hpc_launcher_hostlist.txt")
    with open(hostlist) as f:
        s = f.read()
        s = s.strip("]\n")
        cluster_list = re.split(r'[,\s]+', s)
        print(f'I found cluster list {cluster_list}')
        hosts = []
        for cluster in cluster_list:
            if cluster == 'lassen710' and \
               ((isinstance(autodetect.autodetect_current_system(), Sierra)) or \
                os.getenv('LSB_HOSTS')):
                print(f'I am skippihng {cluster}')
                continue

            print(f'I am checking cluster {cluster}')
            if '[' in cluster:
                (hostname, inst_array) = cluster.split("[")
                print(f'I found hostname and inst {hostname}, {inst_array}')
                # This only works up to two nodes
                instances = re.split(r'[,-]+', inst_array)
                for i in instances:
                    hosts.append(hostname + i)
            else:
                print(f'I found cluster {cluster}')
                hosts.append(cluster)

        i = 0
        matched = []
        unmatched = []
        for h in hosts:
            print(f'I am looking for host {h}')
            regex = re.compile(
                '.*({}) reporting it is rank ({}).*'.format(h, i),
                re.MULTILINE | re.DOTALL)
            match = regex.match(stdout_buffer)
            if match:
                assert match.group(
                    2
                ) != i, f'{match.group(1)} has the incorrect rank in test {exp_dir}'
                print(
                    f'\n{match.group(1)} is correctly reporting that it was assigned rank {match.group(2)}'
                )
                matched.append(h)
                i += 1
                if i == num_ranks:
                    print(f'Found {i} matches with matched hosts {matched}')
                    break
            else:
                unmatched.append(h)
                print(f'{h} not found in output in test {exp_dir} - only {i} found: {matched}')
                # assert False, f'{h} not found in output in test {exp_dir} - only {i} found: {matched}'

        assert len(matched) == num_ranks, f'Incorrect number of ranks reported, required {num_ranks} -- matched: {matched} and unmatched: {unmatched}'

@pytest.mark.parametrize("local", [True, False])
def test_launcher_one_node(local):
    try:
        import torch
    except (ImportError, ModuleNotFoundError):
        pytest.skip("torch not found")
    if (not local and not shutil.which("srun") and not shutil.which("flux")
            and not shutil.which("jsrun")):
        pytest.skip("No distributed launcher found")

    # Get full path to torch_dist_driver.py
    driver_file = os.path.join(os.path.dirname(__file__),
                               "torch_dist_driver.py")

    cmd = [
        sys.executable, "-m", "hpc_launcher.cli.torchrun_hpc", "-v",
        "--local" if local else "-n1", "-N1", driver_file
    ]
    proc = subprocess.run(cmd, universal_newlines=True, capture_output=True)
    m = re.search(r'^.*Script filename: (\S+)$', proc.stderr,
                  re.MULTILINE | re.DOTALL)
    if m:
        script = m.group(1)
        exp_dir = os.path.dirname(script)
        check_hostlist_file(exp_dir, proc.stdout, 1)
    else:
        assert False, f'Unable to find expected hostlist: hpc_launcher_hostlist.txt'

    assert proc.returncode == 0


@pytest.mark.parametrize('num_nodes', [2])
@pytest.mark.parametrize('procs_per_node', [1])
@pytest.mark.parametrize('rdv', ('mpi', 'tcp'))
def test_launcher_multinode(num_nodes, procs_per_node, rdv):
    try:
        import torch
    except (ImportError, ModuleNotFoundError):
        pytest.skip("torch not found")
    if (not shutil.which("srun") and not shutil.which("flux")
            and not shutil.which("jsrun")):
        pytest.skip("No distributed launcher found")

    # Get full path to torch_dist_driver.py
    driver_file = os.path.join(os.path.dirname(__file__),
                               "torch_dist_driver.py")

    cmd = [
        sys.executable, "-m", "hpc_launcher.cli.torchrun_hpc", "-v", f"-N{num_nodes}",
        f"-n{procs_per_node}", f"-r{rdv}", driver_file
    ]
    proc = subprocess.run(cmd, universal_newlines=True, capture_output=True)
    exp_dir = None
    m = re.search(r'^.*Script filename: (\S+)$', proc.stderr,
                  re.MULTILINE | re.DOTALL)
    if m:
        script = m.group(1)
        exp_dir = os.path.dirname(script)
        check_hostlist_file(exp_dir, proc.stdout, num_nodes * procs_per_node)
    else:
        assert False, f'Unable to find expected hostlist: hpc_launcher_hostlist.txt'

    regex = re.compile('.*Initializing distributed PyTorch using protocol: ({})://.*'.format(rdv), re.MULTILINE | re.DOTALL)
    match = regex.match(proc.stdout)
    if match:
        assert match.group(1) == rdv, f'{match.group(1)} is the incorrect rendezvous protocol: requested {rdv}'
    else:
        assert False, f'Unable to detect a valid rendezvous protocol for test {rdv}'
    assert proc.returncode == 0

    if exp_dir:
        shutil.rmtree(exp_dir, ignore_errors=True)

    
if __name__ == '__main__':
    test_launcher_one_node(False)
#    test_launcher_one_node(True)
