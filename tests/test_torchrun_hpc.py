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
import os
import re
import sys
import shutil


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
        hostlist = os.path.join(exp_dir, "hpc_launcher_hostlist.txt")
        if not os.path.exists(hostlist):
            print(".:", os.listdir("."))
            print(exp_dir, ":", os.listdir(exp_dir))
        with open(hostlist) as f:
            s = f.read()
            hostname = s.strip()
            match = re.search(r'\s*(\S+)\s+reporting it is rank\s+(\S+)\s*',
                              proc.stdout)
            if match:
                assert match.group(
                    1) == hostname, f'Hostname mismatch in test {exp_dir}'
    else:
        assert False, f'Unable to find expected hostlist: hpc_launcher_hostlist.txt'

    assert proc.returncode == 0


def test_launcher_twonodes():
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
        sys.executable, "-m", "hpc_launcher.cli.torchrun_hpc", "-v", "-N2",
        "-n1", driver_file
    ]
    proc = subprocess.run(cmd, universal_newlines=True, capture_output=True)
    m = re.search(r'^.*Script filename: (\S+)$', proc.stderr,
                  re.MULTILINE | re.DOTALL)
    if m:
        script = m.group(1)
        exp_dir = os.path.dirname(script)
        hostlist = os.path.join(exp_dir, "hpc_launcher_hostlist.txt")
        with open(hostlist) as f:
            s = f.read()
            s = s.strip("]\n")
            (hostname, inst_array) = s.split("[")
            instances = re.split(r'[,-]+', inst_array)
            hosts = []
            for i in instances:
                hosts += [hostname + i]

            i = 0
            for h in hosts:
                regex = re.compile(
                    '.*({}) reporting it is rank ({}).*'.format(h, i),
                    re.MULTILINE | re.DOTALL)
                match = regex.match(proc.stdout)
                if match:
                    assert match.group(
                        2
                    ) != i, f'{match.group(1)} has the incorrect rank in test {exp_dir}'
                    print(
                        f'\n{match.group(1)} is correctly reporting that it was assigned rank {match.group(2)}'
                    )
                    i += 1
                else:
                    assert False, f'{h} not found in output in test {exp_dir}'

    else:
        assert False, f'Unable to find expected hostlist: hpc_launcher_hostlist.txt'

    assert proc.returncode == 0
