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
import os
import shutil
import pytest

from hpc_launcher.schedulers.local import LocalScheduler
from hpc_launcher.schedulers.slurm import SlurmScheduler
from hpc_launcher.schedulers.flux import FluxScheduler
from hpc_launcher.systems import configure


def test_output_capture_local():
    # Configure scheduler
    system, nodes, procs_per_node = configure.configure_launch(
        None, 1, 1, None, None)
    scheduler = LocalScheduler(nodes, procs_per_node)

    files_before = os.listdir(os.getcwd())

    jobid = scheduler.launch(
        system, 'python',
        [os.path.join(os.path.dirname(__file__), 'output_capture.py')])
    assert jobid is None

    files_after = os.listdir(os.getcwd())
    new_files = set(files_after) - set(files_before)
    assert len(new_files) == 1

    launch_dir = os.path.join(os.getcwd(), new_files.pop())
    assert os.path.isdir(launch_dir)
    assert os.path.isfile(os.path.join(launch_dir, 'out.log'))
    assert os.path.isfile(os.path.join(launch_dir, 'err.log'))
    assert open(os.path.join(launch_dir, 'out.log'), 'r').read() == 'output\n'
    assert open(os.path.join(launch_dir, 'err.log'), 'r').read() == 'error\n'
    shutil.rmtree(launch_dir, ignore_errors=True)


@pytest.mark.parametrize('scheduler_class', (SlurmScheduler, FluxScheduler))
@pytest.mark.parametrize('processes', [1, 2])
def test_output_capture_scheduler(scheduler_class, processes):
    if scheduler_class is SlurmScheduler and not shutil.which('srun'):
        pytest.skip('SLURM not available')

    if scheduler_class is FluxScheduler and not shutil.which('flux'):
        pytest.skip('SLURM not available')

    # Configure scheduler
    system, nodes, procs_per_node = configure.configure_launch(
        None, 1, processes, None, None)
    scheduler = scheduler_class(nodes, procs_per_node)

    files_before = os.listdir(os.getcwd())

    jobid = scheduler.launch(
        system, 'python',
        [os.path.join(os.path.dirname(__file__), 'output_capture.py')])

    files_after = os.listdir(os.getcwd())
    new_files = set(files_after) - set(files_before)
    assert len(new_files) == 1

    launch_dir = os.path.join(os.getcwd(), new_files.pop())
    assert os.path.isdir(launch_dir)
    assert os.path.isfile(os.path.join(launch_dir, 'out.log'))
    assert os.path.isfile(os.path.join(launch_dir, 'err.log'))
    outfile = open(os.path.join(launch_dir, 'out.log'), 'r').read()
    errfile = open(os.path.join(launch_dir, 'err.log'), 'r').read()
    assert outfile.count('output') == processes
    assert errfile.count('error') == processes
    shutil.rmtree(launch_dir, ignore_errors=True)


if __name__ == '__main__':
    test_output_capture_local()
    if shutil.which('srun'):
        test_output_capture_scheduler(SlurmScheduler, 1)
        test_output_capture_scheduler(SlurmScheduler, 2)
    if shutil.which('flux'):
        test_output_capture_scheduler(FluxScheduler, 1)
        test_output_capture_scheduler(FluxScheduler, 2)
