# Copyright (c) 2014-2025, Lawrence Livermore National Security, LLC.
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
import sys

from hpc_launcher.schedulers.local import LocalScheduler
from hpc_launcher.schedulers.slurm import SlurmScheduler
from hpc_launcher.schedulers.flux import FluxScheduler
from hpc_launcher.schedulers.lsf import LSFScheduler
from hpc_launcher.systems import autodetect, configure
from hpc_launcher.systems.lc.sierra_family import Sierra

@pytest.mark.parametrize("launch_dir", ["", "."])
def test_output_capture_local(launch_dir: bool):
    # Configure scheduler
    system, nodes, procs_per_node, gpus_per_proc = configure.configure_launch(None, 1, 1, 1, None, None)
    scheduler = LocalScheduler(nodes, procs_per_node, gpus_per_proc)

    command = sys.executable
    script = "output_capture.py"
    # Set the request for the launch dir to the empty string to use a auto-generated folder
    _, launch_dir = scheduler.create_launch_folder_name(
        command, "launch", ""
    )

    script_file = scheduler.create_launch_folder(launch_dir, True)

    jobid = scheduler.launch(
        system,
        launch_dir,
        script_file,
        command,
        [os.path.join(os.path.dirname(__file__), "output_capture.py")],
    )

    assert os.path.exists(launch_dir)
    assert os.path.isdir(launch_dir)
    assert os.path.isfile(os.path.join(launch_dir, "out.log"))
    assert os.path.isfile(os.path.join(launch_dir, "err.log"))
    assert open(os.path.join(launch_dir, "out.log"), "r").read() == "output\n"
    assert open(os.path.join(launch_dir, "err.log"), "r").read() == "error\n"
    if launch_dir != "" or launch_dir != ".":
        shutil.rmtree(launch_dir, ignore_errors=True)
    else:
        os.unlink(f"{launch_dir}/out.log")
        os.unlink(f"{launch_dir}/err.log")
        os.unlink(f"{launch_dir}/launch.sh")


@pytest.mark.parametrize(
    "scheduler_class", (SlurmScheduler, FluxScheduler, LSFScheduler)
)
@pytest.mark.parametrize("processes", [1, 2])
def test_output_capture_scheduler(scheduler_class, processes):
    if scheduler_class is SlurmScheduler and not shutil.which("srun"):
        pytest.skip("SLURM not available")

    if scheduler_class is FluxScheduler and (
        not shutil.which("flux") or not os.path.exists("/run/flux/local")
    ):
        pytest.skip("FLUX not available")

    if scheduler_class is SlurmScheduler and (
        shutil.which("flux") and os.path.exists("/run/flux/local")
    ):
        pytest.skip("Emulated SLURM on FLUX system - don't test - output redirect is bad")

    if scheduler_class is LSFScheduler and not shutil.which("jsrun"):
        pytest.skip("LSF not available")

    # Configure scheduler
    system, nodes, procs_per_node, gpus_per_proc = configure.configure_launch(
        None, 1, processes, 1, None, None
    )
    scheduler = scheduler_class(nodes, procs_per_node, gpus_per_proc)
    # Reset class
    scheduler.submit_only_args.clear()
    scheduler.run_only_args.clear()
    scheduler.common_launch_args.clear()

    command = sys.executable
    _, launch_dir = scheduler.create_launch_folder_name(command, "launch", "")

    script_file = scheduler.create_launch_folder(launch_dir, True)

    jobid = scheduler.launch(
        system,
        launch_dir,
        script_file,
        command,
        [os.path.join(os.path.dirname(__file__), "output_capture.py")],
    )

    assert os.path.exists(launch_dir)
    assert os.path.isdir(launch_dir)
    assert os.path.isfile(os.path.join(launch_dir, "out.log"))
    assert os.path.isfile(os.path.join(launch_dir, "err.log"))
    outfile = open(os.path.join(launch_dir, "out.log"), "r").read()
    errfile = open(os.path.join(launch_dir, "err.log"), "r").read()
    assert outfile.count("output") == processes
    if (
        (scheduler_class is LSFScheduler or scheduler_class is SlurmScheduler)
        and isinstance(autodetect.autodetect_current_system(), Sierra)
    ) and not os.getenv("LSB_HOSTS"):
        # bsub -Is has a bad behavior where the error stream is appended to the output stream
        assert outfile.count("error") == processes
    else:
        assert errfile.count("error") == processes
    shutil.rmtree(launch_dir, ignore_errors=True)


if __name__ == "__main__":
    test_output_capture_local(False)
    test_output_capture_local(True)
    if shutil.which("srun"):
        test_output_capture_scheduler(SlurmScheduler, 1)
        test_output_capture_scheduler(SlurmScheduler, 2)
    if shutil.which("flux"):
        test_output_capture_scheduler(FluxScheduler, 1)
        test_output_capture_scheduler(FluxScheduler, 2)
    if shutil.which("jsrun"):
        test_output_capture_scheduler(LSFScheduler, 1)
        test_output_capture_scheduler(LSFScheduler, 2)
