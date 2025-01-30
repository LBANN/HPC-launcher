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
import argparse
from hpc_launcher.cli import common_args, launch_helpers
from hpc_launcher.schedulers import get_schedulers
from hpc_launcher.schedulers.scheduler import Scheduler
from hpc_launcher.schedulers.local import LocalScheduler

try:
    import mpi4py

    mpi = True
except (ImportError, ModuleNotFoundError):
    mpi = None

import logging
import os
import shutil
import sys

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="A wrapper script that launches and runs distributed PyTorch on HPC systems."
    )
    common_args.setup_arguments(parser)
    parser.add_argument(
        "-r",
        "--rdv",
        default=None,
        help="Specifies rendezvous protocol to use: mpi | tcp",
    )

    parser.add_argument(
        "--fraction-max-gpu-mem",
        type=float,
        default=None,
        help="Use the torch.cuda.set_per_process_memory_fraction "
        "to limit how much GPU memory can be allocated.",
    )

    # Grab the rest of the command line to launch
    parser.add_argument("command", help="Command to be executed")
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments to the command that should be executed",
    )

    args = parser.parse_args()

    launch_helpers.setup_logging(logger, args.verbose)

    # Process special arguments that can autoselect the number of ranks / GPUs
    system = common_args.process_arguments(args, logger)
    # Pick batch scheduler
    scheduler = launch_helpers.select_scheduler(args, logger, system)

    if args.rdv is None:
        if mpi and not isinstance(scheduler, LocalScheduler):
            env_list = scheduler.setup_rendezvous_protocol("mpi")
        else:
            env_list = scheduler.setup_rendezvous_protocol("tcp")
    else:
        if args.rdv == "mpi":
            if not mpi:
                raise Exception("MPI rendezvous requested but not available")
            else:
                env_list = scheduler.setup_rendezvous_protocol("mpi")
        elif args.rdv == "tcp":
            env_list = scheduler.setup_rendezvous_protocol("tcp")
        else:
            raise Exception(f"Unknown rendezvous {args.rdv} requested.")

    if args.fraction_max_gpu_mem and args.fraction_max_gpu_mem != 1.0:
        env_list.append(("TORCHRUN_HPC_MAX_GPU_MEM", args.fraction_max_gpu_mem))
    else:
        if system.active_system_params:
            env_list.append(
                (
                    "TORCHRUN_HPC_MAX_GPU_MEM",
                    system.active_system_params.fraction_max_gpu_mem,
                )
            )

    system.extend_environment_variables(env_list)

    try:
        import torch
    except (ModuleNotFoundError, ImportError):
        print(
            "PyTorch is not installed on this system, but is required for torchrun-hpc."
        )
        exit(1)

    _, folder_name = scheduler.create_launch_folder_name(
        args.command, "torchrun_hpc", args.no_launch_dir
    )

    script_file = scheduler.create_launch_folder(
        folder_name, not args.bg, args.output_script, args.run_from_launch_dir
    )

    trampoline_file = "torchrun_hpc_trampoline.py"

    if os.path.exists(folder_name):
        copied_trampoline_file = folder_name + "/" + trampoline_file
        package_path = os.path.dirname(os.path.abspath(__file__))
        shutil.copy(
            os.path.join(package_path, "..", "torch", trampoline_file),
            copied_trampoline_file,
        )

    command = sys.executable
    launch_args = [
        "-u",
        f"{os.path.abspath(folder_name)}/{trampoline_file}",
        os.path.abspath(args.command),
    ]
    launch_args += args.args

    logger.info(f"Running job in directory: {folder_name}")

    jobid = scheduler.launch(
        system,
        folder_name,
        script_file,
        command,
        launch_args,
        not args.bg,
        # args.output_script,
        args.setup_only,
        args.color_stderr,
        args.run_from_launch_dir,
        (args.save_hostlist or args.verbose),
    )

    if jobid:
        logger.info(f"Job ID: {jobid}")


if __name__ == "__main__":
    main()
