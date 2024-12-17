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
from hpc_launcher.schedulers.local import LocalScheduler

import logging
import os
import shutil

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description=
        'A wrapper script that launches and runs distributed PyTorch on HPC systems.'
    )
    common_args.setup_arguments(parser)

    # Grab the rest of the command line to launch
    parser.add_argument('command', help='Command to be executed')
    parser.add_argument(
        'args',
        nargs=argparse.REMAINDER,
        help='Arguments to the command that should be executed')

    args = parser.parse_args()

    launch_helpers.setup_logging(logger, args.verbose)


    # Process special arguments that can autoselect the number of ranks / GPUs
    system = common_args.process_arguments(args, logger)
    # Pick batch scheduler
    scheduler = launch_helpers.select_scheduler(args, logger, system)

    env_list = scheduler.setup_rendezvous_protocol('port')

    system.extend_environment_variables(env_list)

    try:
        import torch
    except (ModuleNotFoundError, ImportError):
        print(
            'PyTorch is not installed on this system, but is required for torchrun-hpc.'
        )
        exit(1)

    command_as_folder_name, folder_name = scheduler.create_launch_folder_name(args.command,
                                                                                'torchrun_hpc',)

    script_file = scheduler.create_launch_folder(folder_name,
                                                 not args.bg,
                                                 args.output_script,
                                                 args.run_from_dir)

    stub_file = 'torchrun_hpc_' + command_as_folder_name

    if os.path.exists(folder_name):
        copied_stub_file = folder_name + '/' +  stub_file
        package_path = os.path.dirname(os.path.abspath(__file__))
        shutil.copy(os.path.join(package_path, 'torchrun_hpc_stub.py'), copied_stub_file)

    command = f'python3 -u {os.path.abspath(folder_name)}/{stub_file} ' + os.path.abspath(args.command)

    jobid = scheduler.launch(system, folder_name, script_file,
                             command, args.args, not args.bg,
                             # args.output_script,
                             args.setup_only,
                             args.color_stderr, args.run_from_dir)

    if jobid:
        logger.info(f'Job ID: {jobid}')





if __name__ == '__main__':
    main()
