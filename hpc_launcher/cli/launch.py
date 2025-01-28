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

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description=
        'Launches a distributed job on the current HPC cluster or cloud.')
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

    _, folder_name = scheduler.create_launch_folder_name(args.command, 'launch', args.no_launch_dir)

    script_file = scheduler.create_launch_folder(folder_name,
                                                 not args.bg,
                                                 args.output_script,
                                                 args.run_from_launch_dir)

    jobid = scheduler.launch(system, folder_name, script_file,
                             args.command, args.args, not args.bg,
                             args.setup_only,
                             args.color_stderr, args.run_from_launch_dir,
                             args.save_hostlist)

    if jobid:
        logger.info(f'Job ID: {jobid}')


if __name__ == '__main__':
    main()
