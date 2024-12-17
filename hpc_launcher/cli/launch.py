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
from hpc_launcher.cli import common_args
from hpc_launcher.systems import configure
from hpc_launcher.schedulers import get_schedulers
from hpc_launcher.schedulers.local import LocalScheduler
from hpc_launcher.utils import ceildiv

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

    if args.verbose:
        # Another option: format='%(levelname)-7s: %(message)s',
        logging.basicConfig(level=logging.INFO,
                            format='\033[2mhpc-launcher\033[0m: %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING,
                            format='\033[2mhpc-launcher\033[0m: %(message)s')

    logger.info(f'Verbose mode enabled')

    common_args.validate_arguments(args)

    # Set system and launch configuration based on arguments
    system, args.nodes, args.procs_per_node = configure.configure_launch(
        args.queue, args.nodes, args.procs_per_node, args.gpus_at_least,
        args.gpumem_at_least)

    # Pick batch scheduler
    if args.local:
        scheduler_class = LocalScheduler
    elif args.scheduler:
        scheduler_class = get_schedulers()[args.scheduler]
    else:
        scheduler_class = system.preferred_scheduler
    logger.info(f'Using {scheduler_class.__name__}')

    scheduler_args = common_args.create_scheduler_arguments(**vars(args))
    scheduler = scheduler_class(**scheduler_args)

    logger.info(
        f'system parameters: node={scheduler.nodes} ppn={scheduler.procs_per_node}'
    )

    jobid = scheduler.launch(system, args.command, args.args, not args.bg,
                             args.output_script, args.setup_only,
                             args.color_stderr, args.run_from_dir)

    if jobid:
        logger.info(f'Job ID: {jobid}')


if __name__ == '__main__':
    main()
