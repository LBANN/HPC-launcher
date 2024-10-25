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
from hpc_launcher.systems import autodetect
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
                            format='hpc-launcher: %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING,
                            format='hpc-launcher: %(message)s')

    logger.info(f'Verbose: {args.verbose}')

    system = autodetect.autodetect_current_system()
    logger.info(
        f'Detected system: {system.system_name} [{type(system).__name__}-class]'
    )
    system_params = system.system_parameters(args.queue)

    # If the user requested a specific number of process per node, honor that
    procs_per_node = args.procs_per_node

    # Otherwise ...
    # If there is a valid set of system parameters, try to fill in the blanks provided by the user
    if system_params is not None:
        procs_per_node = system_params.procs_per_node()
        if args.gpus_at_least > 0:
            args.nodes = ceildiv(args.gpus_at_least, procs_per_node)
        elif args.gpumem_at_least > 0:
            num_gpus = ceildiv(args.gpumem_at_least, system_params.mem_per_gpu)
            args.nodes = ceildiv(num_gpus, procs_per_node)
            if args.nodes == 1:
                procs_per_node = num_gpus

    common_args.validate_arguments(args)
    # Pick batch scheduler
    if args.local:
        scheduler_class = LocalScheduler
    elif args.scheduler:
        scheduler_class = get_schedulers()[args.scheduler]
    else:
        scheduler_class = system.preferred_scheduler
    logger.info(f'Using {scheduler_class.__name__}')

    scheduler = scheduler_class(args.nodes, procs_per_node, partition=args.queue)

    if args.out:
        scheduler.out_log_file = f'{args.out}'
    if args.err:
        scheduler.err_log_file = f'{args.err}'

    logger.info(
        f'system parameters: node={scheduler.nodes} ppn={scheduler.procs_per_node}'
    )

    jobid = scheduler.launch(system, args.command, args.args, not args.bg,
                             args.output_script, args.setup_only,
                             args.color_stderr)

    if jobid:
        logger.info(f'Job ID: {jobid}')


if __name__ == '__main__':
    main()
