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
"""
Common arguments for CLI utilities.
"""
import argparse
from hpc_launcher.schedulers import get_schedulers


def setup_arguments(parser: argparse.ArgumentParser):
    """
    Adds common arguments for CLI utilities.

    :param parser: The ``argparse`` parser of the tool.
    """
    parser.add_argument('--verbose',
                        '-v',
                        action='store_true',
                        default=False,
                        help='Run in verbose mode')

    # Job size arguments
    group = parser.add_argument_group(
        'Job size',
        'Determines the number of nodes, accelerators, and ranks for the job')
    group.add_argument('-N',
                       '--nodes',
                       type=int,
                       default=0,
                       help='Specifies the number of requested nodes')
    group.add_argument(
        '-n',
        '--procs-per-node',
        type=int,
        default=None,
        help='Specifies the number of requested processes per node')

    group.add_argument(
        '-q',
        '--queue',
        default=None,
        help='Specifies the queue to use')

    # Constraints
    group.add_argument(
        '-g',
        '--gpus-at-least',
        type=int,
        default=0,
        help='Specifies the total number of accelerators requested. Mutually '
        'exclusive with "--procs-per-node" and "--nodes"')

    group.add_argument('--gpumem-at-least',
                       type=int,
                       default=0,
                       help='A constraint that specifies how much accelerator '
                       'memory is needed for the job (in gigabytes). If this '
                       'flag is specified, the number of nodes and processes '
                       'are not necessary. Requires the system to be '
                       'registered with the launcher.')

    group.add_argument('--local',
                       action='store_true',
                       default=False,
                       help='Run locally (i.e., one process without a batch '
                       'scheduler)')

    # Schedule
    group = parser.add_argument_group(
        'Schedule', 'Arguments that determine when a job will run')

    # Blocking
    group.add_argument(
        '--bg',
        action='store_true',
        default=False,
        help='If set, the job will be run in the background. Otherwise, the '
        'launcher will wait for the job to start and forward the outputs to '
        'the console')

    group.add_argument('--scheduler',
                       type=str,
                       default=None,
                       choices=get_schedulers().keys(),
                       help='If set, overrides the default batch scheduler')

    group = parser.add_argument_group('Logging', 'Logging parameters')
    group.add_argument(
        '--out',
        default=None,
        help='Capture standard output to a log file. If not given, only prints '
        'out logs to the console')
    group.add_argument(
        '--err',
        default=None,
        help='Capture standard error to a log file. If not given, only prints '
        'out logs to the console')
    group.add_argument(
        '--color-stderr',
        action='store_true',
        default=False,
        help='If True, uses terminal colors to color the standard error '
        'outputs in red. This does not affect the output files')

    group = parser.add_argument_group('Script',
                                      'Batch scheduler script parameters')

    group.add_argument(
        '-o',
        '--output-script',
        default=None,
        help='Output job setup script file. If not given, uses a temporary file'
    )

    group.add_argument(
        '--setup-only',
        action='store_true',
        default=False,
        help='If set, the launcher will only write the job setup script file, '
        'without scheduling it.')


def validate_arguments(args: argparse.Namespace):
    """
    Validation checks for the commong arguments. Raises exceptions on failure.

    :param args: The parsed arguments.
    """
    # TODO(later): Convert some mutual exclusive behavior to constraints on
    #              number of nodes/ranks
    # if (args.nodes and not args.procs_per_node) or (not args.nodes
    #                                                 and args.procs_per_node):
    if (not args.nodes and not args.gpus_at_least) or (not args.nodes
                                                    and not args.gpumem_at_least):
        raise ValueError(
            'One of the following flags has to be set: --nodes, --gpus-at-least, or --gpumem-at-least')
    if args.gpus_at_least and args.procs_per_node:
        raise ValueError('The --gpus-at-least and --procs-per-node flags '
                         'are mutually exclusive')
    if args.gpumem_at_least and args.procs_per_node:
        raise ValueError('The --gpumem-at-least and --procs-per-node flags '
                         'are mutually exclusive')
    if args.gpumem_at_least and args.gpus_at_least:
        raise ValueError('The --gpumem-at-least and --gpus-at-least flags '
                         'are mutually exclusive')
    if (not args.procs_per_node and not args.gpumem_at_least
            and not args.gpus_at_least and not args.local):
        raise ValueError(
            'Number of processes must be provided via --procs-per-node, '
            '--gpus-at-least, or constraints such as --gpumem-at-least')
    if args.local and args.bg:
        raise ValueError('"--local" jobs cannot be run in the background')
    if args.local and args.scheduler:
        raise ValueError('The --local and --scheduler flags are mutually '
                         'exclusive')
    if args.setup_only and not args.output_script:
        raise ValueError('Cannot use "--setup-only" without an output script '
                         'file. Use -o to save the script to a file.')
