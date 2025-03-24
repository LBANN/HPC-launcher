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
"""
Helper functions for CLI utilities.
"""
import argparse
from hpc_launcher.cli import common_args
from hpc_launcher.schedulers import get_schedulers
from hpc_launcher.schedulers.scheduler import Scheduler
from hpc_launcher.schedulers.local import LocalScheduler
from hpc_launcher.systems.system import System, GenericSystem
from hpc_launcher.systems import autodetect
import logging
import sys


def setup_logging(logger: logging.Logger, verbose: bool):
    if verbose:
        # Another option: format='%(levelname)-7s: %(message)s',
        logging.basicConfig(
            level=logging.INFO, format="\033[2mhpc-launcher\033[0m: %(message)s"
        )
    else:
        logging.basicConfig(
            level=logging.WARNING, format="\033[2mhpc-launcher\033[0m: %(message)s"
        )

    logger.info(f"Verbose mode enabled")


# Using the args and an autodetected system, select a scheduler
def select_scheduler(
    args: argparse.Namespace, logger: logging.Logger, system: System
) -> Scheduler:
    # Pick batch scheduler
    if args.local:
        scheduler_class = LocalScheduler
    elif args.scheduler:
        scheduler_class = get_schedulers()[args.scheduler]
    else:
        scheduler_class = system.preferred_scheduler
    logger.info(f"Using {scheduler_class.__name__}")

    scheduler_args = common_args.create_scheduler_arguments(**vars(args))
    scheduler = scheduler_class(**scheduler_args)
    scheduler.command_line = sys.argv

    logger.info(
        f"system parameters: node={scheduler.nodes} ppn={scheduler.procs_per_node}"
    )

    return scheduler
