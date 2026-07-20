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
import argparse
import sys
from hpc_launcher.cli import common_args, launch_helpers
from hpc_launcher.schedulers import get_schedulers
from hpc_launcher.schedulers.local import LocalScheduler

import logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Launches a distributed job on the current HPC cluster or cloud."
    )
    common_args.setup_arguments(parser)

    # Grab the rest of the command line to launch
    parser.add_argument("command",
                        nargs='?',
                        default=None,
                        help="Command to be executed")
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments to the command that should be executed",
    )

    args = parser.parse_args()

    launch_helpers.setup_logging(logger, args.verbose)

    # Default the launch directory *before* validation so that the ephemeral
    # rejection of --out/--err/-o/--save-hostlist only fires for genuinely
    # ephemeral runs (no -l, blocking, no batch script) rather than for configs
    # that actually do get a launch directory (finding H3).
    if args.bg and args.launch_dir is None:  # or args.batch_script
        # If running a batch job with no launch directory argument,
        # run in the generated timestamped directory
        args.launch_dir = ""
    if args.launch_dir is None and args.batch_script:
        args.launch_dir = ""
        logger.info(f"Using a predefined launch script needs to run jobs from a launch directory -- automatically setting the -l (--launch-dir) CLI argument")

    # Process special arguments that can autoselect the number of ranks / GPUs
    system = common_args.process_arguments(args, logger)

    # Pick batch scheduler
    scheduler = launch_helpers.select_scheduler(args, logger, system)

    folder_name = None
    script_file = None
    if args.output_script:
        script_file = args.output_script
    elif args.batch_script:
        script_file = args.batch_script
    # (launch-dir defaulting now happens before validation above -- finding H3)
    if args.launch_dir is not None:
        _, folder_name = scheduler.create_launch_folder_name(
            args.command or args.batch_script.rsplit('.', 1)[0], "launch", args.launch_dir
        )

        script_file = scheduler.create_launch_folder(
            folder_name, not args.bg, script_file, args.dry_run
        )

    result = scheduler.launch(
        system,
        folder_name,
        script_file,
        args.command,
        args.args,
        args.override_args,
        not args.bg,
        args.setup_only,
        args.color_stderr,
        args.dry_run,
        args.launch_dir != None and args.save_hostlist,
        args.batch_script != "", # If a batch script is provided don't allow it to be modified
    )

    if result.job_id:
        msg = f"Job ID: {result.job_id} launched from {folder_name}"
        logger.info(msg)
        if not args.verbose:
            print(msg)

    # Propagate the job's exit status: a successful non-blocking submission has
    # no return code yet (job still running) and exits 0.
    sys.exit(result.returncode or 0)

if __name__ == "__main__":
    main()
