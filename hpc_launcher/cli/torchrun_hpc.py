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

    env_list = []
    env_list.append(('MASTER_ADDR', '$(flux hostlist local | /bin/hostlist -n 1)'))
    env_list.append(('MASTER_PORT', '23456'))
    env_list.append(('RANK', '$FLUX_TASK_RANK'))
    env_list.append(('WORLD_SIZE', '$FLUX_JOB_SIZE'))
    env_list.append(('LOCAL_RANK', '$FLUX_TASK_LOCAL_ID'))
    env_list.append(('LOCAL_WORLD_SIZE', '$(($FLUX_JOB_SIZE / $FLUX_JOB_NNODES))'))
    env_list.append(('TOKENIZERS_PARALLELISM', 'false'))
    env_list.append(('TORCH_NCCL_ENABLE_MONITORING', '0'))

    system.extend_environment_variables(env_list)

    # Pick batch scheduler
    scheduler = launch_helpers.select_scheduler(args, logger, system)

    # try:
    #     import torch
    # except (ModuleNotFoundError, ImportError):
    #     print(
    #         'PyTorch is not installed on this system, but is required for torchrun-hpc.'
    #     )
    #     exit(1)

    # print('Verbose:', args.verbose)

    jobid = scheduler.launch(system, args.command, args.args, not args.bg,
                             args.output_script, args.setup_only,
                             args.color_stderr, args.run_from_dir, True)

    if jobid:
        logger.info(f'Job ID: {jobid}')





if __name__ == '__main__':
    main()
