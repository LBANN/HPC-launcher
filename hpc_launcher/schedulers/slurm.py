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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
from io import StringIO
import os

if TYPE_CHECKING:
    # If type-checking, import the other class
    from hpc_launcher.systems.system import System

from hpc_launcher.schedulers.scheduler import Scheduler

import logging

logger = logging.getLogger(__name__)

def _time_string(minutes):
    """Time D-hh:mm:ss format."""
    minutes = max(minutes, 0)
    seconds = int(round((minutes % 1) * 60))
    hours, minutes = divmod(int(minutes), 60)
    days, hours = divmod(hours, 24)
    return f'{days}-{hours:02}:{minutes:02}:{seconds:02}'


def select_interactive_or_batch(tmp: str,
                                header: StringIO,
                                cmd_args: list[str],
                                blocking: bool = True) -> (str, list[str]):
    if blocking:
        cmd_args += [tmp]
    else:
        header.write(f'#SBATCH {tmp}\n')
    return

@dataclass
class SlurmScheduler(Scheduler):

    def build_command_string_and_batch_script(self,
                                              system: 'System',
                                              blocking: bool = True) -> (str, list[str]):

        env_vars = system.environment_variables()
        passthrough_env_vars = system.passthrough_environment_variables()
        # Enable the system to apply some customization to the scheduler instance
        system.customize_scheduler(self)

        header = StringIO()
        header.write('#!/bin/sh\n')
        cmd_args = []

        cmd_args = []
        if self.out_log_file and not blocking:
            header.write(f'#SBATCH --output={self.out_log_file}\n')
        if self.err_log_file and not blocking:
            header.write(f'#SBATCH --error={self.err_log_file}\n')

        # Unbuffered output - Only pass to srun
        if blocking:
            tmp = '-u'
            cmd_args += [tmp]

        # Number of Nodes
        tmp = f'--nodes={self.nodes}'
        cmd_args += [tmp]
        if not blocking:
            header.write(f'#SBATCH {tmp}\n')

        # Total number of Tasks / Processes
        tmp = f'--ntasks={self.nodes * self.procs_per_node}'
        cmd_args += [tmp]
        if not blocking:
            header.write(f'#SBATCH {tmp}\n')

        # Number of Tasks per node
        tmp = f'--ntasks-per-node={self.nodes * self.procs_per_node}'
        cmd_args += [tmp]
        if not blocking:
            header.write(f'#SBATCH {tmp}\n')

        if self.work_dir:
            tmp = f'--chdir={os.path.abspath(self.work_dir)}'
            select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.ld_preloads:
            tmp = f'--export=ALL,LD_PRELOAD={",".join(self.ld_preloads)}'
            select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.time_limit is not None:
            tmp = f'--time={_time_string(self.time_limit)}'
            select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.job_name:
            tmp = f'--job-name={self.job_name}'
            select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.queue:
            tmp = f'--partition={self.queue}'
            select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.account:
            tmp = f'--account={self.account}'
            select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.reservation:
            tmp = f'--reservation={self.reservation}'
            select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.launcher_flags:
            for f in self.launcher_flags:
                select_interactive_or_batch(f, header, cmd_args, blocking)

        for k, v in env_vars:
            header.write(f'export {k}={v}\n')

        for k, v in passthrough_env_vars:
            if not blocking:
                cmd_args += [f' --env={k}={v}']
            else:
                header += f'export {k}={v}\n'

        return (header.getvalue(), cmd_args)

    def launch_command(self, system: 'System', blocking: bool = True) -> list[str]:
        # Launch command only use the cmd_args to construct the shell script to be launched
        (header_lines, cmd_args) = self.build_command_string_and_batch_script(system, blocking)

        if not blocking:
            return ['sbatch'] + cmd_args

        return ['srun'] + cmd_args

    def launcher_script(self,
                        system: 'System',
                        command: str,
                        args: Optional[list[str]] = None,
                        blocking: bool = True) -> str:

        script = ''
        # Launch command only use the cmd_args to construct the shell script to be launched
        (header_lines, cmd_args) = self.build_command_string_and_batch_script(system, blocking)

        # Configure header and command line with Slurm job options
        script += header_lines
        script += '\n'

        if not blocking:
            script += 'srun -u '
            script += ' '.join(cmd_args)
            script += ' '

        script += f'{command}'

        for arg in args:
            script += f' {arg}'

        script += '\n'

        return script

    def get_job_id(self, output: str) -> Optional[str]:
        # The job ID is the last number in the printout
        last_line = output.strip().split('\n')[-1].strip()
        if last_line.startswith('Submitted batch job'):
            return last_line.split(' ')[-1]
        return None
