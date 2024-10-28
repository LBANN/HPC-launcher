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

if TYPE_CHECKING:
    # If type-checking, import the other class
    from hpc_launcher.systems.system import System

from hpc_launcher.schedulers.scheduler import Scheduler


def _time_string(minutes):
    """Time D-hh:mm:ss format."""
    minutes = max(minutes, 0)
    seconds = int(round((minutes % 1) * 60))
    hours, minutes = divmod(int(minutes), 60)
    days, hours = divmod(hours, 24)
    return f'{days}-{hours:02}:{minutes:02}:{seconds:02}'


@dataclass
class SlurmScheduler(Scheduler):

    def build_command_string_and_batch_script(self,
                                              system: 'System') -> (str, str):

        env_vars = system.environment_variables()
        passthrough_env_vars = system.passthrough_environment_variables()
        # Enable the system to apply some customization to the scheduler instance
        system.customize_scheduler(self)

        header_lines = '#!/bin/sh\n'
        cmd_string = ''

        return (header_lines, cmd_string)

    def launch_command(self, system: 'System', blocking: bool = True) -> list[str]:
        if not blocking:
            return ['sbatch']

        cmd_string = ['srun']
        if self.launcher_flags:
            cmd_string.extend(self.launcher_flags)
        cmd_string += ['-u']  # Unbuffered
        cmd_string += [f'--nodes={self.nodes}']
        cmd_string += [f'--ntasks={self.nodes * self.procs_per_node}']
        cmd_string += [f'--ntasks-per-node={self.procs_per_node}']
        if self.ld_preloads:
            cmd_string += [
                f'--export=ALL,LD_PRELOAD={",".join(self.ld_preloads)}'
            ]
        if self.time_limit:
            cmd_string += [f'--time={self.time_limit}m']
        if self.job_name:
            cmd_string += [f'--job-name={self.job_name}']
        if self.partition:
            cmd_string += [f'--partition={self.partition}']
        if self.account:
            cmd_string += [f'--account={self.account}']

        return cmd_string

    def launcher_script(self,
                        system: 'System',
                        command: str,
                        args: Optional[list[str]] = None,
                        blocking: bool = True) -> str:
        env_vars = system.environment_variables()
        passthrough_env_vars = system.passthrough_environment_variables()
        # Enable the system to apply some customization to the scheduler instance
        system.customize_scheduler(self)

        script = ''
        header_lines = '#!/bin/sh\n'
        cmd_string = ''
        if self.out_log_file:
            header_lines += f'#SBATCH --output={self.out_log_file}\n'
        if self.err_log_file:
            header_lines += f'#SBATCH --error={self.err_log_file}\n'

        if self.launcher_flags:
            cmd_string += f' {" ".join(self.launcher_flags)}'

        cmd_string += ' -u'  # Unbuffered

        cmd_string += f' --nodes={self.nodes}'
        header_lines += f'#SBATCH --nodes={self.nodes}\n'

        cmd_string += f' --ntasks={self.nodes * self.procs_per_node}'
        header_lines += f'#SBATCH --ntasks={self.nodes * self.procs_per_node}\n'

        cmd_string += f' --ntasks-per-node={self.procs_per_node}'
        header_lines += f'#SBATCH --ntasks-per-node={self.procs_per_node}\n'

        #cmd_string += ' -o nosetpgrp'

        if self.work_dir:
            if not blocking:
                cmd_string += f'--chdir={self.work_dir}'
            else:
                header_lines += f'cd {self.work_dir}\n'
            header_lines += f'#SBATCH --chdir={self.work_dir}\n'

        if self.ld_preloads:
            cmd_string += f'--export=ALL,LD_PRELOAD={",".join(self.ld_preloads)}'

        for k, v in passthrough_env_vars:
            if not blocking:
                cmd_string += f' --env={k}={v}'
            else:
                header_lines += f'export {k}={v}\n'

        if self.time_limit:
            cmd_string += f' --time={self.time_limit}m'
            header_lines += f'#SBATCH --time={_time_string(self.time_limit)}\n'

        if self.job_name:
            cmd_string += f' --job-name={self.job_name}'
            header_lines += f'#SBATCH --job-name={self.job_name}\n'
        if self.partition:
            cmd_string += f' --partition={self.partition}'
            header_lines += f'#SBATCH --partition={self.partition}\n'
        if self.account:
            cmd_string += f' --account={self.account}'
            header_lines += f'#SBATCH --account={self.account}\n'

        # Configure header and command line with Slurm job options
        script += header_lines
        for k, v in env_vars:
            script += f'export {k}={v}\n'

        if not blocking:
            script += ' '.join(self.launch_command(True))
            script += cmd_string
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
