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
    from hpc_launcher.systems import System

from hpc_launcher.schedulers.scheduler import Scheduler
from hpc_launcher.systems import autodetect

def _time_string(minutes):
    """Time D-hh:mm:ss format."""
    minutes = max(minutes, 0)
    seconds = int(round((minutes % 1) * 60))
    hours, minutes = divmod(int(minutes), 60)
    days, hours = divmod(hours, 24)
    return f'{days}-{hours:02}:{minutes:02}:{seconds:02}'

@dataclass
class SlurmScheduler(Scheduler):
    def launch_command(self, blocking: bool = True) -> list[str]:
        return 'srun' if blocking else 'sbatch'

    def launcher_script(self, system: 'System', command: str,
                        args: Optional[list[str]] = None) -> str:
        # String IO

        env_vars = system.environment_variables()
        passthrough_env_vars = system.passthrough_environment_variables()
        # Enable the system to apply some customization to the scheduler instance
        system.customize_scheduler(self)

        script = ''
        header_lines = ''
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

        if self.work_dir:
            cmd_string += f' --setattr=system.cwd={self.work_dir}'

        cmd_string += ' -o nosetpgrp'

        if self.work_dir is not None:
            cmd_string += f'--chdir={self.work_dir}'
            header_lines += f'#SBATCH --chdir={self.work_dir}\n'

        if self.ld_preloads:
            cmd_string += f'--export=ALL,LD_PRELOAD={",".join(self.ld_preloads)}'

        for k,v in passthrough_env_vars:
            cmd_string += f' --env={k}={v}'

        if self.time_limit is not None:
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
        for k,v in env_vars:
            script += f'export {k}={v}\n'

        script += self.launch_command(True)
        script += cmd_string
        script += f' {command}'

        for arg in args:
            script += f' {arg}'

        return script
