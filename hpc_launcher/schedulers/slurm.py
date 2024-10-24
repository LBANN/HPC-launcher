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

        system = autodetect.autodetect_current_system()
        env_vars = system.environment_variables()
        passthrough_env_vars = system.passthrough_environment_variables()
        # Enable the system to apply some customization to the scheduler instance
        system.customize_scheduler(self)

        script = ''
        if self.out_log_file:
            script += f'#SBATCH --output={self.out_log_file}\n'
        if self.err_log_file:
            script += f'#SBATCH --error={self.err_log_file}\n'

        # Configure header with Slurm job options
        # self.add_header_line(f'#SBATCH --chdir={self.work_dir}')
        # self.add_header_line(f'#SBATCH --nodes={self.nodes}')
        # self.add_header_line(f'#SBATCH --ntasks={self.nodes * self.procs_per_node}')
        # self.add_header_line(f'#SBATCH --ntasks-per-node={self.procs_per_node}')
        # if self.time_limit is not None:
        #     self.add_header_line(f'#SBATCH --time={_time_string(self.time_limit)}')
        # if self.job_name:
        #     self.add_header_line(f'#SBATCH --job-name={self.job_name}')
        # if self.partition:
        #     self.add_header_line(f'#SBATCH --partition={self.partition}')
        # if self.account:
        #     self.add_header_line(f'#SBATCH --account={self.account}')

        for k,v in env_vars:
            script += f'export {k}={v}\n'

        script += self.launch_command(True)
        if self.launcher_flags:
            script += f' {" ".join(self.launcher_flags)}'

        script += ' -u'  # Unbuffered
        script += f'--nodes={self.nodes}'
        script += f'--ntasks={self.nodes * self.procs_per_node}'
        script += f'--ntasks-per-node={self.procs_per_node}'

        if self.work_dir:
            script += f' --setattr=system.cwd={self.work_dir}'

        script += ' -o nosetpgrp'

        if self.work_dir is not None:
            script += f'--chdir={self.work_dir}'

        if self.ld_preloads:
            script += f'--export=ALL,LD_PRELOAD={",".join(self.ld_preloads)}'

        for k,v in passthrough_env_vars:
            script += f' --env={k}={v}'

        if self.time_limit is not None:
            script += f' --time={self.time_limit}m'
        if self.job_name:
            script += f' --job-name={self.job_name}'
        if self.partition:
            script += f' --partition={self.partition}'
        if self.account:
            script += f' --account={self.account}'

        script += f' {command}'

        for arg in args:
            script += f' {arg}'

        return script
