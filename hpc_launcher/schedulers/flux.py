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


@dataclass
class FluxScheduler(Scheduler):

    def launch_command(self, blocking: bool = True) -> list[str]:
        return ['flux', 'run'] if blocking else ['flux', 'batch']

    def launcher_script(self,
                        system: 'System',
                        command: str,
                        args: Optional[list[str]] = None) -> str:
        # String IO

        system = autodetect.autodetect_current_system()
        env_vars = system.environment_variables()
        passthrough_env_vars = system.passthrough_environment_variables()
        # Enable the system to apply some customization to the scheduler instance
        system.customize_scheduler(self)

        script = ''
        if self.out_log_file:
            script += f'#flux --output={self.out_log_file}\n'
        if self.err_log_file:
            script += f'#flux --error={self.err_log_file}\n'

        for k, v in env_vars:
            script += f'export {k}={v}\n'

        script += self.launch_command(True)
        if self.launcher_flags:
            script += f' {" ".join(self.launcher_flags)}'

        script += ' -u'  # Unbuffered
        script += f' -N{self.nodes}'  # --nodes
        script += f' -n{self.nodes * self.procs_per_node}'  # --ntasks

        if self.work_dir:
            script += f' --setattr=system.cwd={self.work_dir}'

        script += ' -o nosetpgrp'

        if self.ld_preloads:
            script += f' --env=LD_PRELOAD={",".join(self.ld_preloads)}'

        for k, v in passthrough_env_vars:
            script += f' --env={k}={v}'

        if self.time_limit is not None:
            script += f' --time={self.time_limit}m'
        if self.job_name:
            script += f' --job-name={self.job_name}'
        if self.partition:
            script += f' --queue={self.partition}'
        if self.account:
            script += f' --account={self.account}'

        script += f' {command}'

        for arg in args:
            script += f' {arg}'

        return script

    def get_job_id(self, output: str) -> Optional[str]:
        # The job ID is the only printout when calling flux batch
        return output.strip()
