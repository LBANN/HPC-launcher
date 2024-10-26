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

if TYPE_CHECKING:
    # If type-checking, import the other class
    from hpc_launcher.systems import System

from hpc_launcher.schedulers.scheduler import Scheduler
from hpc_launcher.systems import autodetect

def select_interactive_or_batch(tmp: str,
                                header: StringIO(),
                                cmd_args: list[str],
                                blocking: bool = True) -> (str, list[str]):
    if blocking:
        cmd_args += [tmp]
    else:
        header.write(f'# FLUX: {tmp}\n')
    return(header, cmd_args)

@dataclass
class FluxScheduler(Scheduler):

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
        if self.out_log_file:
            header += f'# FLUX: --output={self.out_log_file}\n'
        if self.err_log_file:
            header += f'# FLUX: --error={self.err_log_file}\n'

        # Unbuffered output
        tmp = '-u'
        (header, cmd_args) = select_interactive_or_batch(tmp, header, cmd_args, blocking)

        # Number of Nodes
        tmp = f'-N{self.nodes}'
        (header, cmd_args) = select_interactive_or_batch(tmp, header, cmd_args, blocking)

        # Total number of Tasks / Processes
        tmp = f'-n{self.nodes * self.procs_per_node}'
        (header, cmd_args) = select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.work_dir:
            tmp = f'--setattr=system.cwd={self.work_dir}'
            (header, cmd_args) = select_interactive_or_batch(tmp, header, cmd_args, blocking)

        tmp = '-o nosetpgrp'
        (header, cmd_args) = select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.ld_preloads:
            tmp = f'--env=LD_PRELOAD={",".join(self.ld_preloads)}'
            (header, cmd_args) = select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.time_limit is not None:
            tmp = f'--time={self.time_limit}m'
            (header, cmd_args) = select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.job_name:
            tmp = f'--job-name={self.job_name}'
            (header, cmd_args) = select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.partition:
            tmp = f'--queue={self.partition}'
            (header, cmd_args) = select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.account:
            tmp = f'--account={self.account}'
            (header, cmd_args) = select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.launcher_flags:
            tmp = f'{" ".join(self.launcher_flags)}'
            (header, cmd_args) = select_interactive_or_batch(tmp, header, cmd_args, blocking)

        for k, v in env_vars:
            header.write(f'export {k}={v}\n')

        for k, v in passthrough_env_vars:
            if not blocking:
                cmd_args += [f' --env={k}={v}']
            else:
                header += f'export {k}={v}\n'

        return (header.getvalue(), cmd_args)

    def launch_command(self, system: 'System', blocking: bool = True) -> list[str]:
        if not blocking:
            return ['flux', 'batch']

        # Launch command only use the cmd_args to construct the shell script to be launched
        (header_lines, cmd_args) = self.build_command_string_and_batch_script(system, blocking)

        return ['flux', 'run'] + cmd_args

    def launcher_script(self,
                        system: 'System',
                        command: str,
                        args: Optional[list[str]] = None,
                        blocking: bool = True) -> str:

        script = ''
        # Launcher script only use the header_lines to construct the shell script to be launched
        (header_lines, cmd_string) = self.build_command_string_and_batch_script(system, blocking)
        script += header_lines
        script += '\n'

        if not blocking:
            script += 'flux run '

        script += f'{command}'

        for arg in args:
            script += f' {arg}'

        script += '\n'

        return script

    def get_job_id(self, output: str) -> Optional[str]:
        # The job ID is the only printout when calling flux batch
        return output.strip()
