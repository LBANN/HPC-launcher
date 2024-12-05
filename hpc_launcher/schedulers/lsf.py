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

def select_interactive_or_batch(tmp: str,
                                header: StringIO,
                                cmd_args: list[str],
                                blocking: bool = True) -> (str, list[str]):
    if blocking:
        cmd_args += [tmp]
    else:
        header.write(f'#BSUB {tmp}\n')
    return


@dataclass
class LSFScheduler(Scheduler):
    def build_command_string_and_batch_script(self,
                                              system: 'System',
                                              blocking: bool = True) -> (str, list[str], list[str]):

        env_vars = system.environment_variables()
        passthrough_env_vars = system.passthrough_environment_variables()
        # Enable the system to apply some customization to the scheduler instance
        system.customize_scheduler(self)

        header = StringIO()
        header.write('#!/bin/sh\n')
        cmd_args = []
        parallel_run_args = []

        # Number of Nodes
        parallel_run_args += [f'--nrs={self.nodes}']
        tmp = f'-nnodes {self.nodes}'
        cmd_args += [tmp]
        if not blocking:
            header.write(f'#BSUB -nnodes {self.nodes}\n')

        cmd_args += ['--shared-launch']

        # jsrun options
        parallel_run_args += ['--rs_per_host 1']
        parallel_run_args += [f'--tasks_per_rs={self.procs_per_node}']
        parallel_run_args += ['--launch_distribution packed']
        parallel_run_args += ['--cpu_per_rs ALL_CPUS']
        parallel_run_args += ['--gpu_per_rs ALL_GPUS']

        if self.out_log_file and not blocking:
            header.write(f'#BSUB -o {self.out_log_file}\n')
        if self.err_log_file and not blocking:
            header.write(f'#BSUB -e {self.err_log_file}\n')

        # Configure header with LSF job options
        if self.time_limit:
            minutes = int(round(max(self.time_limit, 0)))
            hours, minutes = divmod(minutes, 60)
            header.write(f'#BSUB -W {hours}:{minutes:02}\n')
        if self.job_name:
            tmp = f'-J {self.job_name}'
            select_interactive_or_batch(tmp, header, cmd_args, blocking)
        if self.queue:
            tmp = f'-q {self.queue}'
            select_interactive_or_batch(tmp, header, cmd_args, blocking)
        if self.account:
            tmp = f'-G {self.account}'
            select_interactive_or_batch(tmp, header, cmd_args, blocking)
        if self.reservation:
            header.write(f'#BSUB -U {self.reservation}\n')

        if self.work_dir:
            cmd_args += [f'--chdir={self.work_dir}']
            header.write(f'#BSUB -cwd {self.work_dir}\n')

        if self.launcher_flags:
            for flag in self.launcher_flags:
                cmd_args += [f'{flag}']

        for k, v in env_vars:
            header.write(f'export {k}={v}\n')

        for k, v in passthrough_env_vars:
            if not blocking:
                cmd_args += [f' --env={k}={v}']
            else:
                header += f'export {k}={v}\n'

        return (header.getvalue(), cmd_args, parallel_run_args)

    def launch_command(self, system: 'System', blocking: bool = True) -> list[str]:
        # Launch command only use the cmd_args to construct the shell script to be launched
        (header_lines, cmd_args, parallel_run_args) = self.build_command_string_and_batch_script(system, blocking)

        if not blocking:
            return ['bsub'] + cmd_args

        return ['bsub', '-Is'] + cmd_args

    def launcher_script(self,
                        system: 'System',
                        command: str,
                        args: Optional[list[str]] = None,
                        blocking: bool = True) -> str:

        script = ''
        # Launcher script only use the header_lines to construct the shell script to be launched
        (header_lines, cmd_string, parallel_run_args) = self.build_command_string_and_batch_script(system, blocking)
        script += header_lines
        script += '\n'

        # system = autodetect.autodetect_current_system()
        # env_vars = system.environment_variables()
        # passthrough_env_vars = system.passthrough_environment_variables()
        # # Enable the system to apply some customization to the scheduler instance
        # system.customize_scheduler(self)

        # script = ''
        # header_lines = ''
        # cmd_string = ''
        # if self.out_log_file:
        #     header_lines += f'#BSUB -o {self.out_log_file}\n'
        # if self.err_log_file:
        #     header_lines += f'#BSUB -e {self.err_log_file}\n'

#BVE
        # if self.launcher_flags:
        #     cmd_string += f' {" ".join(self.launcher_flags)}'

        # cmd_string += f' --nrs={self.nodes}'
        # header_lines += f'#BSUB -nnodes {self.nodes}\n'

        # cmd_string += ' --rs_per_host 1'
        # cmd_string += f' --tasks_per_rs={self.procs_per_node}'
        # cmd_string += ' --launch_distribution packed'
        # cmd_string += ' --cpu_per_rs ALL_CPUS'
        # cmd_string += ' --gpu_per_rs ALL_GPUS'

        # cmd_string += ' -o nosetpgrp'

        # if self.work_dir:
        #     cmd_string += f'--chdir={self.work_dir}'
        #     header_lines += f'#BSUB -cwd {self.work_dir}\n'

        # if self.ld_preloads:
        #     cmd_string += f'--export=ALL,LD_PRELOAD={",".join(self.ld_preloads)}'

        # for k,v in passthrough_env_vars:
        #     cmd_string += f' --env={k}={v}'

        # if self.time_limit:
        #     minutes = int(round(max(self.time_limit, 0)))
        #     hours, minutes = divmod(minutes, 60)
        #     header_lines += f'#BSUB -W {hours}:{minutes:02}\n'
        # if self.job_name:
        #     header_lines += f'#BSUB -J {self.job_name}\n'
        # if self.partition:
        #     header_lines += f'#BSUB -q {self.partition}\n'
        # if self.account:
        #     header_lines += f'#BSUB -G {self.account}\n'
        # if self.reservation:
        #     header_lines += f'#BSUB -U {self.reservation}\n'

        # # Configure header and command line with Slurm job options
        # script += header_lines
        # for k,v in env_vars:
        #     script += f'export {k}={v}\n'

        # script += self.launch_command(True)
        # script += cmd_string
        # script += f' {command}'

        # #jsrun

        # for arg in args:
        #     script += f' {arg}'

        if not blocking or blocking:
            script += 'jsrun '
            script += ' '.join(parallel_run_args)
            script += ' '

        script += f'{command}'

        for arg in args:
            script += f' {arg}'

        script += '\n'

        return script
