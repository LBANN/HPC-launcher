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
    from hpc_launcher.systems import System

from hpc_launcher.schedulers.scheduler import Scheduler

import logging

logger = logging.getLogger(__name__)


@dataclass
class FluxScheduler(Scheduler):

    def select_interactive_or_batch(self,
                                    tmp: list[str],
                                    header: StringIO,
                                    cmd_args: list[str],
                                    blocking: bool = True) -> None:
        if blocking:
            cmd_args += tmp
        else:
            header.write(f'# FLUX: {" ".join(tmp)}\n')
        return

    def build_command_string_and_batch_script(self,
                                              system: 'System',
                                              blocking: bool = True
                                              ) -> (str, list[str]):

        env_vars = system.environment_variables()
        passthrough_env_vars = system.passthrough_environment_variables()
        # Enable the system to apply some customization to the scheduler instance
        system.customize_scheduler(self)

        header = StringIO()
        header.write('#!/bin/sh\n')
        cmd_args = []
        if self.out_log_file and not blocking:
            header.write(f'# FLUX: --output={self.out_log_file}\n')
        if self.err_log_file and not blocking:
            header.write(f'# FLUX: --error={self.err_log_file}\n')

        # Unbuffered output
        tmp = '-u'
        cmd_args += [tmp]
        if not blocking:
            header.write(f'# FLUX: {tmp}\n')

        # Number of Nodes
        tmp = f'-N{self.nodes}'
        cmd_args += [tmp]
        if not blocking:
            header.write(f'# FLUX: {tmp}\n')

        # Total number of Tasks / Processes
        tmp = f'-n{self.nodes * self.procs_per_node}'
        cmd_args += [tmp]
        if not blocking:
            header.write(f'# FLUX: {tmp}\n')

        if self.work_dir:
            tmp = [f'--setattr=system.cwd={os.path.abspath(self.work_dir)}']
            self.select_interactive_or_batch(tmp, header, cmd_args, blocking)

        tmp = ['-onosetpgrp']
        self.select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.ld_preloads:
            tmp = [f'--env=LD_PRELOAD={",".join(self.ld_preloads)}']
            self.select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.time_limit is not None:
            tmp = [f'--time={self.time_limit}m']
            self.select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.job_name:
            tmp = [f'--job-name={self.job_name}']
            self.select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.queue:
            tmp = [f'--queue={self.queue}']
            self.select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.account:
            tmp = [f'--account={self.account}']
            self.select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.reservation:
            logger.warning(
                f'WARNING: Unsupported option requested: --reservation={self.reservation}'
            )

        if self.launcher_flags:
            for flag in self.launcher_flags:
                # These flag should only be on the launcher commands not the batch commands
                cmd_args += [flag]

        for k, v in env_vars:
            header.write(f'export {k}={v}\n')

        for k, v in passthrough_env_vars:
            if not blocking:
                cmd_args += [f' --env={k}={v}']
            else:
                header += f'export {k}={v}\n'

        return (header.getvalue(), cmd_args)

    def launch_command(self,
                       system: 'System',
                       blocking: bool = True) -> list[str]:
        # Launch command only use the cmd_args to construct the shell script to be launched
        (header_lines, cmd_args) = self.build_command_string_and_batch_script(
            system, blocking)

        if not blocking:
            return ['flux', 'batch'] + cmd_args

        return ['flux', 'run'] + cmd_args

    def launcher_script(self,
                        system: 'System',
                        command: str,
                        args: Optional[list[str]] = None,
                        blocking: bool = True) -> str:

        script = ''
        # Launcher script only use the header_lines to construct the shell script to be launched
        (header_lines,
         cmd_string) = self.build_command_string_and_batch_script(
             system, blocking)
        script += header_lines
        script += '\n'
        script += 'export HPC_LAUNCHER_HOSTLIST=$(flux hostlist local)\n'

        if not blocking:
            script += 'flux run '
            script += ' '.join(cmd_string)
            script += ' '

        script += f'{command}'

        for arg in args:
            script += f' {arg}'

        script += '\n'

        return script

    def get_job_id(self, output: str) -> Optional[str]:
        # The job ID is the only printout when calling flux batch
        return output.strip()

    @classmethod
    def get_parallel_configuration(cls) -> tuple[int, int, int, int]:
        env_vars = [
            'FLUX_JOB_SIZE', 'FLUX_TASK_RANK', 'FLUX_TASK_LOCAL_ID',
            'FLUX_JOB_NNODES'
        ]
        env = {}
        for e in env_vars:
            if not os.getenv(e):
                msg = f'Unable to launch torchrun_hpc on FLUX scheduler - {e} not defined'
                raise Exception(msg)
            else:
                env[e] = int(os.getenv(e))

        world_size = env['FLUX_JOB_SIZE']
        rank = env['FLUX_TASK_RANK']
        local_rank = env['FLUX_TASK_LOCAL_ID']
        nodes_per_job = env['FLUX_JOB_NNODES']
        local_world_size = world_size // nodes_per_job
        return (world_size, rank, local_world_size, local_rank)

    def dynamically_configure_rendezvous_protocol(self, protocol: str) -> list[str]:
        env_list = []
        if protocol.lower() == 'tcp':
            env_list.append(('TORCHRUN_HPC_MASTER_ADDR', '`flux hostlist local | /bin/hostlist -n 1`'))
            env_list.append(('TORCHRUN_HPC_MASTER_PORT', '23456'))
            return env_list
        elif protocol.lower() == 'mpi':
            # To use MPI, pass `init_method="mpi://"` - no special work here.
            return env_list
        else:
            msg = f'Unsupported rendezvous protocol {protocol} for scheduler {type(self).__name__}'
            raise Exception(msg)
