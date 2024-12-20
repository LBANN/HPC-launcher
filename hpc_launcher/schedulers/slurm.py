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
import subprocess

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


@dataclass
class SlurmScheduler(Scheduler):

    def select_interactive_or_batch(self,
                                    tmp: list[str],
                                    header: StringIO,
                                    cmd_args: list[str],
                                    blocking: bool = True) -> None:
        if blocking:
            cmd_args += tmp
        else:
            header.write(f'#SBATCH {" ".join(tmp)}\n')
        return

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
            cmd_args += ['-u']

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
            tmp = [f'--chdir={os.path.abspath(self.work_dir)}']
            self.select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.ld_preloads:
            tmp = [f'--export=ALL,LD_PRELOAD={",".join(self.ld_preloads)}']
            self.select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.time_limit is not None:
            tmp = [f'--time={_time_string(self.time_limit)}']
            self.select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.job_name:
            tmp = [f'--job-name={self.job_name}']
            self.select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.queue:
            tmp = [f'--partition={self.queue}']
            self.select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.account:
            tmp = [f'--account={self.account}']
            self.select_interactive_or_batch(tmp, header, cmd_args, blocking)

        if self.reservation:
            tmp = [f'--reservation={self.reservation}']
            self.select_interactive_or_batch(tmp, header, cmd_args, blocking)

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

    @classmethod
    def get_parallel_configuration(cls) -> tuple[int, int, int, int]:
        # Interesting but unused variables SLURM_JOB_NUM_NODES, SLURM_NPROCS, SLURM_DISTRIBUTION
        # Skipping 'SLURM_TASKS_PER_NODE' because this field has a weird format e.g. 2(x2)
        env_vars = ['SLURM_NTASKS', 'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_NNODES']
        env = {}
        for e in env_vars:
            if not os.getenv(e):
                msg = f'Unable to launch torchrun_hpc on SLURM scheduler - {e} not defined'
                raise Exception(msg)
            else:
                env[e] = int(os.getenv(e))

        world_size = env['SLURM_NTASKS']
        rank = env['SLURM_PROCID']
        local_rank = env['SLURM_LOCALID']
        nodes_per_job = env['SLURM_NNODES']
        local_world_size = world_size // nodes_per_job
        # local_world_size = env['SLURM_TASKS_PER_NODE']
        return (world_size, rank, local_world_size, local_rank)

    @classmethod
    def dynamically_configure_rendezvous_protocol(cls, protocol: str) -> str:
        if protocol.lower() == 'tcp':
            command = 'printenv SLURM_JOB_NODELIST | /bin/hostlist -n 1'
            master_addr = subprocess.check_output(command, shell=True, text=True).rstrip()
            master_port = '23456'
            return f'tcp://{master_addr}:{master_port}'
        else:
            msg = f'Unsupported rendezvous protocol {protocol}'
            raise Exception(msg)

    def setup_rendezvous_protocol(self, protocol: str) -> list[str]:
        env_list = []
        env_list.append(('TORCHRUN_HPC_SCHEDULER', type(self).__name__))
        env_list.append(('TORCHRUN_HPC_RDV_PROTOCOL', 'TCP'))
        return env_list
