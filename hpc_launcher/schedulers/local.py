from hpc_launcher.schedulers.scheduler import Scheduler
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import os
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # If type-checking, import the other class
    from hpc_launcher.systems.system import System


@dataclass
class LocalScheduler(Scheduler):
    """
    A class that runs the job without any underlying batch scheduler. Used
    in ``--local`` jobs.
    """

    def launch_command(self,
                       system: 'System',
                       blocking: bool = True) -> list[str]:
        return []

    def launcher_script(self,
                        system: 'System',
                        command: str,
                        args: Optional[list[str]] = None,
                        blocking: bool = True) -> str:
        envvars = [
            f'export {k}={v}' for k, v in system.environment_variables()
        ]
        envvars += [
            f'export {k}={v}'
            for k, v in system.passthrough_environment_variables()
        ]
        envvars += [
            'export RANK=0',
            'export HPC_LAUNCHER_HOSTLIST=$(hostname)',
        ]
        header = '\n'.join(envvars)

        if self.work_dir:
            header += f'\ncd {os.path.abspath(self.work_dir)}\n'

        return f'''#!/bin/sh
# Setup
{header}

# Run
{command} {" ".join(args)}
'''

    def get_job_id(self, output: str) -> Optional[str]:
        return None

    @classmethod
    def get_parallel_configuration(cls) -> tuple[int, int, int, int]:
        return 1, 0, 1, 0

    def dynamically_configure_rendezvous_protocol(self, protocol: str) -> list[str]:
        env_list = []
        if protocol.lower() == 'tcp':
            env_list.append(('TORCHRUN_HPC_MASTER_ADDR', 'localhost'))
            env_list.append(('TORCHRUN_HPC_MASTER_PORT', '23456'))
            return env_list
        else:
            msg = f'Unsupported rendezvous protocol {protocol} for scheduler {type(self).__name__}'
            raise Exception(msg)
