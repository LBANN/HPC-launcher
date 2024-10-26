from hpc_launcher.schedulers.scheduler import Scheduler
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

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

    def launch_command(self, blocking: bool = True) -> list[str]:
        return []

    def launcher_script(self,
                        system: 'System',
                        command: str,
                        args: Optional[list[str]] = None) -> str:
        envvars = [
            f'export {k}={v}' for k, v in system.environment_variables()
        ]
        envvars += [
            f'export {k}={v}'
            for k, v in system.passthrough_environment_variables()
        ]
        envvars = '\n'.join(envvars)
        return f'''#!/bin/sh
# Setup
{envvars}

# Run
{command} {" ".join(args)}
'''

    def get_job_id(self, output: str) -> Optional[str]:
        return None
