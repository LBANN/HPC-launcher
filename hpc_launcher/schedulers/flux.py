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
        return 'flux run' if blocking else 'flux batch'

    def launcher_script(self, system: 'System', command: str,
                        args: Optional[list[str]] = None,
                        blocking: bool = True) -> str:
        # String IO

        system = autodetect.autodetect_current_system()
        env_vars = system.environment_variables()

        script = ''
        for k,v in env_vars:
            script += f'export {k}={v}\n'

        script += self.launch_command(blocking)
        script += ' --exclusive'
        script += ' -u'  # Unbuffered
        script += f' -N{self.nodes}' # --nodes
        script += f' -n{self.nodes * self.procs_per_node}' # --ntasks

        if self.work_dir:
            script += f' --setattr=system.cwd={self.work_dir}'

        script += ' -o nosetpgrp'

        script += f' {command}'

        for arg in args:
            script += f' {arg}'

        return script
