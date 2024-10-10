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
                        args: Optional[list[str]] = None) -> str:
        # String IO

        system = autodetect.autodetect_current_system()
        env_vars = system.environment_variables()
        passthrough_env_vars = system.passthrough_environment_variables()
        # Enable the system to apply some customization to the scheduler instance
        system.customize_scheduler(self)

        script = ''
        for k,v in env_vars:
            script += f'export {k}={v}\n'

        script += self.launch_command(True)
        if self.launcher_flags:
            script += f' {" ".join(self.launcher_flags)}'

        script += ' -u'  # Unbuffered
        script += f' -N{self.nodes}' # --nodes
        script += f' -n{self.nodes * self.procs_per_node}' # --ntasks

        if self.work_dir:
            script += f' --setattr=system.cwd={self.work_dir}'

        script += ' -o nosetpgrp'

        if self.ld_preloads:
            script += f' --env=LD_PRELOAD={",".join(self.ld_preloads)}'

        for k,v in passthrough_env_vars:
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
