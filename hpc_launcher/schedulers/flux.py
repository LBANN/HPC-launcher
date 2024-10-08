from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # If type-checking, import the other class
    from hpc_launcher.systems import System

from hpc_launcher.schedulers.scheduler import Scheduler

@dataclass
class FluxScheduler(Scheduler):
    def launch_command(self, blocking: bool = True) -> list[str]:
        return 'flux run' if blocking else 'flux batch'

    def launcher_script(self, system: 'System') -> str:
        raise NotImplementedError

    def internal_script(self, system: 'System') -> Optional[str]:
        return None

    def launch(self, system: 'System', program: str,
               args: Optional[list[str]] = None,
               blocking: bool = True,
               verbose: bool = False) -> str:
        raise NotImplementedError
