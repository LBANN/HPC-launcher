from hpc_launcher.schedulers.scheduler import Scheduler
from hpc_launcher.schedulers.flux import FluxScheduler
from hpc_launcher.systems.system import System

class ElCapitan(System):
    """
    LLNL LC Systems based on the El Capitan MI300a architecture.
    """

    def environment_variables(self) -> list[tuple[str, str]]:
        return [('foo', 1), ('bar', 2)]

    @property
    def preferred_scheduler(self) -> type[Scheduler]:
        return FluxScheduler
    
