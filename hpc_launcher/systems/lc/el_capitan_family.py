from hpc_launcher.schedulers.scheduler import Scheduler
from hpc_launcher.schedulers.flux import FluxScheduler
from hpc_launcher.systems.system import System, SystemParams
#from hpc_launcher.systems.system import SystemParams
import os


# Supported LC systems
_system_params = SystemParams(64, 8, 'gfx90a,gfx942', 4, 'flux')

# _system_params = {
#     'tioga':    SystemParams(64, 8, 'gfx90a,gfx942', 1, 'flux'),
# }

class ElCapitan(System):
    """
    LLNL LC Systems based on the El Capitan MI300a architecture.
    """

    def environment_variables(self) -> list[tuple[str, str]]:
#flux run --exclusive -N2 -n8 -c21 -g1 ...        
        return [('MPICH_OFI_NIC_POLICY', 'GPU'),
                ('OMP_NUM_THREADS', '21'),
                ('OMP_PLACES', 'threads'),
                ('OMP_PROC_BIND', 'spread'),
        ]


    def customize_scheduler(self, Scheduler):
        use_this_rccl=os.getenv('LBANN_USE_THIS_RCCL')
        Scheduler.launcher_flags = ['--exclusive']
        if use_this_rccl is not None:
            Scheduler.ld_preloads = [f'{use_this_rccl}']
        return

    @property
    def preferred_scheduler(self) -> type[Scheduler]:
        return FluxScheduler
    
