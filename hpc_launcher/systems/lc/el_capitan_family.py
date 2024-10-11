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

        env_list = []
        env_list.append(('NCCL_NET_GDR_LEVEL', '3')) # From HPE to avoid hangs
        env_list.append(('MIOPEN_DEBUG_DISABLE_FIND_DB', '0'))
        env_list.append(('MIOPEN_DISABLE_CACHE', '0'))
        tmpdir = os.environ.get('TMPDIR')
        env_list.append(('MIOPEN_USER_DB_PATH', f'{tmpdir}/MIOpen_user_db'))
        env_list.append(('MIOPEN_CUSTOM_CACHE_DIR', f'{tmpdir}/MIOpen_custom_cache'))

        if os.getenv('CRAY_LD_LIBRARY_PATH') is not None:
            env_list.append(('LD_LIBRARY_PATH', os.getenv('CRAY_LD_LIBRARY_PATH') + ':${LD_LIBRARY_PATH}'))
        if os.getenv('ROCM_PATH') is not None:
            env_list.append(('LD_LIBRARY_PATH', os.path.join(os.getenv('ROCM_PATH'), 'llvm', 'lib') + ':${LD_LIBRARY_PATH}'))

        different_ofi_plugin = os.getenv('LBANN_USE_THIS_OFI_PLUGIN')
        if different_ofi_plugin is not None:
            env_list.append(('LD_LIBRARY_PATH', different_ofi_plugin + ':${LD_LIBRARY_PATH}'))

        env_list.append(('MPICH_OFI_NIC_POLICY', 'GPU'))
        env_list.append(('OMP_NUM_THREADS', '21'))
        env_list.append(('OMP_PLACES', 'threads'))
        env_list.append(('OMP_PROC_BIND', 'spread'))

        return env_list

    def customize_scheduler(self, Scheduler):
        use_this_rccl=os.getenv('LBANN_USE_THIS_RCCL')
        Scheduler.launcher_flags = ['--exclusive']
        if use_this_rccl is not None:
            Scheduler.ld_preloads = [f'{use_this_rccl}']
        return

    @property
    def preferred_scheduler(self) -> type[Scheduler]:
        return FluxScheduler
    
