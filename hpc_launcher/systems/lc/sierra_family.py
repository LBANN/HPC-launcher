from hpc_launcher.schedulers.scheduler import Scheduler
from hpc_launcher.schedulers.lsf import LSFScheduler
from hpc_launcher.systems.system import System, SystemParams
import os


# Supported LC systems
_sierra_node = SystemParams(16, 4, "sm_70", 16.0, 2, "lsf")
_system_params = {
    "lassen": (
        "pbatch",
        {
            "pdebug": _sierra_node,
            "pbatch": _sierra_node,
            "standby": _sierra_node,
        },
    ),
    "rzansel": (
        "pbatch",
        {
            "pdebug": _sierra_node,
            "pbatch": _sierra_node,
        },
    ),
    "sierra": (
        "pbatch",
        {
            "pdebug": _sierra_node,
            "pbatch": _sierra_node,
        },
    ),
}


class Sierra(System):
    """
    LLNL LC Systems based on the Sierra Power9 + Nvidia V100 architecture.
    """

    def __init__(self, system_name):
        super().__init__(system_name, _system_params)

    def environment_variables(self) -> list[tuple[str, str]]:

        env_list = []

        # Hack to enable process forking
        # Note: InfiniBand is known to experience hangs if an MPI
        # process is forked (see
        # https://www.open-mpi.org/faq/?category=openfabrics#ofa-fork).
        # Setting IBV_FORK_SAFE seems to fix this issue, but it may
        # hurt performance (see
        # https://linux.die.net/man/3/ibv_fork_init).
        env_list.append(("IBV_FORK_SAFE", 1))
        # Hacked bugfix for hcoll (1/23/19)
        # Note: Fixes hangs in MPI_Bcast.
        env_list.append(("HCOLL_ENABLE_SHARP", 0))
        env_list.append(("OMPI_MCA_coll_hcoll_enable", 0))

        # Hacked bugfix for Spectrum MPI PAMI (9/17/19)
        env_list.append(("PAMI_MAX_NUM_CACHED_PAGES", 0))

        # Configure NVSHMEM to load Spectrum MPI
        env_list.append(("NVSHMEM_MPI_LIB_NAME", "libmpi_ibm.so"))

        for i in self._aux_env_list:
            env_list.append(i)

        return env_list

    def customize_scheduler(self, scheduler):
        # Note: There are actually 22 cores/socket, but it seems that
        # powers of 2 are better for performance.
        cores_per_socket = 16
        procs_per_node = 2
        procs_per_socket = (procs_per_node + 1) // 2
        cores_per_proc = cores_per_socket // procs_per_socket
        if isinstance(scheduler, LSFScheduler):
            scheduler.run_only_args["--bind"] = "packed:{}".format(cores_per_proc)
            scheduler.run_only_args["--smpiargs"] = '"-gpu"'
        return

    @property
    def preferred_scheduler(self) -> type[Scheduler]:
        return LSFScheduler
