from hpc_launcher.schedulers.scheduler import Scheduler

class System:
    """
    Represents a system (specific supercomputer, HPC cluster, or
    cloud service provider) that can be used to launch distributed
    jobs.
    """

    def environment_variables(self) -> list[tuple[str, str]]:
        """
        Returns a list of environment variables that configures the
        system for efficient use.

        :return: A list of (environment variable name, value) tuples.
        """
        raise NotImplementedError

    def passthrough_environment_variables(self) -> list[tuple[str, str]]:
        """
        Returns a list of environment variables that are passed through
        the scheduler to the command.

        :return: A list of (environment variable name, value) tuples.
        """
        return []

    @property
    def preferred_scheduler(self) -> type[Scheduler]:
        """
        Returns the preferred batch scheduler on the system.

        :return: A class type representing the first choice of batch
                 scheduler used in this system.
        """
        raise NotImplementedError


class GenericSystem(System):
    """
    A generic System type that does not specify any particular behavior.
    """

    def environment_variables(self) -> list[tuple[str, str]]:
        return []

    @property
    def preferred_scheduler(self) -> type[Scheduler]:
        raise NotImplementedError  # TODO: Use SLURM?


# ==============================================
# Set system parameters
# ==============================================

class SystemParams:
    """Simple data structure to describe an LC system."""
    def __init__(self, cores_per_node, gpus_per_node, gpu_arch, numa_domains, scheduler):
        self.cores_per_node = cores_per_node
        self.gpus_per_node = gpus_per_node
        self.scheduler = scheduler
        self.gpu_arch = gpu_arch
        self.numa_domains = numa_domains

    def print_params():
        print(f'c={self.cores_per_node} g={self.gpus_per_node} s={self.scheduler} arch={self.gpu_arch} numa={self.numa_domains}')

# Supported LC systems
# _system_params = {
#     'corona':   SystemParams(48, 8, 'flux'),
#     'lassen':   SystemParams(44, 4, 'lsf'),
#     'pascal':   SystemParams(36, 2, 'slurm'),
#     'rzansel':  SystemParams(44, 4, 'lsf'),
#     'rzvernal': SystemParams(64, 8, 'flux'),
#     'sierra':   SystemParams(44, 4, 'lsf'),
#     'tioga':    SystemParams(64, 8, 'flux'),
# }

    

# def is_lc_system(system = system()):
#     """Whether current system is a supported LC system."""
#     return _system in _system_params.keys()

# def gpus_per_node(system = system()):
#     """Number of GPUs per node."""
#     if not is_lc_system(system):
#         raise RuntimeError('unknown system (' + system + ')')
#     return _system_params[system].gpus_per_node

# def has_gpu(system = system()):
#     """Whether LC system has GPUs."""
#     return gpus_per_node(system) > 0

# def cores_per_node(system = system()):
#     """Number of CPU cores per node."""
#     if not is_lc_system(system):
#         raise RuntimeError('unknown system (' + system + ')')
#     return _system_params[system].cores_per_node

# def scheduler(system = system()):
#     """Job scheduler for LC system."""
#     if not is_lc_system(system):
#         raise RuntimeError('unknown system (' + system + ')')
#     return _system_params[system].scheduler

# def procs_per_node(system = system()):
#     """Default number of processes per node."""
#     if has_gpu(system):
#         return gpus_per_node(system)
#     else:
#         # Catalyst and Quartz have 2 sockets per node
#         ### @todo Think of a smarter heuristic
#         return 2
