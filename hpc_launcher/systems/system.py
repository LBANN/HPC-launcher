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

    def preferred_scheduler(self) -> type[Scheduler]:
        raise NotImplementedError  # TODO: Use SLURM?



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
