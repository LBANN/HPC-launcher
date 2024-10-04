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
