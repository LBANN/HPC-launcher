from hpc_launcher.schedulers.scheduler import Scheduler
import warnings

# ==============================================
# Set system parameters
# ==============================================

class SystemParams:
    """Simple data structure to describe an LC system."""
    def __init__(self, cores_per_node, gpus_per_node, gpu_arch, mem_per_gpu, cpus_per_node, numa_domains, scheduler):
        self.cores_per_node = cores_per_node
        self.gpus_per_node = gpus_per_node
        self.gpu_arch = gpu_arch
        self.mem_per_gpu = mem_per_gpu # GB
        self.cpus_per_node = cpus_per_node
        self.scheduler = scheduler
        self.numa_domains = numa_domains

    def print_params(self):
        print(f'c={self.cores_per_node} g={self.gpus_per_node} s={self.scheduler} arch={self.gpu_arch} numa={self.numa_domains}')

    def has_gpu(self):
        """Whether LC system has GPUs."""
        return self.gpus_per_node > 0

    def procs_per_node(self):
        """Default number of processes per node."""
        if self.has_gpu():
            return self.gpus_per_node
        else:
            # Assign one rank / process to each NUMA domain to play nice with OPENMP
            return self.numa_domains


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

class System:
    """
    Represents a system (specific supercomputer, HPC cluster, or
    cloud service provider) that can be used to launch distributed
    jobs.
    """
    def __init__(self, system_name, known_systems = None):
        self.system_name = system_name
        self.default_queue = None
        self.system_params = None
        self.known_systems = known_systems
#        print(f'BVE initialie System with {system_name}')
        if self.known_systems:
            if system_name in self.known_systems.keys():
                (self.default_queue, self.system_params) = self.known_systems[system_name]
#                print(f'BVE the default queue is {self.default_queue} and the paras {self.system_params}')
            else:
                warnings.warn('Could not auto-detect current system parameters')
        else:
            warnings.warn('No list of known systems')

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

    def customize_scheduler(self, Scheduler):
        """
        Add any system specific customizations to the scheduler.
        """
        return

    def system_parameters(self, requested_queue = None) -> SystemParams:
        queue = self.default_queue
        if requested_queue:
            queue = requested_queue
        if self.system_params:
            if queue not in self.system_params:
                warnings.warn(f'Unknown queue {queue} on system {self.system_name} using system parameters from default queue {self.default_queue}')
                params = self.system_params[self.default_queue]
            else:
                params = self.system_params[queue]
            return params
        else:
            return None


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
