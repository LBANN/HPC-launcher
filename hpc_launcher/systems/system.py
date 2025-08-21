# Copyright (c) 2014-2025, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. See the top-level LICENSE file for details.
#
# LLNL-CODE-697807.
# All rights reserved.
#
# This file is part of LBANN: Livermore Big Artificial Neural Network
# Toolkit. For details, see http://software.llnl.gov/LBANN or
# https://github.com/LBANN and https://github.com/LLNL/LBANN.
#
# SPDX-License-Identifier: (Apache-2.0)
from dataclasses import dataclass, asdict
import json
from typing import Optional
import logging
from hpc_launcher.schedulers.scheduler import Scheduler

logger = logging.getLogger(__name__)

# ==============================================
# Set system parameters
# ==============================================


@dataclass
class SystemParams:
    """Simple data structure to describe an LC system."""

    # Number of CPU cores per compute node
    cores_per_node: int = 0
    # Number of GPUs per node
    gpus_per_node: int = 0
    # Vendor specific GPU compiler architecture
    gpu_arch: str = None
    # Number of GB of memory per GPU
    mem_per_gpu: float = 0.0
    # Number of NUMA domains
    numa_domains: int = 0
    # String name of the Scheduler class
    scheduler: str = None
    # Optional system level guard to limit GPU/APU memory utilization
    fraction_max_gpu_mem: Optional[float] = 1.0

    def __repr__(self):
        return json.dumps(asdict(self))

    def __str__(self):
        return json.dumps(asdict(self))

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

    def prettyprint(self):
        if self.fraction_max_gpu_mem != 1.0:
            effective_gpu_mem = self.fraction_max_gpu_mem * self.mem_per_gpu
            max_gpu_mem = f" ({self.mem_per_gpu} GB max)"
        else:
            effective_gpu_mem = self.mem_per_gpu
            max_gpu_mem = ""

        return f"{self.scheduler} scheduled system with {self.cores_per_node} cores in {self.numa_domains} NUMA domains with {self.gpus_per_node} x {effective_gpu_mem}GB {self.gpu_arch} GPUs{max_gpu_mem}"

class System:
    """
    Represents a system (specific supercomputer, HPC cluster, or
    cloud service provider) that can be used to launch distributed
    jobs.
    """

    def __init__(self, system_name, known_systems=None):
        self.system_name = system_name
        self.default_queue = None
        self.system_params = None
        self.active_system_params = None
        self.known_systems = known_systems
        if self.known_systems:
            if system_name in self.known_systems.keys():
                (self.default_queue, self.system_params) = self.known_systems[
                    system_name
                ]
            else:
                logger.warning("Could not auto-detect current system parameters")
        # Optional system level flag to indicate primary communication method for distributed jobs
        self.job_comm_protocol = None

    _aux_env_list = []

    def extend_environment_variables(self, env_list: list[tuple[str, str]]):
        """
        Enable a CLI to extend the environment variables list defined by the system
        """
        for i in env_list:
            self._aux_env_list.append(i)

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

    def customize_scheduler(self, scheduler: Scheduler):
        """
        Add any system specific customizations to the scheduler.
        """
        return

    def system_parameters(self, requested_queue=None) -> SystemParams:
        queue = self.default_queue
        if requested_queue:
            queue = requested_queue
        if self.system_params:
            if queue not in self.system_params:
                logger.warning(
                    f"Unknown queue {queue} on system {self.system_name} using system parameters from default queue {self.default_queue}"
                )
                params = self.system_params[self.default_queue]
            else:
                params = self.system_params[queue]
            self.active_system_params = params
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, system_name="generic", **kwargs)

    def environment_variables(self) -> list[tuple[str, str]]:
        env_list = []
        for i in self._aux_env_list:
            env_list.append(i)

        return env_list

    @property
    def preferred_scheduler(self) -> type[Scheduler]:
        # SLURM is a relatively safe bet for a scheduler
        from hpc_launcher.schedulers.slurm import SlurmScheduler

        return SlurmScheduler
