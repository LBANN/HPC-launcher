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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
from io import StringIO
import os
import subprocess
import re

if TYPE_CHECKING:
    # If type-checking, import the other class
    from hpc_launcher.systems import System

from hpc_launcher.schedulers.scheduler import Scheduler
from hpc_launcher.schedulers import parse_env_list

import logging

logger = logging.getLogger(__name__)


@dataclass
class FluxScheduler(Scheduler):

    def build_scheduler_specific_arguments(
        self, system: "System", blocking: bool = True
    ):
        if self.out_log_file and not blocking:
            self.submit_only_args[f"--output"] = f"{self.out_log_file}"
        if self.err_log_file and not blocking:
            self.submit_only_args[f"--error"] = f"{self.err_log_file}"

        # Number of Nodes
        self.common_launch_args[f"-N{self.nodes}"] = None

        # Total number of Tasks / Processes
        self.common_launch_args[f"-n{self.nodes * self.procs_per_node}"] = None

        # Unbuffered output
        self.common_launch_args["-u"] = None

        # Set the Number of GPUs per task
        # There is a difference in option names between tasks and allocations
        if self.gpus_per_proc > 0:
            tmp = f"{self.gpus_per_proc}"
            # command line flag for a task
            self.run_only_args["--gpus-per-task"] = tmp
            # command and shell flags for an allocation
            if not blocking:
                self.submit_only_args["--gpus-per-slot"] = tmp

        # Request for node exclusivity
        if self.exclusive:
            self.submit_only_args["--exclusive"] = ""

        if self.work_dir:
            self.submit_only_args["--setattr=system.cwd"] = f"{os.path.abspath(self.work_dir)}"

        self.common_launch_args["-onosetpgrp"] = None

        if self.ld_preloads:
            self.common_launch_args['--env=LD_PRELOAD'] = f'{",".join(self.ld_preloads)}'

        if self.time_limit is not None:
            if blocking:
                self.common_launch_args["--time"] = f"{self.time_limit}m"
            else:
                self.submit_only_args["--time"] = f"{self.time_limit}m"

        if self.dependency is not None:
            self.common_launch_args["--dependency"] = f"{self.dependency}"
        dependency = self.common_launch_args.get('--dependency', None)
        if self.override_launch_args and self.override_launch_args.get('--dependency', None):
            dependency = self.override_launch_args['--dependency']
        if dependency and not blocking:
            try:
                del self.common_launch_args['--dependency']
            except KeyError:
                pass
            try:
                if self.override_launch_args:
                    del self.override_launch_args['--dependency']
            except KeyError:
                pass
            self.submit_only_args["--dependency"] = dependency

        if self.job_name:
            self.common_launch_args["--job-name"] = f"{self.job_name}"

        if self.queue:
            if os.getenv("FLUX_URI"):
                logger.warning(
                    f"WARNING: Dropping unsupported option requested when running inside of an allocation: --queue={self.queue}"
                )
            else:
                self.submit_only_args["--queue"] = f"{self.queue}"

        if self.account:
            self.submit_only_args["--bank"] = f"{self.account}"

        if self.reservation:
            logger.warning(
                f"WARNING: Unsupported option requested: --reservation={self.reservation}"
            )

        return

    def batch_script_prefix(self) -> str:
        return "# FLUX:"

    def blocking_launch_command(self) -> list[str]:
        return ["flux", "run"]

    def nonblocking_launch_command(self) -> list[str]:
        return ["flux", "batch"]

    def cli_env_arg(self, env_list) -> None:
        for e in env_list:
            if len(e) == 1:
                continue
            elif len(e) == 2:
                k,v = e
                self.submit_only_args[f"--env={k}"] = f"{v}"
            elif len(e) == 3:
                k,v,m = e
                self.submit_only_args[f"--env={k}"] = f"{v}"
        return

    def export_hostlist(self) -> str:
        return "export HPC_LAUNCHER_HOSTLIST=$(flux hostlist local)\n"

    def internal_script_run_command(self) -> str:
        return "flux run "

    def get_job_id(self, output: str) -> Optional[str]:
        # The job ID is the only printout when calling flux batch
        return output.strip()

    @classmethod
    def num_nodes_in_allocation(cls) -> Optional[int]:
        if os.getenv("FLUX_URI"):
            cmd = ["flux", "resource", "info"]
            proc = subprocess.run(cmd, universal_newlines=True, capture_output=True)
            m = re.search(r"^(\d*) Nodes, (\d*) Cores, (\d*) GPUs$", proc.stdout)
            if m:
                return int(m.group(1))

        return None

    @classmethod
    def get_parallel_rank_env_variable(self) -> str:
        return "${FLUX_TASK_RANK}"

    @classmethod
    def get_parallel_configuration(cls) -> tuple[int, int, int, int]:
        env_vars = [
            "FLUX_JOB_SIZE",
            "FLUX_TASK_RANK",
            "FLUX_TASK_LOCAL_ID",
            "FLUX_JOB_NNODES",
        ]
        env = {}
        for e in env_vars:
            if not os.getenv(e):
                msg = (
                    f"Unable to launch torchrun_hpc on FLUX scheduler - {e} not defined"
                )
                raise Exception(msg)
            else:
                env[e] = int(os.getenv(e))

        world_size = env["FLUX_JOB_SIZE"]
        rank = env["FLUX_TASK_RANK"]
        local_rank = env["FLUX_TASK_LOCAL_ID"]
        nodes_per_job = env["FLUX_JOB_NNODES"]
        local_world_size = world_size // nodes_per_job
        return (world_size, rank, local_world_size, local_rank)

    def dynamically_configure_rendezvous_protocol(self, protocol: str) -> list[str]:
        env_list = []
        env_list.append(("RANK", self.get_parallel_rank_env_variable()))
        if protocol.lower() == "tcp":
            env_list.append(
                (
                    "TORCHRUN_HPC_MASTER_ADDR",
                    "`flux hostlist local | /bin/hostlist -n 1`",
                )
            )
            env_list.append(("TORCHRUN_HPC_MASTER_PORT", "23456"))
            return env_list
        elif protocol.lower() == "mpi":
            # To use MPI, pass `init_method="mpi://"` - no special work here.
            return env_list
        else:
            msg = f"Unsupported rendezvous protocol {protocol} for scheduler {type(self).__name__}"
            raise Exception(msg)
