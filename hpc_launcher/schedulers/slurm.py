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
    from hpc_launcher.systems.system import System

from hpc_launcher.systems.lc.sierra_family import Sierra

from hpc_launcher.schedulers.scheduler import Scheduler
from hpc_launcher.schedulers import parse_env_list

import logging

logger = logging.getLogger(__name__)


def _time_string(minutes):
    """Time D-hh:mm:ss format."""
    minutes = max(minutes, 0)
    seconds = int(round((minutes % 1) * 60))
    hours, minutes = divmod(int(minutes), 60)
    days, hours = divmod(hours, 24)
    return f"{days}-{hours:02}:{minutes:02}:{seconds:02}"


@dataclass
class SlurmScheduler(Scheduler):

    def build_scheduler_specific_arguments(
        self, system: "System", blocking: bool = True
    ):
        if self.out_log_file and not blocking:
            self.submit_only_args["--output"] = f"{self.out_log_file}"
        if self.err_log_file and not blocking:
            self.submit_only_args["--error"] = f"{self.err_log_file}"

        # Unbuffered output - Only pass to srun
        if blocking and not isinstance(system, Sierra):
            # On Sierra family systems srun is a proxy to lrun and lacks this flag
            self.run_only_args["-u"] = None

        # Number of Nodes
        self.common_launch_args["--nodes"] = f"{self.nodes}"

        # Total number of Tasks / Processes
        self.common_launch_args["--ntasks"] = f"{self.nodes * self.procs_per_node}"

        # Number of Tasks per node
        self.common_launch_args["--ntasks-per-node"] = f"{self.procs_per_node}"

        # Set the Number of GPUs per task
        if self.gpus_per_proc > 0:
            self.common_launch_args["--gpus-per-task"] = f"{self.gpus_per_proc}"

        # Request for node exclusivity
        if self.exclusive:
            self.submit_only_args["--exclusive"] = ""

        if self.work_dir:
            self.submit_only_args["--chdir"] = f"{os.path.abspath(self.work_dir)}"

        if self.ld_preloads:
            self.common_launch_args['--export=ALL,LD_PRELOAD'] = f'{",".join(self.ld_preloads)}'

        if self.time_limit is not None:
            self.common_launch_args["--time"] = f"{_time_string(self.time_limit)}"

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
            self.submit_only_args["--partition"] = f"{self.queue}"

        if self.account:
            self.submit_only_args["--account"] = f"{self.account}"

        if self.reservation:
            self.submit_only_args["--reservation"] = f"{self.reservation}"

        return

    def batch_script_prefix(self) -> str:
        return "#SBATCH"

    def blocking_launch_command(self) -> list[str]:
        return ["srun"]

    def nonblocking_launch_command(self) -> list[str]:
        return ["sbatch"]

    def cli_env_arg(self, env_list) -> None:
        env_vars = []
        for e in env_list:
            if len(e) == 1:
                continue
            elif len(e) == 2:
                k,v = e
                env_vars += [f"{k}={v}"]
            elif len(e) == 3:
                k,v,m = e
                env_vars += [f"{k}={v}"]
        if "--export" in self.submit_only_args:
            self.submit_only_args["--export"] += "," + ",".join(env_vars)
        else:
            self.submit_only_args["--export"] = "ALL," + ",".join(env_vars)
        return

    def export_hostlist(self) -> str:
        return "export HPC_LAUNCHER_HOSTLIST=${SLURM_JOB_NODELIST}\n"

    def internal_script_run_command(self) -> str:
        return "srun -u "

    def get_job_id(self, output: str) -> Optional[str]:
        # The job ID is the last number in the printout
        last_line = output.strip().split("\n")[-1].strip()
        if last_line.startswith("Submitted batch job"):
            return last_line.split(" ")[-1]
        return None

    @classmethod
    def num_nodes_in_allocation(cls) -> Optional[int]:
        if os.getenv("FLUX_URI"):
            cmd = ["flux", "resource", "info"]
            proc = subprocess.run(cmd, universal_newlines=True, capture_output=True)
            m = re.search(r"^(\d*) Nodes, (\d*) Cores, (\d*) GPUs$", proc.stdout)
            if m:
                return int(m.group(1))
        elif os.getenv("SLURM_JOB_NUM_NODES"):
            return int(os.getenv("SLURM_JOB_NUM_NODES"))
        elif os.getenv("LLNL_NUM_COMPUTE_NODES"):
            return int(os.getenv("LLNL_NUM_COMPUTE_NODES"))

        return None

    @classmethod
    def get_parallel_rank_env_variable(self) -> str:
        return "${SLURM_PROCID}"

    @classmethod
    def get_parallel_configuration(cls) -> tuple[int, int, int, int]:
        # Interesting but unused variables SLURM_JOB_NUM_NODES, SLURM_NPROCS, SLURM_DISTRIBUTION
        # Skipping 'SLURM_TASKS_PER_NODE' because this field has a weird format e.g. 2(x2)
        env_vars = ["SLURM_NTASKS", "SLURM_PROCID", "SLURM_LOCALID", "SLURM_NNODES"]
        env = {}
        for e in env_vars:
            if not os.getenv(e):
                msg = f"Unable to launch torchrun_hpc on SLURM scheduler - {e} not defined"
                raise Exception(msg)
            else:
                env[e] = int(os.getenv(e))

        world_size = env["SLURM_NTASKS"]
        rank = env["SLURM_PROCID"]
        local_rank = env["SLURM_LOCALID"]
        nodes_per_job = env["SLURM_NNODES"]
        local_world_size = world_size // nodes_per_job
        # local_world_size = env['SLURM_TASKS_PER_NODE']
        return (world_size, rank, local_world_size, local_rank)

    @classmethod
    def dynamically_configure_rendezvous_protocol(self, protocol: str) -> str:
        env_list = []
        env_list.append(("RANK", self.get_parallel_rank_env_variable()))
        if protocol.lower() == "tcp":
            env_list.append(
                (
                    "TORCHRUN_HPC_MASTER_ADDR",
                    "`scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1`",
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
