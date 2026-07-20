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
import shlex

if TYPE_CHECKING:
    # If type-checking, import the other class
    from hpc_launcher.systems import System

from hpc_launcher.schedulers.scheduler import Scheduler
from hpc_launcher.schedulers import parse_env_list


@dataclass
class LSFScheduler(Scheduler):

    # bsub flags that take a value using LSF's native space-separated CLI
    # syntax (e.g. ``-nnodes 2``), not the GNU-style ``--flag=value`` syntax
    # used by Slurm/Flux (finding E1). Keyed by flag name so the correct
    # formatting applies regardless of which argument bucket a given flag
    # currently lives in.
    _SPACE_SEPARATED_FLAGS = frozenset(
        {
            "-nnodes",
            "-J",
            "-q",
            "-G",
            "-U",
            "-W",
            "-o",
            "-e",
            "-cwd",
            "--chdir",
            "--shared-launch",
        }
    )

    def build_scheduler_specific_arguments(
        self, system: "System", blocking: bool = True
    ):
        # Number of Nodes
        self.run_only_args["--nrs"] = f"{self.nodes}"
        self.common_launch_args["-nnodes"] = f"{self.nodes}"

        self.common_launch_args["--shared-launch"] = None

        # jsrun options (do we need to guard this with something like if os.getenv("LSB_HOSTS"):
        self.run_only_args["--rs_per_host"] = "1"
        self.run_only_args["--tasks_per_rs"] = f"{self.procs_per_node}"
        self.run_only_args["--launch_distribution"] = "packed"
        self.run_only_args["--cpu_per_rs"] = "ALL_CPUS"
        self.run_only_args["--gpu_per_rs"] = "ALL_GPUS"

        if self.out_log_file and not blocking:
            self.submit_only_args["-o"] = f"{self.out_log_file}"
        if self.err_log_file and not blocking:
            self.submit_only_args["-e"] = f"{self.err_log_file}"

        # Configure header with LSF job options
        if self.time_limit:
            minutes = int(round(max(self.time_limit, 0)))
            hours, minutes = divmod(minutes, 60)
            self.submit_only_args["-W"] = f"{hours}:{minutes:02}"
        if self.job_name:
            # Flag and value are now separate dict entries, so the job name
            # flows through as a normal value and is quoted by the
            # scheduler's central serialization (format_common_arg /
            # build_command_string_and_batch_script) wherever it is emitted
            # (finding D1). No source-level quoting needed here.
            self.common_launch_args["-J"] = self.job_name
        if self.queue:
            self.common_launch_args["-q"] = f"{self.queue}"
        if self.account:
            self.common_launch_args["-G"] = f"{self.account}"
        if self.reservation:
            self.submit_only_args["-U"] = f"{self.reservation}"

        if self.work_dir:
            if blocking:
                # Use on the command line
                self.submit_only_args["--chdir"] = f"{self.work_dir}"
            else:
                # Add to the batch script #BSUB
                self.submit_only_args["-cwd"] = f"{self.work_dir}"

        return

    def _format_lsf_arg(self, k: str, v: Optional[str], quote_value: bool = False) -> list[str]:
        # Both launch paths execute argv lists (or an internal run-command
        # line built from them) without a shell, so a bsub flag and its
        # value must be separate tokens like ``["-nnodes", "2"]`` rather than
        # one token containing a literal space (finding E1).
        if not v:
            return [k]
        val = shlex.quote(v) if quote_value else v
        if k in self._SPACE_SEPARATED_FLAGS:
            return [k, val]
        return [f"{k}={val}"]

    def format_submit_arg(self, k: str, v: Optional[str], quote_value: bool = False) -> list[str]:
        return self._format_lsf_arg(k, v, quote_value)

    def format_common_arg(self, k: str, v: Optional[str], quote_value: bool = False) -> list[str]:
        return self._format_lsf_arg(k, v, quote_value)

    def batch_script_prefix(self) -> str:
        return "#BSUB"

    def blocking_launch_command(self) -> list[str]:
        if os.getenv("LSB_HOSTS"):
            return ["jsrun"]
        else:
            return ["bsub", "-Is"]

    def nonblocking_launch_command(self) -> list[str]:
        return ["bsub"]

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

        key_found = False
        for key in self.submit_only_args:
            if key.startswith("--env"):
                existing_env = key.split(" ")
                new_env = existing_env[2:]
                stripped_env = [s.strip(",") for s in new_env]
                cleaned_env = [s.strip('"') for s in stripped_env]
                revised_env = cleaned_env + env_vars
                new_key = '--env "ALL, ' + ", ".join(revised_env) + '"'
                self.submit_only_args[new_key] = None
                del self.submit_only_args[key]
                key_found = True

        if not key_found:
            self.submit_only_args['--env "ALL, ' + ", ".join(env_vars) + '"'] = None
        return

    def export_hostlist(self) -> str:
        return "export HPC_LAUNCHER_HOSTLIST=$(echo $LSB_HOSTS | tr ' ' '\\n' | sort -u)\n"

    def enable_run_args_on_launch_command(self) -> bool:
        if os.getenv("LSB_HOSTS"):
            return True
        else:
            return False

    def require_parallel_internal_run_command(self, blocking: bool) -> bool:
        if not blocking or (blocking and not os.getenv("LSB_HOSTS")):
            return True
        else:
            return False

    def internal_script_run_command(self) -> str:
        return "jsrun "

    def get_job_id(self, output: str) -> Optional[str]:
        raise NotImplementedError

    @classmethod
    def num_nodes_in_allocation(cls) -> Optional[int]:
        if os.getenv("LLNL_NUM_COMPUTE_NODES"):
            return int(os.getenv("LLNL_NUM_COMPUTE_NODES"))

        return None

    @classmethod
    def get_parallel_rank_env_variable(self) -> str:
        return "${OMPI_COMM_WORLD_RANK}"

    @classmethod
    def get_parallel_configuration(cls) -> tuple[int, int, int, int]:
        env_vars = [
            "OMPI_COMM_WORLD_SIZE",
            "OMPI_COMM_WORLD_RANK",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "OMPI_COMM_WORLD_LOCAL_SIZE",
        ]
        env = {}
        for e in env_vars:
            if not os.getenv(e):
                msg = (
                    f"Unable to launch torchrun_hpc on LSF scheduler - {e} not defined"
                )
                raise Exception(msg)
            else:
                env[e] = int(os.getenv(e))

        world_size = env["OMPI_COMM_WORLD_SIZE"]
        rank = env["OMPI_COMM_WORLD_RANK"]
        local_rank = env["OMPI_COMM_WORLD_LOCAL_RANK"]
        local_world_size = env["OMPI_COMM_WORLD_LOCAL_SIZE"]
        return (world_size, rank, local_world_size, local_rank)

    def dynamically_configure_rendezvous_protocol(self, protocol: str) -> list[str]:
        env_list = []
        env_list.append(("RANK", self.get_parallel_rank_env_variable()))
        if protocol.lower() == "tcp":
            if os.getenv("LSB_HOSTS"):
                # When runing under an allocation use the current node as the coordinator
                env_list.append(("TORCHRUN_HPC_MASTER_ADDR", os.getenv("HOSTNAME")))
            else:
                env_list.append(
                    ("TORCHRUN_HPC_MASTER_ADDR", "`jsrun --nrs 1 -r 1 /bin/hostname`")
                )
            env_list.append(("TORCHRUN_HPC_MASTER_PORT", "23456"))
            return env_list
        elif protocol.lower() == "mpi":
            # To use MPI, pass `init_method="mpi://"` - no special work here.
            return env_list
        else:
            msg = f"Unsupported rendezvous protocol {protocol} for scheduler {type(self).__name__}"
            raise Exception(msg)
