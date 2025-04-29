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

    def select_interactive_or_batch(
        self,
        # tmp: list[str],
        k: str,
        v: str,
        header: StringIO,
        cmd_args: list[str],
        blocking: bool = True,
    ) -> None:
        if blocking:
            # cmd_args += tmp
            self.run_launch_args[k] = v
        else:
            # header.write(f'# FLUX: {" ".join(tmp)}\n')
            self.batch_script_header[k] = v
            self.batch_submit_args[k] = v
        return

    def build_scheduler_specific_arguments(
        self, blocking: bool = True
    ):
        if self.out_log_file and not blocking:
            self.submit_only_args[f"--output"] = f"{self.out_log_file}"
        if self.err_log_file and not blocking:
            self.submit_only_args[f"--error"] = f"{self.err_log_file}"

        # Unbuffered output
        self.common_launch_args["-u"] = None

        # Number of Nodes
        self.common_launch_args[f"-N{self.nodes}"] = None

        # Total number of Tasks / Processes
        self.common_launch_args[f"-n{self.nodes * self.procs_per_node}"] = None

        # Set the Number of GPUs per task
        # There is a difference in option names between tasks and allocations
        if self.gpus_per_proc > 0:
            tmp = f"{self.gpus_per_proc}"
            # command line flag for a task
            self.run_only_args["--gpus-per-task"] = tmp
            # command and shell flags for an allocation
            if not blocking:
                self.submit_only_args["--gpus-per-slot"] = tmp

        if self.work_dir:
            self.submit_only_args["--setattr=system.cwd"] = f"{os.path.abspath(self.work_dir)}"

        self.common_launch_args["-onosetpgrp"] = None

        if self.ld_preloads:
            self.common_launch_args['--env=LD_PRELOAD'] = f'{",".join(self.ld_preloads)}'

        if self.time_limit is not None:
            self.common_launch_args["--time"] = f"{self.time_limit}m"

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
            self.submit_only_args["--account"] = f"{self.account}"

        if self.reservation:
            logger.warning(
                f"WARNING: Unsupported option requested: --reservation={self.reservation}"
            )

        print(self.common_launch_args)
        
        return

#     def build_command_string_and_batch_script(
#         self, system: "System", blocking: bool = True
#     ) -> (str, list[str]):

#         env_vars = system.environment_variables()
#         passthrough_env_vars = system.passthrough_environment_variables()
#         # Enable the system to apply some customization to the scheduler instance
#         system.customize_scheduler(self)

#         header = StringIO()
#         header.write("#!/bin/sh\n")
#         cmd_args = []
#         if self.out_log_file and not blocking:
#             self.submit_only_args[f"--output"] = f"{self.out_log_file}"
#             # self.batch_script_header[f"--output"] = f"{self.out_log_file}"
#         if self.err_log_file and not blocking:
#             self.submit_only_args[f"--error"] = f"{self.err_log_file}"
#             # self.batch_script_header[f"--error"] = f"{self.err_log_file}"

#         # Unbuffered output
#         # tmp = "-u"
#         self.common_launch_args["-u"] = None
#         # self.run_launch_args[tmp] = None
#         # if not blocking:
#         #     self.batch_script_header[tmp] = None

#         # Number of Nodes
#         # tmp = f"-N{self.nodes}"
#         # cmd_args += [tmp]
#         self.common_launch_args[f"-N{self.nodes}"] = None
#         # self.run_launch_args[tmp] = None
#         # if not blocking:
#         #     # header.write(f"# FLUX: {tmp}\n")
#         #     self.batch_script_header[tmp] = None

#         # Total number of Tasks / Processes
#         # tmp = f"-n{self.nodes * self.procs_per_node}"
#         # cmd_args += [tmp]
#         self.common_launch_args[f"-n{self.nodes * self.procs_per_node}"] = None
#         # self.run_launch_args[tmp] = None
#         # if not blocking:
#         #     # header.write(f"# FLUX: {tmp}\n")
#         #     self.batch_script_header[tmp] = None

#         # Set the Number of GPUs per task
#         # There is a difference in option names between tasks and allocations
#         if self.gpus_per_proc > 0:
#             tmp = f"{self.gpus_per_proc}"
#             # command line flag for a task
#             self.run_only_args["--gpus-per-task"] = tmp
#             # self.run_launch_args["--gpus-per-task"] = tmp
#             # command and shell flags for an allocation
#             if not blocking:
#                 self.submit_only_args["--gpus-per-slot"] = tmp
#             # self.batch_submit_args["--gpus-per-slot"] = tmp
#             # self.batch_script_header["--gpus-per-slot"] = tmp

#         if self.work_dir:
#             # tmp = [f"--setattr=system.cwd={os.path.abspath(self.work_dir)}"]
#             self.submit_only_args["--setattr=system.cwd"] = f"{os.path.abspath(self.work_dir)}"
#             # self.select_interactive_or_batch("--setattr=system.cwd", f"{os.path.abspath(self.work_dir)}", header, cmd_args, blocking)

#         # tmp = ["-onosetpgrp"]
#         self.common_launch_args["-onosetpgrp"] = None
#         # self.select_interactive_or_batch("-onosetpgrp", None, header, cmd_args, blocking)

#         if self.ld_preloads:
# #            tmp = [f'--env=LD_PRELOAD={",".join(self.ld_preloads)}']
#             self.common_launch_args['--env=LD_PRELOAD'] = f'{",".join(self.ld_preloads)}'
#             # self.select_interactive_or_batch('--env=LD_PRELOAD', f'{",".join(self.ld_preloads)}', header, cmd_args, blocking)

#         if self.time_limit is not None:
#             self.common_launch_args["--time"] = f"{self.time_limit}m"
#             # self.select_interactive_or_batch("--time", f"{self.time_limit}m", header, cmd_args, blocking)

#         if self.job_name:
#             # tmp = [f"--job-name={self.job_name}"]
#             self.common_launch_args["--job-name"] = f"{self.job_name}"
#             # self.select_interactive_or_batch("--job-name", f"{self.job_name}", header, cmd_args, blocking)

#         if self.queue:
#             if os.getenv("FLUX_URI"):
#                 logger.warning(
#                     f"WARNING: Dropping unsupported option requested when running inside of an allocation: --queue={self.queue}"
#                 )
#             else:
#                 # tmp = [f"--queue={self.queue}"]
#                 self.submit_only_args["--queue"] = f"{self.queue}"
#                 # self.select_interactive_or_batch("--queue", f"{self.queue}", header, cmd_args, blocking)

#         if self.account:
#             # tmp = [f"--account={self.account}"]
#             self.submit_only_args["--account"] = f"{self.account}"
#             # self.select_interactive_or_batch("--account", f"{self.account}", header, cmd_args, blocking)

#         if self.reservation:
#             logger.warning(
#                 f"WARNING: Unsupported option requested: --reservation={self.reservation}"
#             )

#         if self.launcher_flags:
#             for flag in self.launcher_flags:
#                 # These flag should only be on the launcher commands not the batch commands
#                 # cmd_args += [flag]
#                 # If an = exists, split on the last only
#                 k = flag.rsplit("=", 1)
#                 print(f'BVE I have a flag {flag}')
#                 if len(k) == 1:
#                     self.common_launch_args[k[0]] = None
#                     # self.run_launch_args[k[0]] = None
#                 elif len(k) == 2:
#                     print(f'BVE I have {k[0]}, {k[1]} from {flag}')
#                     self.common_launch_args[k[0]] = k[1]
#                     # self.run_launch_args[k[0]] = k[1]
#                 else:
#                     logger.error(f"Unknown launcher flag {flag}")
#                     exit(1)

#         print(f'BVE I have override args {self.override_launch_args}')
#         if self.override_launch_args:
#             for k,v in self.override_launch_args.items():
#                 print(f'BVE I have found {k}={v}')
#                 # if k in self.batch_script_header:
#                 #     print(f'BVE I have found {k} in header {self.batch_script_header}')
#                 if k in self.common_launch_args:
#                     tmp = self.common_launch_args[k]
#                     print(f'BVE I have found {k} in common_launch_args {self.common_launch_args}={tmp}')
#                     self.common_launch_args[k] = v
#                 elif k in self.run_only_args:
#                     tmp = self.run_only_args[k]
#                     print(f'BVE I have found {k} in run {self.run_only_args}={tmp}')
#                     self.run_only_args[k] = v
#                 elif k in self.submit_only_args:
#                     tmp = self.submit_only_args[k]
#                     print(f'BVE I have found {k} in run {self.submit_only_args}={tmp}')
#                     self.submit_only_args[k] = v
#                 else:
#                     print(f'BVE adding unqiue override found {k} in run {self.run_args}')
#                     self.common_launch_args[k] = v
                    
#         if not blocking: # Only add batch script header items on non-blocking calls
#             # for k,v in self.batch_script_header.items():
#             for k,v in self.submit_only_args.items():
#                 if not v:
#                     header.write(f"# FLUX: {k}\n")
#                 else:
#                     header.write(f"# FLUX: {k}={v}\n")
#             for k,v in self.common_launch_args.items():
#                 if not v:
#                     header.write(f"# FLUX: {k}\n")
#                 else:
#                     header.write(f"# FLUX: {k}={v}\n")

#         for e in env_vars:
#             header.write(parse_env_list(*e))

#         for k, v in passthrough_env_vars:
#             if not blocking:
#                 cmd_args += [f" --env={k}={v}"]
#             else:
#                 header += f"export {k}={v}\n"

#         return (header.getvalue(), cmd_args)

    def batch_script_prefix(self) -> str:
        return "# FLUX:"

    def blocking_launch_command(self) -> list[str]:
        return ["flux", "run"]

    def nonblocking_launch_command(self) -> list[str]:
        return ["flux", "batch"]

    def export_hostlist(self) -> str:
        return "export HPC_LAUNCHER_HOSTLIST=$(flux hostlist local)\n"

    def batch_script_run_command(self) -> str:
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
        env_list.append(("RANK", "${FLUX_TASK_RANK}"))
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
