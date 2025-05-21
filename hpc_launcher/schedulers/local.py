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
from hpc_launcher.schedulers.scheduler import Scheduler
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import os
import logging
from hpc_launcher.schedulers import parse_env_list

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # If type-checking, import the other class
    from hpc_launcher.systems.system import System


@dataclass
class LocalScheduler(Scheduler):
    """
    A class that runs the job without any underlying batch scheduler. Used
    in ``--local`` jobs.
    """

    def launch_command(self, system: "System", blocking: bool = True, cli_env_only: bool = False) -> list[str]:
        return []

    def launcher_script(
        self,
        system: "System",
        command: str,
        args: Optional[list[str]] = None,
        blocking: bool = True,
        save_hostlist: bool = False,
        launch_dir: str = "",
    ) -> str:
        envvars = [parse_env_list(*e) for e in system.environment_variables()]
        envvars += [
            f"export {k}={v}" for k, v in system.passthrough_environment_variables()
        ]
        envvars += [
            "export RANK=0",
        ]
        if save_hostlist:
            envvars += [
                "export HPC_LAUNCHER_HOSTLIST=$(hostname)\n",
                'if [ "${RANK}" = "0" ]; then\n',
                "    echo ${HPC_LAUNCHER_HOSTLIST} > " + os.path.join(launch_dir, f"hpc_launcher_hostlist.txt\n"),
                "fi\n\n",
            ]
        header = "\n".join(envvars)

        if self.work_dir:
            header += f"\ncd {os.path.abspath(self.work_dir)}\n"

        return f"""#!/bin/sh
# Setup
{header}

# Run
{command} {" ".join(args)}
"""

    def get_job_id(self, output: str) -> Optional[str]:
        return None

    @classmethod
    def get_parallel_configuration(cls) -> tuple[int, int, int, int]:
        return 1, 0, 1, 0

    def dynamically_configure_rendezvous_protocol(self, protocol: str) -> list[str]:
        env_list = []
        if protocol.lower() == "tcp":
            env_list.append(("TORCHRUN_HPC_MASTER_ADDR", "localhost"))
            env_list.append(("TORCHRUN_HPC_MASTER_PORT", "23456"))
            return env_list
        else:
            msg = f"Unsupported rendezvous protocol {protocol} for scheduler {type(self).__name__}"
            raise Exception(msg)
