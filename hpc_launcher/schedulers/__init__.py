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
import os
import re
import subprocess
from typing import Optional


def num_nodes_in_current_allocation() -> Optional[int]:
    """
    The node count of the scheduler allocation this process is running
    inside, or ``None`` when not inside an allocation.

    Unlike ``Scheduler.num_nodes_in_allocation`` this is scheduler-agnostic:
    it is consulted *before* a scheduler has been selected (CLI argument
    validation), so it probes every scheduler's environment marker rather
    than assuming one. The probes mirror the per-scheduler classmethods:
    Flux (``FLUX_URI``), Slurm (``SLURM_JOB_NUM_NODES``), and LSF
    (``LLNL_NUM_COMPUTE_NODES``).

    :return: Number of nodes in the enclosing allocation, or None.
    """
    if os.getenv("FLUX_URI"):
        proc = subprocess.run(
            ["flux", "resource", "info"],
            universal_newlines=True,
            capture_output=True,
        )
        m = re.search(r"^(\d+) Nodes, (\d+) Cores, (\d+) GPUs$", proc.stdout)
        if m:
            return int(m.group(1))
    if os.getenv("SLURM_JOB_NUM_NODES"):
        return int(os.getenv("SLURM_JOB_NUM_NODES"))
    if os.getenv("LLNL_NUM_COMPUTE_NODES"):
        return int(os.getenv("LLNL_NUM_COMPUTE_NODES"))
    return None


def get_schedulers():
    from .local import LocalScheduler
    from .flux import FluxScheduler
    from .slurm import SlurmScheduler
    from .lsf import LSFScheduler

    return {
        None: LocalScheduler,
        "local": LocalScheduler,
        "LocalScheduler": LocalScheduler,
        "flux": FluxScheduler,
        "FluxScheduler": FluxScheduler,
        "slurm": SlurmScheduler,
        "SlurmScheduler": SlurmScheduler,
        "lsf": LSFScheduler,
        "LSFScheduler": LSFScheduler,
    }

def parse_env_list(*e) -> str:
    if len(e) == 1:
        m = e[0]
        return f"{m}\n"
    elif len(e) == 2:
        k,v = e
        return f"export {k}={v}\n"
    elif len(e) == 3:
        k,v,m = e
        return f"export {k}={v}\t\t# {m}\n"
    else:
        return f'{e}'
