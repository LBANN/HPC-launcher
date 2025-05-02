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
import argparse
#from hpc_launcher.systems.system import System, SystemParams
#from hpc_launcher.schedulers.scheduler import Scheduler
from hpc_launcher.systems.configure import autodetect
from hpc_launcher.cli import common_args, launch_helpers
from unittest.mock import patch

from hpc_launcher.systems.lc.el_capitan_family import ElCapitan

# Instantiate a system

# Get an mock el cap system
# override the arguments
# see if the override propagates
import logging

logger = logging.getLogger(__name__)



@patch(
    "hpc_launcher.systems.autodetect.autodetect_current_system",
    return_value=ElCapitan("tuolumne"),
)
# @patch(
#     "hpc_launcher.cli.launch_helpers.select_scheduler",
#     return_value=FluxScheduler(),
# )

def test_cli_argument_override(*xargs):
#def test_cli_argument_override(*args, nodes, procs_per_node, gpus_per_proc):

    system = autodetect.autodetect_current_system()

    print(f"BVE I think that I have system {system}")

#    scheduler = system.preferred_scheduler
    parser = argparse.ArgumentParser(
        description="Launches a distributed job on the current HPC cluster or cloud."
    )
    common_args.setup_arguments(parser)
    args = parser.parse_args()

    args.nodes = xargs[1]
    args.procs_per_node = xargs[2]
    args.gpus_per_proc = xargs[3]
    blocking = xargs[4]
    print(f"bve {args}")
    scheduler = launch_helpers.select_scheduler(args, logger, system)
    
    print(f"BVE I think that I have run_args {scheduler.run_only_args}")
    
    # scheduler = launch_helpers.select_scheduler(args, logger, system)

    #     self.override_launch_args = override_launch_args
        
    cmd = scheduler.launch_command(system, blocking)
    print(f"BVE here is my command {cmd}")

if __name__ == "__main__":
    test_cli_argument_override(2, 2, 1, False)
        
