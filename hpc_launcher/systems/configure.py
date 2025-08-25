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
import logging
from typing import Optional
from dataclasses import dataclass, fields, asdict
from hpc_launcher.systems import autodetect
from hpc_launcher.systems.system import System, SystemParams
from hpc_launcher.utils import ceildiv

logger = logging.getLogger(__name__)

def convert_to_type_of_another(variable_to_convert, reference_variable):
    return type(reference_variable)(variable_to_convert)

def configure_launch(
    queue: str,
    nodes: int,
    procs_per_node: int,
    gpus_per_proc: Optional[int],
    gpus_at_least: int = 0,
    gpumem_at_least: int = 0,
    cli_system_params: Optional[dict[str, str]] = None,
    job_comm_protocol: Optional[str] = None,
) -> tuple[System, int, int, int]:
    """
    See if the system can be autodetected and then process some special
    arguments that can autoselect the number of ranks / GPUs.

    :param queue: The queue to use for the job
    :param nodes: The number of nodes to use (or 0 if not specified)
    :param procs_per_node: The number of processes per node given by the user
                           (or 0 if not specified)
    :param gpus_per_proc: The number of GPUs per process given by the user
                           (or None if not specified)
    :param gpus_at_least: The minimum number of GPUs to use (or 0 if not
                          specified)
    :param gpumem_at_least: The minimum amount of GPU memory (in gigabytes) to
                            use (or 0 if not specified)
    :param cli_system_params: CLI provide description of the system configuration
                            (or None if not specified)
    :param job_comm_protocol: CLI provide description of the jos intended communication protocol
                            (or None if not specified)
    :return: A tuple of (autodetected System, number of nodes, number of
             processes per node)
    """
    system = autodetect.autodetect_current_system()
    # Pass the job's intended communication protocol to the system object
    system.job_comm_protocol = job_comm_protocol
    logger.info(
        f"Detected system: {system.system_name} [{type(system).__name__}-class]"
    )
    system_params = system.system_parameters(queue)

    # If any system parameters were provided on the command line, potentially overriding any known or discovered system parameters
    msg = ""
    if cli_system_params:
        msg = " (CLI Override) "
        if not system_params: # Use a default set of system parameters
            # for the active system params
            system.active_system_params = SystemParams()
            system_params = system.active_system_params()
        for field in fields(system_params):
            if field.name in cli_system_params:
                system_params.__dict__[field.name] = convert_to_type_of_another(cli_system_params[field.name], system_params.__dict__[field.name])
                del cli_system_params[field.name]

        for unused_field in cli_system_params.keys():
            raise ValueError(f"System Parameters CLI attempt to overwrite unknown field: {unused_field}")

    logger.info(
        f"Active System Parameters{msg}: {system.active_system_params.prettyprint()}"
    )

    if not gpus_per_proc:
        gpus_per_proc = 0

    # If not provided, attempt to figure out the basics of procs_per_node and gpus_per_proc
    if system_params is not None:
        if not procs_per_node:
            procs_per_node = system_params.procs_per_node()
        if gpus_per_proc == 0 and system_params.gpus_per_node > 0:
            # If gpus_per_proc wasn't set and there are gpus on the node set it to a default of 1
            gpus_per_proc = 1
        if procs_per_node * gpus_per_proc > system_params.gpus_per_node:
            logger.info(
                f"Requested number of GPUs per process {gpus_per_proc} exceeds the number of GPUs per node {system_params.gpus_per_node}"
            )
            # If no, or an invalid, configuration is given, set the gpus_per_proc
            if gpus_per_proc == 0 or gpus_per_proc > system_params.gpus_per_node:
                gpus_per_proc = max(system_params.gpus_per_node // procs_per_node, 1)

        if procs_per_node and procs_per_node * gpus_per_proc > system_params.gpus_per_node:
            logger.info(
                f"The combination of {procs_per_node} processes per node and {gpus_per_proc} GPUs per process exceeds the number of GPUs per node {system_params.gpus_per_node} - Job will not launch, please fix requested parameters"
            )

    # If the user requested a specific number of processes per node, honor that
    if nodes and procs_per_node:
        return system, nodes, procs_per_node, gpus_per_proc

    # Otherwise, if there is a valid set of system parameters, try to fill in
    # the blanks provided by the user
    if system_params is not None:
        if gpus_at_least > 0:
            nodes = ceildiv(gpus_at_least, procs_per_node)
        elif gpumem_at_least > 0:
            num_gpus = ceildiv(gpumem_at_least, system_params.mem_per_gpu)
            nodes = ceildiv(num_gpus, procs_per_node)
            if nodes == 1:
                procs_per_node = num_gpus
    else:
        # If no system parameters are available, fall back to one process
        if not nodes:
            nodes = 1
        if not procs_per_node:
            procs_per_node = 1
        if not gpus_per_proc:
            gpus_per_proc = 1

    return system, nodes, procs_per_node, gpus_per_proc
