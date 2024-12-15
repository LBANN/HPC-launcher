import logging
from hpc_launcher.systems import autodetect
from hpc_launcher.systems.system import System
from hpc_launcher.utils import ceildiv

logger = logging.getLogger(__name__)


def configure_launch(queue: str, nodes: int, procs_per_node: int,
                     gpus_at_least: int,
                     gpumem_at_least: int) -> tuple[System, int, int]:
    """
    See if the system can be autodetected and then process some special
    arguments that can autoselect the number of ranks / GPUs.

    :param queue: The queue to use for the job
    :param nodes: The number of nodes to use (or 0 if not specified)
    :param procs_per_node: The number of processes per node given by the user
                           (or 0 if not specified)
    :param gpus_at_least: The minimum number of GPUs to use (or 0 if not
                          specified)
    :param gpumem_at_least: The minimum amount of GPU memory (in gigabytes) to
                            use (or 0 if not specified)
    :return: A tuple of (autodetected System, number of nodes, number of
             processes per node)
    """
    system = autodetect.autodetect_current_system()
    logger.info(
        f'Detected system: {system.system_name} [{type(system).__name__}-class]'
    )
    system_params = system.system_parameters(queue)

    # If the user requested a specific number of processes per node, honor that
    if nodes and procs_per_node:
        return system, nodes, procs_per_node

    # Otherwise, if there is a valid set of system parameters, try to fill in
    # the blanks provided by the user
    if system_params is not None:
        if not procs_per_node:
            procs_per_node = system_params.procs_per_node()
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

    return system, nodes, procs_per_node
