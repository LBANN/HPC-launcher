# Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
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
from hpc_launcher.systems.system import System, GenericSystem, SystemParams
from hpc_launcher.systems.lc.el_capitan_family import ElCapitan
from hpc_launcher.systems.lc.cts2 import CTS2
from hpc_launcher.systems.lc.sierra_family import Sierra
import logging
import socket
import re
import math
import os

logger = logging.getLogger(__name__)

# Detect system lazily
_system = None

# ==============================================
# Access functions
# ==============================================

def find_AMD_gpus() -> (int, float, str):
    try:
        from pyrsmi import rocml
        try:
            rocml.smi_initialize()
        except:
            return (0, 0, None)

        num_gpus = rocml.smi_get_device_count()
        mem_per_gpu = 0
        gpu_arch = None
        if num_gpus > 0:
            mem_per_gpu = math.floor(rocml.smi_get_device_memory_total(0) / (1024*1024*1024))
            device_id = rocml.smi_get_device_id(0)
            if device_id == 0x74a0:
                gpu_arch = 'gfx942'
            if device_id == 0x7408:
                gpu_arch = 'gfx90a'
        rocml.smi_shutdown()
        return (num_gpus, mem_per_gpu, gpu_arch)
    except (ImportError, ModuleNotFoundError):
        return (0, 0, None)

def find_NVIDIA_gpus() -> (int, float, str):
    try:
        import pynvml
        num_gpus = 0
        mem_per_gpu = 0
        gpu_arch = None
        try:
            pynvml.nvmlInit()
        except:
            return (0, 0, None)

        deviceCount = pynvml.nvmlDeviceGetCount()
        # Assume that the GPUs are homogenous on a system
        if deviceCount > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_arch = f"sm_{major}{minor}"
            mem_per_gpu = math.floor(info.total / (1024*1024*1024))
        return (deviceCount, mem_per_gpu, gpu_arch)
    except (ImportError, ModuleNotFoundError):
        return (0, 0, None)

def find_gpus() -> (str, int, float, str):
    (num_AMD_gpus, mem_per_AMD_gpu, AMD_arch) = find_AMD_gpus()
    (num_NVIDIA_gpus, mem_per_NVIDIA_gpu, NVIDIA_arch) = find_NVIDIA_gpus()
    if num_AMD_gpus == 0 and num_NVIDIA_gpus == 0:
        print('Unable to autodetect any GPUs on this system')
        return ("Generic CPU", 0, 0, None)
    if num_AMD_gpus != 0 and num_NVIDIA_gpus != 0:
        print('Autodetected both AMD and NVIDIA GPUs on this system - Aborting autodectection')
        return ("Generic AMD+NVIDIA", 0, 0, None)
    if num_AMD_gpus > 0:
        return ("Generic AMD", num_AMD_gpus, mem_per_AMD_gpu, AMD_arch)
    else:
        return ("Generic NVIDIA", num_NVIDIA_gpus, mem_per_NVIDIA_gpu, NVIDIA_arch)


def count_cpus():
    try:
        import psutil
        num_cpus = psutil.cpu_count(logical=False)
        return num_cpus
    except (ImportError, ModuleNotFoundError):
        return 0

def num_NUMA_domains():
    """
    Get the number of NUMA domains on a Linux system, filtering out nodes without CPUs.

    Returns:
        int: The number of NUMA domains with CPUs attached.
    Errors:
        If an error is detected just return that there is 1 NUMA domain
    """
    numa_nodes_path = "/sys/devices/system/node/"
    try:
        # List all entries in the NUMA nodes directory
        entries = os.listdir(numa_nodes_path)

        # Filter entries that match the pattern "nodeX" where X is a number
        numa_nodes = [entry for entry in entries if entry.startswith("node") and entry[4:].isdigit()]

        # Check if each NUMA node has CPUs attached
        nodes_with_cpus = 0
        for node in numa_nodes:
            cpulist_path = os.path.join(numa_nodes_path, node, "cpulist")
            if os.path.exists(cpulist_path):
                with open(cpulist_path, "r") as cpulist_file:
                    cpulist = cpulist_file.read().strip()
                    if cpulist:  # If the cpulist is not empty, the node has CPUs
                        nodes_with_cpus += 1

        return nodes_with_cpus
    except FileNotFoundError:
        # The path does not exist, likely indicating NUMA is not supported
        return 1
    except Exception as e:
        # Handle unexpected errors
        print(f"Error while determining NUMA domains: {e}")
        return 1

def find_scheduler():
    import shutil
    import os
    scheduler = None
    if shutil.which("flux") and os.path.exists("/run/flux/local"):
        scheduler = "flux"
    elif shutil.which("jsrun"):
        scheduler = "lsf"
    elif shutil.which("srun"):
        scheduler = "slurm"

    return scheduler


def system():
    """Name of system.

    Hostname with trailing digits removed.

    """
    global _system
    if _system is None:
        _system = re.sub(r"\d+", "", socket.gethostname())
    return _system


def clear_autodetected_system():
    """
    Clears the autodetected system. Used for testing.
    """
    global _system
    _system = None


def autodetect_current_system(quiet: bool = False) -> System:
    """
    Tries to detect the current system based on information such
    as the hostname and HPC center.
    """

    sys = system()
    if sys in ("tioga", "tuolumne", "elcap", "rzadams", "tenaya"):
        return ElCapitan(sys)

    if sys == "ipa":
        return CTS2(sys)

    if sys in ("lassen", "sierra", "rzanzel"):
        return Sierra(sys)

    # Try to find current system via other means
    (generic_name, num_gpus, mem_per_gpu, gpu_arch) = find_gpus()
    num_cpus = count_cpus()
    scheduler = find_scheduler()
    generic_sys = GenericSystem()
    autodetected_system_params = SystemParams(
                cores_per_node=num_cpus,
                gpus_per_node=num_gpus,
                gpu_arch=gpu_arch,
                mem_per_gpu=mem_per_gpu,
                scheduler=scheduler,
                numa_domains=num_NUMA_domains())
    generic_sys.system_params = {"auto": autodetected_system_params}
    generic_sys.default_queue = "auto"
    generic_sys.system_name = generic_name

    if not quiet:
        logger.warning(
            "Could not auto-detect current system, defaulting " f"to {generic_name} system: "
            f'{autodetected_system_params}'
        )
    return generic_sys
