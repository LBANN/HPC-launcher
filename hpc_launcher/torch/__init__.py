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
from psutil import Process
import inspect
import os

# Optionally explicitly override HIP_VISIBLE_DEVICES with ROCR_VISIBLE_DEVICES before torch is imported
# ROCR_VISIBLE_DEVICES - controls which GPUs a process can see
# HIP_VISIBLE_DEVICES - controls which GPUs a process can use
# PyTorch - behaves better when it can see all of the devices on a node, but knows which it should use
unswap_rocr_hip_vis_dev_env = os.getenv("TORCHRUN_HPC_UNSWAP_ROCR_HIP_VIS_DEV", "False")
# Convert to boolean
unswap_rocr_hip_vis_dev = False
if unswap_rocr_hip_vis_dev_env.lower() in ("true", "1", "yes", "on"):
    unswap_rocr_hip_vis_dev = True

if os.getenv("ROCR_VISIBLE_DEVICES") and not unswap_rocr_hip_vis_dev:
    if os.getenv("HIP_VISIBLE_DEVICES"):
        print(f'WARNING: overwriting HIP_VISIBLE_DEVICES {os.getenv("HIP_VISIBLE_DEVICES")} with ROCR_VISIBLE_DEVICES {os.getenv("ROCR_VISIBLE_DEVICES")}')

    os.environ["HIP_VISIBLE_DEVICES"] = os.getenv("ROCR_VISIBLE_DEVICES")
    del os.environ["ROCR_VISIBLE_DEVICES"]

affinity = None
if hasattr(Process, "cpu_affinity") and inspect.isfunction(Process.cpu_affinity):
    # Save affinity before importing torch
    affinity = Process().cpu_affinity()

import torch

if affinity is not None:
    # Restore affinity after importing torch
    Process().cpu_affinity(affinity)
import os

if torch.cuda.is_available():
    fraction_max_gpu_mem = float(os.getenv("HPC_LAUNCHER_MAX_GPU_MEM", 1.0))
    if fraction_max_gpu_mem != 1.0:
        torch.cuda.set_per_process_memory_fraction(fraction_max_gpu_mem)
