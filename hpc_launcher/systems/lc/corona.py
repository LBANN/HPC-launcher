# Copyright (c) 2014-2026, Lawrence Livermore National Security, LLC.
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
from hpc_launcher.schedulers.flux import FluxScheduler
from hpc_launcher.systems.system import System, SystemParams
import os


# Corona (toss_4_x86_64_ib): AMD MI50 (gfx906), 48 CPU cores/node, 8 GPUs/node,
# 32 GiB VRAM/GPU. This system does not have a Slingshot network, so do not
# enable any Slingshot/Cassini RCCL-OFI tuning by default.
_mi50_node = SystemParams(48, 8, "gfx906", 32.0, 2, "flux")
_system_params = {
    "corona": (
        "pbatch",
        {
            "pbatch": _mi50_node,
            "pdebug": _mi50_node,
        },
    ),
}


class Corona(System):
    """
    LLNL LC system profile for Corona (AMD GPUs, non-Slingshot network).
    """

    def __init__(self, system_name: str):
        super().__init__(system_name, _system_params)

    def environment_variables(self) -> list[tuple[str, str]]:
        env_list = []

        # ROCm/RCCL tuning that is not network-fabric specific
        env_list.append(("NCCL_MIN_NCHANNELS", "24"))

        # MIOpen cache locations (avoid home filesystem contention)
        env_list.append(("MIOPEN_DEBUG_DISABLE_FIND_DB", "0"))
        env_list.append(("MIOPEN_DISABLE_CACHE", "0"))
        tmpdir = os.environ.get("TMPDIR")
        if tmpdir:
            env_list.append(("MIOPEN_USER_DB_PATH", f"{tmpdir}/MIOpen_user_db"))
            env_list.append(("MIOPEN_CUSTOM_CACHE_DIR", f"{tmpdir}/MIOpen_custom_cache"))

        # If running on a Cray environment, preserve CRAY_LD_LIBRARY_PATH.
        if os.getenv("CRAY_LD_LIBRARY_PATH") is not None:
            env_list.append(
                (
                    "LD_LIBRARY_PATH",
                    os.getenv("CRAY_LD_LIBRARY_PATH") + ":${LD_LIBRARY_PATH}",
                )
            )

        # Ensure ROCm LLVM libs are visible if ROCM_PATH is set.
        if os.getenv("ROCM_PATH") is not None:
            rocm_path = os.getenv("ROCM_PATH")
            env_list.append(
                (
                    "LD_LIBRARY_PATH",
                    os.path.join(f"{rocm_path}", "llvm", "lib") + ":${LD_LIBRARY_PATH}",
                )
            )

        # Allow user override to add a specific OFI plugin path, even though
        # Corona is expected not to need RCCL-OFI by default.
        different_ofi_plugin = os.getenv("LBANN_USE_THIS_OFI_PLUGIN")
        if different_ofi_plugin is not None:
            env_list.append(
                ("LD_LIBRARY_PATH", different_ofi_plugin + ":${LD_LIBRARY_PATH}")
            )

        for i in self._aux_env_list:
            env_list.append(i)

        return env_list

    def customize_scheduler(self, scheduler):
        use_this_rccl = os.getenv("LBANN_USE_THIS_RCCL")
        if isinstance(scheduler, FluxScheduler):
            scheduler.common_launch_args["--exclusive"] = None

        if use_this_rccl is not None:
            scheduler.ld_preloads = [f"{use_this_rccl}"]

        return

    @property
    def preferred_scheduler(self) -> type[Scheduler]:
        return FluxScheduler
