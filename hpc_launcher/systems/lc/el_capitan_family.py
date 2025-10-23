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
from hpc_launcher.schedulers.flux import FluxScheduler
from hpc_launcher.systems.system import System, SystemParams
import os

import logging

logger = logging.getLogger(__name__)

# Known LC systems
_mi250x_node = SystemParams(64, 8, "gfx90a", 64.0, 4, "flux")
# APUs can run into a snarl where they OOM if too much GPU memory is allocated
_mi300a_node = SystemParams(96, 4, "gfx942", 128.0, 4, "flux", 0.8)
_system_params = {
    "tioga": (
        "pdebug",
        {
            "pdebug": _mi250x_node,
            "mi300a": _mi300a_node,
        },
    ),
    "tuolumne": (
        "pbatch",
        {
            "pbatch": _mi300a_node,
            "pdebug": _mi300a_node,
        },
    ),
    "elcap": (
        "pbatch",
        {
            "pbatch": _mi300a_node,
            "pdebug": _mi300a_node,
        },
    ),
    "rzadams": (
        "pbatch",
        {
            "pbatch": _mi300a_node,
            "pdebug": _mi300a_node,
        },
    ),
    "rzvernal": (
        "pdebug",
        {
            "pdebug": _mi250x_node,
            "pllm":   _mi250x_node,
        },
    ),
    "tenaya": (
        "pbatch",
        {
            "pbatch": _mi300a_node,
            "pdebug": _mi300a_node,
        },
    ),
}


class ElCapitan(System):
    """
    LLNL LC Systems based on the El Capitan MI300a architecture.
    """

    def __init__(self, system_name):
        super().__init__(system_name, _system_params)

    def environment_variables(self) -> list[tuple[str, str]]:
        env_list = []
        env_list.append(("NCCL_NET_GDR_LEVEL", "3"))  # From HPE to avoid hangs
        env_list.append(
            ("NCCL_MIN_NCHANNELS", "24")
        )  # From AMD to improve collective performance
        env_list.append(("MIOPEN_DEBUG_DISABLE_FIND_DB", "0"))
        env_list.append(("MIOPEN_DISABLE_CACHE", "0"))
        tmpdir = os.environ.get("TMPDIR")
        env_list.append(("MIOPEN_USER_DB_PATH", f"{tmpdir}/MIOpen_user_db"))
        env_list.append(("MIOPEN_CUSTOM_CACHE_DIR", f"{tmpdir}/MIOpen_custom_cache"))

        if os.getenv("CRAY_LD_LIBRARY_PATH") is not None:
            env_list.append(
                (
                    "LD_LIBRARY_PATH",
                    os.getenv("CRAY_LD_LIBRARY_PATH") + ":${LD_LIBRARY_PATH}",
                )
            )

        optimize_rccl_protocol = False
        optimize_comm_protocol = ""
        if self.job_comm_protocol:
            optimize_comm_protocol = self.job_comm_protocol
        if optimize_comm_protocol.upper() == "RCCL" or optimize_comm_protocol.upper() == "*CCL":
            optimize_rccl_protocol = True

        if os.getenv("ROCM_PATH") is not None:
            rocm_path = os.getenv("ROCM_PATH")
            env_list.append(
                (
                    "LD_LIBRARY_PATH",
                    os.path.join(f"{rocm_path}", "llvm", "lib")
                    + ":${LD_LIBRARY_PATH}",
                )
            )
            if optimize_rccl_protocol:
                rocm_ver = os.path.basename(rocm_path)
                # Check for and include the AWS_OFI_PLUGIN if it exists
                sys_type = os.getenv("SYS_TYPE")
                aws_ofi_plugin = f'/collab/usr/global/tools/rccl/{sys_type}/{rocm_ver}/install/lib'
                if os.path.isdir(aws_ofi_plugin):
                    logger.info(f"Setting path to default AWS_OFI_RCCL plugin {aws_ofi_plugin} to accelerate RCCL communication protocol.")
                    env_list.append(
                        (
                            "LD_LIBRARY_PATH",
                            aws_ofi_plugin
                            + ":${LD_LIBRARY_PATH}",
                        )
                    )
                else:
                    logger.warn(f"WARNING: using RCCL communication protocol and no default AWS_OFI_RCCL plugin was detected.  Checked {aws_ofi_plugin}. Ensure one is loaded or performance will be degraded.")

        different_ofi_plugin = os.getenv("LBANN_USE_THIS_OFI_PLUGIN")
        if different_ofi_plugin is not None:
            env_list.append(
                ("LD_LIBRARY_PATH", different_ofi_plugin + ":${LD_LIBRARY_PATH}")
            )

        if optimize_rccl_protocol:
            # Performance tuning for HPE Slingshot Cassini NIC (Audited on 3/31/25) - Only use with RCCL
            msg = "Performance tuning for RCCL + HPE Slingshot Cassini NIC (Audited on 3/31/25)"
            env_list.append((f"\n# {msg}",))
            env_list.append(("FI_CXI_RDZV_PROTO", "alt_read", msg))
            env_list.append(("FI_CXI_RDZV_THRESHOLD", "0", msg))
            env_list.append(("FI_CXI_RDZV_GET_MIN", "0", msg))
            env_list.append(("FI_CXI_RDZV_EAGER_SIZE", "0", msg))

        # Known issue with memhooks and RCCL hangs (Audited on 3/31/25)
        # https://support.hpe.com/hpesc/public/docDisplay?docId=dp00004854en_us&docLocale=en_US
        # env_list.append(("FI_MR_CACHE_MAX_COUNT", "0")) # MPI has a significant performance hit
        # kdreg2 will be the future
        env_list.append(("\n# Known issue with memhooks and RCCL hangs (Audited on 3/31/25)",))
        env_list.append(("# https://support.hpe.com/hpesc/public/docDisplay?docId=dp00004854en_us&docLocale=en_US",))
        msg = "Known issue with memhooks and RCCL hang (Audited on 3/31/25)"
        env_list.append(("FI_MR_CACHE_MONITOR", "userfaultfd", msg)) # This should work and be safe and performant
        msg = "Performance tuning for HPE Slingshot Cassini NIC (Audited on 3/31/25)"
        env_list.append(("FI_CXI_DEFAULT_TX_SIZE", "1024", msg))
        env_list.append(("FI_CXI_DISABLE_HOST_REGISTER", "1", msg))
        env_list.append(("FI_CXI_DEFAULT_CQ_SIZE", "131072", msg))
        # Run in hardware until the HW queues are exhausted, then fallback to SW
        env_list.append(("FI_CXI_RX_MATCH_MODE", "hybrid", msg)) # set to software instead when setting up the alt_read

        env_list.append(("\n# General tuning knobs (Audited on 3/31/25)",))
        # =2 may be a future performance improvement (Removes rails configuration)
        env_list.append(("NCCL_CROSS_NIC", "1"))
        # Improve the performance of large scale RCCL initialization - should only be used on wire-up
        env_list.append(("NCCL_SOCKET_IFNAME", "hsi0"))

        for i in self._aux_env_list:
            env_list.append(i)

        return env_list

    def customize_scheduler(self, scheduler):
        use_this_rccl = os.getenv("LBANN_USE_THIS_RCCL")
        if type(scheduler) is FluxScheduler:
            scheduler.common_launch_args["--exclusive"] = None # This is an alloc only on slurm and alloc or run on flux
            # Note that options cannot have a space after the -o flag, e.g. -o<option>
            # Performance tuning for HPE Slingshot Cassini NIC
            scheduler.common_launch_args["-ofastload"] = "on"
            scheduler.common_launch_args["--setattr=rdzv_get_en"] = "0"
            # Avoid bug in OMP that ruins the CPU_SET
            scheduler.common_launch_args["-ompibind"] = "omp_proc_bind,omp_places"

        # if type(scheduler) is SlurmScheduler:
        #     scheduler.submit_args["--exclusive"] = None # This is an alloc only on slurm and alloc or run on flux
            
        if use_this_rccl is not None:
            scheduler.ld_preloads = [f"{use_this_rccl}"]

        return

    @property
    def preferred_scheduler(self) -> type[Scheduler]:
        return FluxScheduler
