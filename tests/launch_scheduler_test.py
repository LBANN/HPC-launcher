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
import pytest
from hpc_launcher.systems.configure import autodetect
from hpc_launcher.cli import common_args, launch_helpers
from unittest.mock import MagicMock, patch
from collections import OrderedDict

from hpc_launcher.systems.lc.el_capitan_family import ElCapitan
from hpc_launcher.schedulers import get_schedulers
from hpc_launcher.schedulers.flux import FluxScheduler
from hpc_launcher.schedulers.slurm import SlurmScheduler
from hpc_launcher.schedulers.lsf import LSFScheduler

import re

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
@patch(
    "hpc_launcher.systems.lc.el_capitan_family.ElCapitan.passthrough_environment_variables",
    return_value=[("foo", "bar"), ("baz", "deadbeef")],
)

@pytest.mark.parametrize("nodes", [1])
@pytest.mark.parametrize("procs_per_node", [2])
@pytest.mark.parametrize("gpus_per_proc", [1])
@pytest.mark.parametrize("blocking", [True, False])
@pytest.mark.parametrize("select_scheduler", ["slurm", "flux", "lsf"])

@pytest.mark.parametrize("override_launch_args", (OrderedDict([("-ofastload", "off")]),
                                                  OrderedDict([("-ompibind", "off")]),
                                                  OrderedDict([("~--exclusive", None)]),
                                                  OrderedDict([("-ofastload", "off"),
                                                               ("-ompibind", "off")]),
                                                  OrderedDict([("-ofastload", "off"),
                                                               ("-ompibind", "off"),
                                                               ("~--exclusive", None)])))
@pytest.mark.parametrize("cli_env_only", [True, False])
def test_cli_argument_override(sys: MagicMock, env: MagicMock, nodes, procs_per_node, gpus_per_proc, blocking, select_scheduler, override_launch_args:OrderedDict[str, str], cli_env_only, *xargs):
    system = autodetect.autodetect_current_system()
    scheduler_keys = get_schedulers()
    scheduler_class = scheduler_keys[select_scheduler]
    args:dict[str,str] = dict()
    args["nodes"] = nodes
    args["procs_per_node"] = procs_per_node
    args["gpus_per_proc"] = gpus_per_proc

    scheduler = scheduler_class(**args)
    # Reset class
    scheduler.submit_only_args.clear()
    scheduler.run_only_args.clear()
    scheduler.common_launch_args.clear()
    scheduler.override_launch_args = None
    scheduler.override_launch_args = override_launch_args

    cmd = scheduler.launch_command(system, blocking, cli_env_only)
    assert len(override_launch_args.items()) > 0
    for k,v in override_launch_args.items():
        if "~" in k:
            k = k.replace("~", "")
            if not v:
                if f"{k}" in cmd:
                    assert not f"{k}" in cmd
                if f"{k}={v}" in cmd:
                    assert not f"{k}={v}" in cmd
        else:
            if not v:
                assert f"{k}" in cmd
            if f"{k}={v}" in cmd:
                assert f"{k}={v}" in cmd

    if type(scheduler) is SlurmScheduler and blocking:
        for c in cmd:
            if c.startswith("--export"):
                pattern = r'--export=ALL,.*foo=bar,baz=deadbeef'
                assert re.search(pattern, c)
    if type(scheduler) is FluxScheduler and blocking:
        assert '--env=foo=bar' in cmd
        assert '--env=baz=deadbeef' in cmd
    if type(scheduler) is LSFScheduler and blocking:
        for c in cmd:
            if c.startswith("--env"):
                pattern = r'--env "ALL,.*foo=bar, baz=deadbeef"'
                assert re.search(pattern, c)

    print(f"Overriden command line: {cmd}")

if __name__ == "__main__":
    test_cli_argument_override(MagicMock(), MagicMock(), 2, 2, 1, False, "slurm",
                               OrderedDict([("-ofastload", "off")]), False)
    test_cli_argument_override(MagicMock(), MagicMock(), 2, 2, 1, False, "flux",
                               OrderedDict([("-ompibind", "off")]), False)
    test_cli_argument_override(MagicMock(), MagicMock(), 2, 2, 1, False, "slurm",
                               OrderedDict([("~--exclusive", None)]), False)
    test_cli_argument_override(MagicMock(), MagicMock(), 2, 2, 1, False, "flux",
                               OrderedDict([("-ofastload", "off"),
                                            ("-ompibind", "off")]), False)
    test_cli_argument_override(MagicMock(), MagicMock(), 2, 2, 1, False, "slurm",
                               OrderedDict([("-ofastload", "off"),
                                            ("-ompibind", "off"),
                                            ("~--exclusive", None)]), False)
    test_cli_argument_override(MagicMock(), MagicMock(), 2, 2, 1, True, "slurm",
                               OrderedDict([("-ofastload", "off")]), False)
    test_cli_argument_override(MagicMock(), MagicMock(), 2, 2, 1, True, "lsf",
                               OrderedDict([("-ofastload", "off")]), False)


