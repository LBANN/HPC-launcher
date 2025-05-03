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

@pytest.mark.parametrize("nodes", [1])
@pytest.mark.parametrize("procs_per_node", [2])
@pytest.mark.parametrize("gpus_per_proc", [1])
@pytest.mark.parametrize("blocking", (True, False))
@pytest.mark.parametrize("override_launch_args", (OrderedDict([("-ofastload", "off")]),
                                                  OrderedDict([("-ompibind", "off")]),
                                                  OrderedDict([("~--exclusive", None)]),
                                                  OrderedDict([("-ofastload", "off"),
                                                               ("-ompibind", "off")]),
                                                  OrderedDict([("-ofastload", "off"),
                                                               ("-ompibind", "off"),
                                                               ("~--exclusive", None)])))
def test_cli_argument_override(_: MagicMock, nodes, procs_per_node, gpus_per_proc, blocking, override_launch_args:OrderedDict[str, str], *xargs):
    system = autodetect.autodetect_current_system()
    scheduler_class = system.preferred_scheduler
    args:dict[str,str] = dict()
    args["nodes"] = nodes
    args["procs_per_node"] = procs_per_node
    args["gpus_per_proc"] = gpus_per_proc

    scheduler = scheduler_class(**args)
    scheduler.override_launch_args = override_launch_args

    cmd = scheduler.launch_command(system, blocking)
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
    print(f"Overriden command line: {cmd}")

if __name__ == "__main__":
    test_cli_argument_override(MagicMock(), 2, 2, 1, False, OrderedDict([("-ofastload", "off")]))
    test_cli_argument_override(MagicMock(), 2, 2, 1, False, OrderedDict([("-ompibind", "off")]))
    test_cli_argument_override(MagicMock(), 2, 2, 1, False, OrderedDict([("~--exclusive", None)]))
    test_cli_argument_override(MagicMock(), 2, 2, 1, False, OrderedDict([("-ofastload", "off"),
                                                            ("-ompibind", "off")]))
    test_cli_argument_override(MagicMock(), 2, 2, 1, False, OrderedDict([("-ofastload", "off"),
                                                                         ("-ompibind", "off"),
                                                                         ("~--exclusive", None)]))


