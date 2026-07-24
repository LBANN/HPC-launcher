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
                                                  OrderedDict([("-oremovable", "off"),
                                                               ("~-oremovable", None)]),
                                                  OrderedDict([("-ofastload", "off"),
                                                               ("-ompibind", "off")]),
                                                  OrderedDict([("-ofastload", "off"),
                                                               ("-ompibind", "off"),
                                                               ("-oremovable", "off"),
                                                               ("~-oremovable", None)])))
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
    scheduler.override_launch_args = override_launch_args

    cmd = scheduler.launch_command(system, blocking, cli_env_only)
    assert len(override_launch_args.items()) > 0

    # Replay the overrides in the same order the scheduler applies them, so
    # that a flag which is added and then removed within the same
    # ``override_launch_args`` (the ``~`` case) is only asserted absent, not
    # also asserted present from its earlier "add" entry.
    expected_present: "OrderedDict[str, str]" = OrderedDict()
    expected_absent: set = set()
    for k, v in override_launch_args.items():
        if "~" in k:
            k = k.replace("~", "")
            expected_present.pop(k, None)
            expected_absent.add(k)
        else:
            expected_present[k] = v
            expected_absent.discard(k)

    for k, v in expected_present.items():
        if not v:
            assert f"{k}" in cmd
        else:
            assert f"{k}={v}" in cmd

    for k in expected_absent:
        assert f"{k}" not in cmd
        assert not any(c.startswith(f"{k}=") for c in cmd)

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
                               OrderedDict([("-oremovable", "off"),
                                            ("~-oremovable", None)]), False)
    test_cli_argument_override(MagicMock(), MagicMock(), 2, 2, 1, False, "flux",
                               OrderedDict([("-ofastload", "off"),
                                            ("-ompibind", "off")]), False)
    test_cli_argument_override(MagicMock(), MagicMock(), 2, 2, 1, False, "slurm",
                               OrderedDict([("-ofastload", "off"),
                                            ("-ompibind", "off"),
                                            ("-oremovable", "off"),
                                            ("~-oremovable", None)]), False)
    test_cli_argument_override(MagicMock(), MagicMock(), 2, 2, 1, True, "slurm",
                               OrderedDict([("-ofastload", "off")]), False)
    test_cli_argument_override(MagicMock(), MagicMock(), 2, 2, 1, True, "lsf",
                               OrderedDict([("-ofastload", "off")]), False)


