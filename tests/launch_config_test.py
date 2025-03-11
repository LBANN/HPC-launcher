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
from hpc_launcher.systems.system import System, SystemParams
from hpc_launcher.schedulers.scheduler import Scheduler
from hpc_launcher.systems.configure import configure_launch
from unittest.mock import patch


class MockScheduler(Scheduler):
    pass


# Mock system for testing
class MockSystem(System):

    def __init__(self):
        super().__init__("mock")
        self.default_queue = "mockq"
        self.system_params = {
            "mockq": SystemParams(
                cores_per_node=24,
                gpus_per_node=3,
                gpu_arch="sm_00",
                mem_per_gpu=11,
                scheduler="MockScheduler",
                numa_domains=3,
            ),
            "nondefault": SystemParams(
                cores_per_node=1,
                gpus_per_node=2,
                gpu_arch="gfx000",
                mem_per_gpu=4,
                scheduler="MockScheduler",
                numa_domains=1,
            ),
            "cpuonly": SystemParams(
                cores_per_node=24,
                gpus_per_node=0,
                gpu_arch=None,
                mem_per_gpu=0,
                scheduler="MockScheduler",
                numa_domains=4,
            ),
        }

    def environment_variables(self) -> list[tuple[str, str]]:
        return [("IS_MOCK", "1")]

    @property
    def preferred_scheduler(self) -> type[Scheduler]:
        return MockScheduler


@patch(
    "hpc_launcher.systems.autodetect.autodetect_current_system",
    return_value=MockSystem(),
)
def test_launch_config(*args):
    """
    Tests various launch configurations for GPU count and memory size.
    """
    # User-specified procs_per_node
    system, nodes, procs_per_node, gpus_per_proc = configure_launch(None, 2, 4, 1, 0, 0, None)
    assert isinstance(system, MockSystem)
    assert nodes == 2
    assert procs_per_node == 4
    assert gpus_per_proc == 1

    # GPU count constraint test
    system, nodes, procs_per_node, gpus_per_proc = configure_launch(None, 0, 0, 1, 6, 0, None)
    assert isinstance(system, MockSystem)
    assert nodes == 2
    assert procs_per_node == 3
    assert gpus_per_proc == 1

    # Memory constraint test
    system, nodes, procs_per_node, gpus_per_proc = configure_launch(None, 0, 0, 1, 0, 22, None)
    assert isinstance(system, MockSystem)
    assert nodes == 1
    assert procs_per_node == 2
    assert gpus_per_proc == 1

    # Just above the memory limit of a single node, this triggers a switch to all gpus per node
    system, nodes, procs_per_node, gpus_per_proc = configure_launch(None, 0, 0, 1, 0, 34, None)
    assert isinstance(system, MockSystem)
    assert nodes == 2
    assert procs_per_node == 3
    assert gpus_per_proc == 1

    # Ask for too many GPUs per proc, this should snap down to the 3 GPUs available
    system, nodes, procs_per_node, gpus_per_proc = configure_launch(None, 2, 1, 4, 0, 0, None)
    assert isinstance(system, MockSystem)
    assert nodes == 2
    assert procs_per_node == 1
    assert gpus_per_proc == 3

    system, nodes, procs_per_node, gpus_per_proc = configure_launch(None, 2, 2, 2, 0, 0, None)
    assert isinstance(system, MockSystem)
    assert nodes == 2
    assert procs_per_node == 2
    assert gpus_per_proc == 2

    system, nodes, procs_per_node, gpus_per_proc = configure_launch(None, 2, 2, None, 0, 0, None)
    assert isinstance(system, MockSystem)
    assert nodes == 2
    assert procs_per_node == 2
    assert gpus_per_proc == 1


@patch(
    "hpc_launcher.systems.autodetect.autodetect_current_system",
    return_value=MockSystem(),
)
def test_nondefault_queue(*args):
    """
    Tests the configuration of a non-default queue.
    """
    system, nodes, procs_per_node, gpus_per_proc = configure_launch("nondefault", 1, 2, 1, 0, 0, None)
    assert isinstance(system, MockSystem)
    assert nodes == 1
    assert procs_per_node == 2
    assert gpus_per_proc == 1

    # Memory constraint test
    system, nodes, procs_per_node, gpus_per_proc = configure_launch("nondefault", 0, 0, 1, 0, 22, None)
    assert isinstance(system, MockSystem)
    assert nodes == 3
    assert procs_per_node == 2
    assert gpus_per_proc == 1


@patch(
    "hpc_launcher.systems.autodetect.autodetect_current_system",
    return_value=MockSystem(),
)
def test_preferred_procs_per_node(*args):
    """
    Tests the configuration of the preferred number of processes per node.
    """

    # User specifies only number of nodes (GPU system)
    system, nodes, procs_per_node, gpus_per_proc = configure_launch(None, 3, 0, 1, 0, 0, None)
    assert isinstance(system, MockSystem)
    assert nodes == 3
    assert procs_per_node == 3
    assert gpus_per_proc == 1

    # User specifies only number of nodes (CPU system)
    system, nodes, procs_per_node, gpus_per_proc = configure_launch("cpuonly", 3, 0, 0, 0, 0, None)
    assert isinstance(system, MockSystem)
    assert nodes == 3
    assert procs_per_node == 4
    assert gpus_per_proc == 0


@patch(
    "hpc_launcher.systems.autodetect.autodetect_current_system",
    return_value=MockSystem(),
)
def test_environment_variables(*args):
    """
    Tests the configuration of environment variables.
    """
    system = MockSystem()
    env_vars = system.environment_variables()
    assert env_vars == [("IS_MOCK", "1")]
    env_vars = system.passthrough_environment_variables()
    assert env_vars == []


if __name__ == "__main__":
    test_launch_config()
    test_nondefault_queue()
    test_preferred_procs_per_node()
    test_environment_variables()
