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
"""
Regression tests for the Sierra family's CPU-affinity binding.

``Sierra.customize_scheduler()`` computed the jsrun ``--bind=packed:N``
value from a hardcoded local ``procs_per_node = 2``, instead of the live
``scheduler.procs_per_node`` that ``LSFScheduler`` itself reads one line
away (in ``build_scheduler_specific_arguments``) for ``--tasks_per_rs``.
Sierra's own ``SystemParams`` (``_sierra_node``, ``gpus_per_node=4``) makes
``SystemParams.procs_per_node()`` -- the value ``configure_launch`` fills in
when the user does not pass ``--procs-per-node`` -- return 4, so every
*default* Sierra job got a mismatched ``--tasks_per_rs=4`` next to
``--bind=packed:16``: the binding scheme's own arithmetic for 4 processes
across 2 sockets gives ``packed:8``, so with ``packed:16`` two processes
sharing a socket are each told to claim all 16 cores instead of
partitioning it.

These tests exercise ``customize_scheduler`` through the real
``launcher_script`` pipeline (no jsrun/bsub actually runs; the sandbox
cannot run LSF) and assert on the internal jsrun line emitted into the
generated batch script -- the same pattern used by
``tests/lsf_scheduler_test.py``.
"""
import pytest

from hpc_launcher.schedulers.lsf import LSFScheduler
from hpc_launcher.systems.lc.sierra_family import Sierra, _sierra_node


def _make_scheduler(procs_per_node: int) -> LSFScheduler:
    return LSFScheduler(
        nodes=1,
        procs_per_node=procs_per_node,
        gpus_per_proc=1,
    )


def _jsrun_line(system, scheduler, monkeypatch, tmp_path) -> str:
    """
    Render the non-blocking batch script (which always folds run_only_args
    into the internal jsrun line, unlike the blocking path, which only does
    so inside an existing LSF allocation) and return that single jsrun line.
    """
    monkeypatch.delenv("LSB_HOSTS", raising=False)
    script = scheduler.launcher_script(
        system, "hostname", [], blocking=False, launch_dir=str(tmp_path)
    )
    jsrun_lines = [
        line
        for line in script.splitlines()
        if "jsrun" in line and not line.lstrip().startswith("#")
    ]
    assert len(jsrun_lines) == 1, (
        f"expected exactly one internal jsrun line, found {len(jsrun_lines)}:\n{script}"
    )
    return jsrun_lines[0]


def test_default_procs_per_node_is_four():
    """
    Sanity check on the premise: Sierra's own SystemParams makes the
    *default* procs_per_node (what ``configure_launch`` fills in when
    ``--procs-per-node`` is not given) 4, not the 2 that
    ``customize_scheduler`` used to hardcode.
    """
    assert _sierra_node.gpus_per_node == 4
    assert _sierra_node.procs_per_node() == 4


def test_bind_packed_matches_default_procs_per_node(monkeypatch, tmp_path):
    """
    The reproducer: a default (no ``--procs-per-node`` override) Sierra
    job requests 4 procs/node. The jsrun line must bind
    ``packed:8`` (16 cores/socket over 2 procs/socket) -- matching
    ``--tasks_per_rs=4`` -- not the old hardcoded ``packed:16``, which
    leaves two same-socket processes both claiming all 16 cores.
    """
    system = Sierra("lassen")
    scheduler = _make_scheduler(procs_per_node=4)

    line = _jsrun_line(system, scheduler, monkeypatch, tmp_path)

    assert "--tasks_per_rs=4" in line
    assert "--bind=packed:8" in line
    assert "--bind=packed:16" not in line


def test_bind_packed_derives_from_arbitrary_procs_per_node(monkeypatch, tmp_path):
    """
    A non-default ``--procs-per-node`` override (6, chosen to differ from
    both the old hardcoded 2 and the default 4) must also be honored, to
    prove the binding is genuinely derived from the live scheduler value
    rather than matching one specific case by coincidence.
    """
    system = Sierra("lassen")
    scheduler = _make_scheduler(procs_per_node=6)

    line = _jsrun_line(system, scheduler, monkeypatch, tmp_path)

    assert "--tasks_per_rs=6" in line
    # procs_per_socket = (6 + 1) // 2 = 3; cores_per_proc = 16 // 3 = 5
    assert "--bind=packed:5" in line


@pytest.mark.parametrize("procs_per_node", [33, 64])
def test_bind_packed_never_reaches_zero(procs_per_node, monkeypatch, tmp_path):
    """
    Deriving the binding from a live value rather than a constant introduces
    a division that the old hardcoded ``procs_per_node = 2`` never exercised:
    past 32 procs/node, ``16 // procs_per_socket`` floors to 0 and the job
    would ask jsrun to bind each process to no cores at all.

    Such a request is already oversubscribed -- 33 processes do not fit a
    2x16-core model -- but the right response is the smallest real binding,
    not an unsatisfiable one. The old code could not produce this because it
    always divided the same two constants.
    """
    system = Sierra("lassen")
    scheduler = _make_scheduler(procs_per_node=procs_per_node)

    line = _jsrun_line(system, scheduler, monkeypatch, tmp_path)

    assert "--bind=packed:0" not in line
    assert "--bind=packed:1" in line


def test_smpiargs_still_set(monkeypatch, tmp_path):
    """The unrelated --smpiargs customization must survive untouched."""
    system = Sierra("lassen")
    scheduler = _make_scheduler(procs_per_node=4)

    line = _jsrun_line(system, scheduler, monkeypatch, tmp_path)

    assert "--smpiargs=-gpu" in line


def test_smpiargs_value_carries_no_literal_quotes(monkeypatch, tmp_path):
    """
    The value must reach jsrun as ``-gpu``, not ``"-gpu"``.

    The quotes one would type around this at a shell prompt belong to the
    shell, and neither consumer has one to strip them. The script path
    shlex-quotes the value, turning an embedded pair into ``'"-gpu"'`` -- so
    the shell that runs the script hands jsrun the quotes rather than
    removing them. The argv path below execs jsrun with no shell at all, so
    whatever is in the string is what the option gets.
    """
    system = Sierra("lassen")
    scheduler = _make_scheduler(procs_per_node=4)

    line = _jsrun_line(system, scheduler, monkeypatch, tmp_path)
    assert '"' not in line.split("--smpiargs=")[1].split()[0]

    # The same value on the in-allocation argv path, where it is exec'd.
    monkeypatch.setenv("LSB_HOSTS", "host1 host2")
    argv_scheduler = _make_scheduler(procs_per_node=4)
    argv = argv_scheduler.launch_command(
        Sierra("lassen"), blocking=True, cli_env_only=True
    )

    smpiargs = [t for t in argv if t.startswith("--smpiargs")]
    assert smpiargs == ["--smpiargs=-gpu"], smpiargs
