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
"""
Tier A tests for env plumbing (review finding E6):

- E6: the rendezvous port was hardcoded to ``23456`` in every scheduler, so
  two jobs whose rank-0 node coincided collided on one TCPStore. It is now
  chosen once per launch (a free ephemeral port on the launch host, with a
  UUID-hash fallback) and baked into ``TORCHRUN_HPC_MASTER_PORT``.

Pure Tier A: no torch, no scheduler binaries. Schedulers are constructed
directly.
"""
import types

from hpc_launcher.schedulers import scheduler as scheduler_mod
from hpc_launcher.schedulers.flux import FluxScheduler
from hpc_launcher.schedulers.local import LocalScheduler


def _master_port(env_list):
    """Return the int ``TORCHRUN_HPC_MASTER_PORT`` value from an env list."""
    for e in env_list:
        if len(e) >= 2 and e[0] == "TORCHRUN_HPC_MASTER_PORT":
            return int(e[1])
    raise AssertionError(f"TORCHRUN_HPC_MASTER_PORT not found in {env_list}")


def _flux():
    return FluxScheduler(nodes=2, procs_per_node=2, gpus_per_proc=0)


def test_rendezvous_port_unique_per_launch():
    """
    Distinct launches (distinct scheduler instances) each pick their own
    in-range port, while all env entries of a single launch agree on one
    port that is stable across regenerations. Exercises the real port picker
    (the ephemeral-socket path where available, the hash fallback otherwise).
    """
    schedulers = [_flux() for _ in range(4)]
    ports = [_master_port(s.setup_rendezvous_protocol("tcp")) for s in schedulers]

    # Every value is a real, in-range TCP port.
    for p in ports:
        assert 1024 <= p <= 65535, f"port {p} out of range"

    # Independent per-launch selection: launches do not all share one port.
    assert len(set(ports)) > 1, f"all launches picked the same port: {ports}"

    # Within one launch the port is cached and stable across regenerations,
    # so every node of that job reads the same value.
    s = schedulers[0]
    first = _master_port(s.setup_rendezvous_protocol("tcp"))
    second = _master_port(s.setup_rendezvous_protocol("tcp"))
    assert first == second == ports[0]
    assert s.rendezvous_port() == ports[0]


def test_rendezvous_port_cached_once_per_launch():
    """
    The port picker runs at most once per scheduler instance; every later
    consumer of the port reads the cached value.
    """
    calls = {"n": 0}
    real = scheduler_mod.pick_rendezvous_port

    def counting():
        calls["n"] += 1
        return real()

    s = _flux()
    original = scheduler_mod.pick_rendezvous_port
    scheduler_mod.pick_rendezvous_port = counting
    try:
        p1 = s.rendezvous_port()
        p2 = s.rendezvous_port()
        p3 = _master_port(s.setup_rendezvous_protocol("tcp"))
    finally:
        scheduler_mod.pick_rendezvous_port = original

    assert p1 == p2 == p3
    assert calls["n"] == 1, "the port picker must run at most once per launch"


def test_rendezvous_port_fallback_in_range(monkeypatch):
    """
    If binding an ephemeral socket fails (e.g. a restrictive sandbox), the
    UUID-hash fallback must still yield an in-range high port.
    """

    def _no_socket(*args, **kwargs):
        raise OSError("sockets disabled for this test")

    fake_socket = types.SimpleNamespace(
        AF_INET=0, SOCK_STREAM=0, socket=_no_socket
    )
    monkeypatch.setattr(scheduler_mod, "socket", fake_socket)

    seen = set()
    for _ in range(200):
        port = scheduler_mod.pick_rendezvous_port()
        assert (
            scheduler_mod._RENDEZVOUS_PORT_FALLBACK_MIN
            <= port
            <= scheduler_mod._RENDEZVOUS_PORT_FALLBACK_MAX
        ), f"fallback port {port} out of range"
        seen.add(port)

    # The fallback draws from a fresh UUID each time, so it varies.
    assert len(seen) > 1


def test_local_scheduler_bakes_real_rendezvous_port():
    """
    The local scheduler's rendezvous env goes through the same per-instance
    port helper, so it bakes a real in-range port rather than a literal.
    """
    scheduler = LocalScheduler(nodes=1, procs_per_node=1, gpus_per_proc=0)
    env_list = scheduler.setup_rendezvous_protocol("tcp")
    port = _master_port(env_list)
    assert 1024 <= port <= 65535
    assert port != 23456 or scheduler.rendezvous_port() == port
