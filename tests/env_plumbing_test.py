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
Tests for env plumbing:

- The rendezvous port was hardcoded to ``23456`` in every scheduler, so
  two jobs whose rank-0 node coincided collided on one TCPStore. It is now
  chosen once per launch (a free ephemeral port on the launch host, with a
  UUID-hash fallback) and baked into ``TORCHRUN_HPC_MASTER_PORT``.
- On the ephemeral blocking path env vars are moved onto the scheduler
  CLI (flux ``--env=``), where no shell interprets them -- literal quotes
  survived, duplicate keys collapsed, and ``${VAR}`` never expanded. The env
  list is now expanded/merged/dequoted in-process exactly as the shell-script
  path would, and the fully expanded values become the CLI env args.
- Whenever the launcher may not write the launch script (an immutable
  user-supplied ``--batch-script``), the environment has to travel on the
  scheduler CLI instead. That redirection used to be gated on the run also
  being blocking, so ``--batch-script`` + ``--bg`` delivered the environment
  through neither channel.
- The generated script puts the invocation directory on ``PYTHONPATH`` so the
  user's own modules stay importable from the launch directory. It used to
  compute that as ``dirname(launch_dir)``, which is the invocation directory
  only for the auto-generated layout -- an absolute ``-l`` exported the launch
  directory's *parent* instead.

No torch or scheduler binaries needed. Schedulers and a
``GenericSystem`` stub are constructed directly; ``os.environ`` is
monkeypatched where the expansion source matters.
"""
import os
import shlex
import types

import pytest

from hpc_launcher.schedulers import scheduler as scheduler_mod
from hpc_launcher.schedulers.flux import FluxScheduler
from hpc_launcher.schedulers.slurm import SlurmScheduler
from hpc_launcher.schedulers.local import LocalScheduler
from hpc_launcher.systems.system import GenericSystem


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


# ---------------------------------------------------------------------------
# Ephemeral CLI env expansion
# ---------------------------------------------------------------------------


def test_cli_env_no_literal_quotes():
    """
    A double-quoted env value (the old
    ``NCCL_NET='"AWS Libfabric"'`` shape) must not reach the flux command as
    an argv token containing a literal ``"`` -- the shell would have removed
    those quotes, and so must the in-process expansion.
    """
    system = GenericSystem()
    system.extend_environment_variables([("NCCL_NET", '"AWS Libfabric"')])
    scheduler = FluxScheduler(nodes=1, procs_per_node=4, gpus_per_proc=1)

    cmd = scheduler.launch_command(system, blocking=True, cli_env_only=True)

    assert not any('"' in token for token in cmd), (
        f"a literal double-quote survived into the flux argv: {cmd}"
    )
    # The value is still present, dequoted and intact (space preserved).
    assert any("NCCL_NET=AWS Libfabric" in token for token in cmd), cmd


def test_cli_env_preserves_duplicate_ld_library_path(monkeypatch):
    """
    Two ``LD_LIBRARY_PATH`` entries must merge the way a sequence of
    ``export`` statements would -- the later entry incorporating the earlier
    -- rather than the second silently dropping the first. Order must match
    the shell (``B:A:<original>``).
    """
    monkeypatch.setenv("LD_LIBRARY_PATH", "/orig")
    scheduler = _flux()

    scheduler.cli_env_arg(
        [
            ("LD_LIBRARY_PATH", "A:${LD_LIBRARY_PATH}"),
            ("LD_LIBRARY_PATH", "B:${LD_LIBRARY_PATH}"),
        ]
    )

    env_keys = [k for k in scheduler.submit_only_args if k.startswith("--env=")]
    assert env_keys == ["--env=LD_LIBRARY_PATH"], (
        f"duplicate keys should collapse to one entry, got {env_keys}"
    )
    value = scheduler.submit_only_args["--env=LD_LIBRARY_PATH"]
    assert value == "B:A:/orig", value
    # Both components survive, in shell-equivalent order.
    assert "A" in value and "B" in value
    assert value.index("B") < value.index("A")


def test_cli_env_expands_var_references(monkeypatch):
    """
    ``${VAR}`` / ``$VAR`` references in an env value arrive expanded to
    the real value (there is no shell on the CLI path to do it later).
    """
    monkeypatch.setenv("HOME", "/home/tester")
    scheduler = _flux()

    scheduler.cli_env_arg(
        [
            ("FOO", "${HOME}/x"),
            ("BAR", "$HOME-bar"),
            ("BAZ", "no-refs-here"),
        ]
    )

    assert scheduler.submit_only_args["--env=FOO"] == "/home/tester/x"
    assert scheduler.submit_only_args["--env=BAR"] == "/home/tester-bar"
    assert scheduler.submit_only_args["--env=BAZ"] == "no-refs-here"


def test_cli_env_unknown_var_expands_empty():
    """Unknown references expand to the empty string, as ``sh`` does."""
    overlay = scheduler_mod.Scheduler.expand_cli_env(
        [("FOO", "x-${DEFINITELY_NOT_SET_12345}-y")]
    )
    assert overlay["FOO"] == "x--y"


def test_cli_env_expansion_applies_to_slurm_export(monkeypatch):
    """
    The expansion is a shared base-class hook, so the slurm ephemeral path
    (``--export=ALL,...``) benefits from it too, not just flux.
    """
    monkeypatch.setenv("HOME", "/home/tester")
    scheduler = SlurmScheduler(nodes=1, procs_per_node=1, gpus_per_proc=0)

    scheduler.cli_env_arg([("FOO", "${HOME}/lib"), ("NCCL_NET", '"AWS Libfabric"')])

    export = scheduler.submit_only_args["--export"]
    assert export.startswith("ALL,")
    assert "FOO=/home/tester/lib" in export
    # Dequoted, so no stray double-quotes in the value.
    assert 'NCCL_NET="AWS Libfabric"' not in export
    assert "NCCL_NET=AWS Libfabric" in export


# ---------------------------------------------------------------------------
# Immutable batch script: the environment must ride the scheduler CLI
# ---------------------------------------------------------------------------

# One representative member of the system tuning block the launcher injects.
# This particular variable is load-bearing on El Capitan -- its in-repo comment
# reads "Known issue with memhooks and RCCL hang" -- which is why silently
# dropping the block is a correctness problem and not a cosmetic one.
_TUNING_ENV = ("FI_MR_CACHE_MONITOR", "userfaultfd")


def _system_with_env():
    """A stub system whose ``environment_variables()`` is exactly one entry."""
    system = GenericSystem()
    system.extend_environment_variables([_TUNING_ENV])
    return system


def _cli_env_tokens(cmd: list[str]) -> list[str]:
    """Every argv token on ``cmd`` that carries environment (slurm or flux)."""
    return [t for t in cmd if t.startswith("--export") or t.startswith("--env=")]


@pytest.mark.parametrize("scheduler_class", [SlurmScheduler, FluxScheduler])
@pytest.mark.parametrize("blocking", [True, False])
def test_immutable_script_env_rides_the_cli(scheduler_class, blocking):
    """
    ``cli_env_only`` means "the launcher will not be writing the launch
    script, so the environment has no second channel" -- it is set for an
    ephemeral run *and* for a user-supplied ``--batch-script``, which the
    launcher copies verbatim and must not modify.

    The redirection onto the scheduler CLI used to also require ``blocking``,
    so ``--batch-script foo.sh --bg`` fell through to the ``else`` branch,
    which writes ``export`` lines into a header buffer that ``launch_command``
    discards -- and no script is written either. The entire launcher-injected
    tuning block vanished with no warning, only when ``--bg`` was added.

    Whether the submission blocks has nothing to do with where the
    environment has to travel, so both values must deliver it.
    """
    system = _system_with_env()
    scheduler = scheduler_class(nodes=1, procs_per_node=4, gpus_per_proc=0)

    cmd = scheduler.launch_command(system, blocking=blocking, cli_env_only=True)

    env_tokens = _cli_env_tokens(cmd)
    assert env_tokens, (
        f"no environment reached the {scheduler_class.__name__} command line "
        f"(blocking={blocking}); the launcher-injected tuning block was "
        f"silently dropped: {cmd}"
    )
    assert any(
        f"{_TUNING_ENV[0]}={_TUNING_ENV[1]}" in t for t in env_tokens
    ), f"{_TUNING_ENV[0]} missing from {env_tokens}"


@pytest.mark.parametrize("scheduler_class", [SlurmScheduler, FluxScheduler])
def test_generated_script_keeps_env_out_of_the_cli(scheduler_class):
    """
    Non-regression companion: when the launcher *does* write the script
    (``cli_env_only`` False -- an ordinary ``--bg`` run with a generated
    ``launch.sh``), the environment belongs in the script as ``export``
    lines and must stay off the submit command line, so the two channels
    never duplicate each other.
    """
    system = _system_with_env()
    scheduler = scheduler_class(nodes=1, procs_per_node=4, gpus_per_proc=0)

    header, _ = scheduler.build_command_string_and_batch_script(
        system, blocking=False, cli_env_only=False
    )
    cmd = scheduler.launch_command(system, blocking=False, cli_env_only=False)

    assert f"export {_TUNING_ENV[0]}={_TUNING_ENV[1]}" in header, header
    assert not _cli_env_tokens(cmd), (
        f"script-borne environment leaked onto the submit command line: {cmd}"
    )


class _PassthroughSystem(GenericSystem):
    """A system that publishes a passthrough (not script-injected) variable."""

    def passthrough_environment_variables(self) -> list[tuple[str, str]]:
        return [("HPC_LAUNCHER_PASSTHROUGH", "1")]


@pytest.mark.parametrize("scheduler_class", [SlurmScheduler, FluxScheduler])
@pytest.mark.parametrize("blocking", [True, False])
def test_immutable_script_passthrough_env_rides_the_cli(scheduler_class, blocking):
    """
    ``passthrough_environment_variables()`` had the identical hole one branch
    below: gated on ``blocking`` alone, a non-blocking immutable-script
    submission wrote it into the discarded header instead of the CLI. No
    in-tree system implements the hook yet, so this is a latent defect that
    would bite the first one that does -- pin it now.
    """
    system = _PassthroughSystem()
    scheduler = scheduler_class(nodes=1, procs_per_node=4, gpus_per_proc=0)

    cmd = scheduler.launch_command(system, blocking=blocking, cli_env_only=True)

    assert any(
        "HPC_LAUNCHER_PASSTHROUGH=1" in t for t in _cli_env_tokens(cmd)
    ), f"passthrough environment missing from the command line: {cmd}"


# ---------------------------------------------------------------------------
# PYTHONPATH: the invocation directory, not the launch directory's parent
# ---------------------------------------------------------------------------


def _exported_pythonpath(script: str) -> str:
    """
    The single directory the generated script prepends to ``PYTHONPATH``, or
    ``None`` when the script exports nothing. The line has the shape
    ``export PYTHONPATH=<quoted dir>:${PYTHONPATH}``; the path is quoted, so
    unquote it before comparing.
    """
    lines = [l for l in script.splitlines() if l.startswith("export PYTHONPATH=")]
    if not lines:
        return None
    assert len(lines) == 1, f"expected at most one PYTHONPATH export:\n{script}"
    value = lines[0][len("export PYTHONPATH="):]
    prefix, sep, rest = value.rpartition(":${PYTHONPATH}")
    assert sep, f"PYTHONPATH export does not preserve the inherited value: {value}"
    return shlex.split(prefix)[0]


@pytest.mark.parametrize("scheduler_class", [SlurmScheduler, FluxScheduler])
def test_pythonpath_export_is_the_invocation_directory(
    scheduler_class, tmp_path, monkeypatch
):
    """
    The job runs from the launch directory, so the script re-adds the
    directory the user launched *from* -- that is what makes ``train.py``'s
    sibling modules importable, and for ``torchrun-hpc`` the launcher is the
    only thing that adds it at all.

    It used to export ``dirname(launch_dir)``. With an absolute ``-l``
    (``torchrun-hpc -l /p/lustre1/shared/runs/job1``) that is the launch
    directory's *parent*, an unrelated directory: the user's own code stops
    being importable, and a plausibly group- or world-writable scratch
    directory is placed ahead of site-packages on every rank's import path,
    so any ``.py`` dropped there (``random.py``, ``numpy.py``) is imported by
    the job.
    """
    invocation_dir = tmp_path / "home_me"
    invocation_dir.mkdir()
    launch_dir = tmp_path / "shared_runs" / "job1"
    launch_dir.mkdir(parents=True)
    monkeypatch.chdir(invocation_dir)

    scheduler = scheduler_class(nodes=1, procs_per_node=1, gpus_per_proc=0)
    script = scheduler.launcher_script(
        GenericSystem(), "python", ["train.py"], blocking=False,
        launch_dir=str(launch_dir),
    )

    exported = _exported_pythonpath(script)
    assert exported == str(invocation_dir), (
        f"expected the invocation directory {invocation_dir} on PYTHONPATH, "
        f"got {exported!r}"
    )
    # Specifically not the launch directory's parent, which is what the
    # dirname() computation produced.
    assert exported != str(launch_dir.parent)


@pytest.mark.parametrize(
    "launch_subdir",
    [
        # The auto-generated layout: <cwd>/launch-<job>_<timestamp>_<uuid>.
        "launch-myjob_2026-01-01_00h00m00s_deadbeef",
        # A relative custom -l, resolved against the invocation directory.
        "myrun",
    ],
)
def test_pythonpath_export_unchanged_for_launch_dirs_under_cwd(
    launch_subdir, tmp_path, monkeypatch
):
    """
    Non-regression for the two layouts that were already correct (the second
    only accidentally so: ``abspath`` resolved it against the cwd and
    ``dirname`` then stripped it back off). Both must still export the
    invocation directory.
    """
    invocation_dir = tmp_path / "home_me"
    launch_dir = invocation_dir / launch_subdir
    launch_dir.mkdir(parents=True)
    monkeypatch.chdir(invocation_dir)

    scheduler = SlurmScheduler(nodes=1, procs_per_node=1, gpus_per_proc=0)
    script = scheduler.launcher_script(
        GenericSystem(), "python", ["train.py"], blocking=False,
        launch_dir=str(launch_dir),
    )

    assert _exported_pythonpath(script) == str(invocation_dir), script


def test_no_pythonpath_export_when_launching_from_the_cwd(tmp_path, monkeypatch):
    """
    ``-l .`` runs the job in the invocation directory itself, so there is
    nothing to re-add and the export is skipped entirely -- no empty or
    duplicate entry on the import path.
    """
    monkeypatch.chdir(tmp_path)

    scheduler = SlurmScheduler(nodes=1, procs_per_node=1, gpus_per_proc=0)
    script = scheduler.launcher_script(
        GenericSystem(), "python", ["train.py"], blocking=False,
        launch_dir=os.getcwd(),
    )

    assert _exported_pythonpath(script) is None, script
