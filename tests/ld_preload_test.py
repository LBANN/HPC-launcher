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
Tests for system-set scheduler *fields* surviving into the launch command.

``Scheduler.build_command_string_and_batch_script`` does two things in
sequence: it asks the scheduler to turn its own fields into launch arguments
(``build_scheduler_specific_arguments``), and it lets the system customize
the scheduler (``System.customize_scheduler``). The order used to be the
wrong way around for anything the system sets as a scheduler *field*.

``ld_preloads`` is the only such field, and ``customize_scheduler`` is its
only producer -- there is no CLI flag for it. On the El Capitan family
(``hpc_launcher/systems/lc/el_capitan_family.py``) and on Corona,
``LBANN_USE_THIS_RCCL=/path/librccl.so`` assigns it, and the Flux/Slurm
argument builders turn it into ``--env=LD_PRELOAD`` / ``--export=ALL,LD_PRELOAD``.
Because the builder ran first, the field was still ``None`` at that point and
the flag was dropped from every *blocking* launch. ``--bg`` masked the defect:
``launcher_script()`` makes a second, independent pass over the same scheduler
instance, by which time the field is populated, so the batch script carried
the flag and the feature appeared to work in batch mode -- a user
benchmarking a custom RCCL build silently got the system RCCL interactively
and the right one in batch.

The asymmetry worth preserving: ``customize_scheduler``'s *dict* edits
(``common_launch_args`` and friends) always survived the first pass, because
``launch_command()`` consumes those dicts itself, after ``customize_scheduler``
has returned. Only fields were read too early, so the companion tests below
pin the dict edits as non-regressions.

No torch, no scheduler binaries, and no real RCCL: commands are constructed
directly from a stub system that mirrors the real producer, and one test
pins that the real producer still behaves the way the stub claims.
"""
import pytest

from hpc_launcher.schedulers.flux import FluxScheduler
from hpc_launcher.schedulers.slurm import SlurmScheduler
from hpc_launcher.systems.lc.el_capitan_family import ElCapitan
from hpc_launcher.systems.system import GenericSystem

# Stands in for a user's hand-built RCCL, as LBANN_USE_THIS_RCCL would name it.
_RCCL = "/my/rccl/librccl.so"

# How each scheduler spells "preload this library", as its argument builder
# emits it from ``self.ld_preloads``.
_PRELOAD_FLAG = {
    FluxScheduler: f"--env=LD_PRELOAD={_RCCL}",
    SlurmScheduler: f"--export=ALL,LD_PRELOAD={_RCCL}",
}

# A Flux-only launch argument the El Capitan family sets as a *dict* edit,
# used to pin the surviving half of the asymmetry described above.
_FLUX_TUNING_ARG = "-ofastload=on"


class _RcclPreloadSystem(GenericSystem):
    """
    A stand-in for the El Capitan family (and Corona): its
    ``customize_scheduler`` sets the ``ld_preloads`` *field* and, for Flux,
    also makes a *dict* edit -- the two kinds of customization whose
    divergent fates are the subject of this file. Mirrors
    ``ElCapitan.customize_scheduler``; ``test_real_producer_sets_the_field``
    keeps the mirror honest.
    """

    def customize_scheduler(self, scheduler):
        if isinstance(scheduler, FluxScheduler):
            scheduler.common_launch_args["-ofastload"] = "on"
        scheduler.ld_preloads = [_RCCL]


@pytest.mark.parametrize("scheduler_class", [FluxScheduler, SlurmScheduler])
def test_ld_preload_on_blocking_launch_command(scheduler_class):
    """
    The whole point of ``LBANN_USE_THIS_RCCL``: an interactive (blocking)
    launch must actually preload the named library. The field is assigned by
    ``customize_scheduler`` and consumed by the argument builder, so it only
    reaches the command line if the customization happens first.
    """
    system = _RcclPreloadSystem()
    scheduler = scheduler_class(nodes=1, procs_per_node=4, gpus_per_proc=1)

    cmd = scheduler.launch_command(system, blocking=True, cli_env_only=False)

    assert scheduler.ld_preloads == [_RCCL], (
        "the system did set the field; if this fails the stub, not the "
        "ordering, is wrong"
    )
    assert _PRELOAD_FLAG[scheduler_class] in cmd, (
        f"LD_PRELOAD was dropped from the blocking {scheduler_class.__name__} "
        f"command: {cmd}"
    )


@pytest.mark.parametrize("scheduler_class", [FluxScheduler, SlurmScheduler])
def test_ld_preload_on_ephemeral_blocking_launch_command(scheduler_class):
    """
    Same for the ephemeral (no launch folder) blocking path, where the
    command line is the only channel there is: no script is ever written, so
    a flag missing here is missing everywhere.

    The ordering fixed here decides whether the preload is emitted at all;
    ``test_slurm_preload_and_environment_share_one_export`` below covers the
    second half, where it also has to survive alongside the environment.
    """
    system = _RcclPreloadSystem()
    scheduler = scheduler_class(nodes=1, procs_per_node=4, gpus_per_proc=1)

    cmd = scheduler.launch_command(system, blocking=True, cli_env_only=True)

    assert _PRELOAD_FLAG[scheduler_class] in cmd, (
        f"LD_PRELOAD was dropped from the ephemeral blocking "
        f"{scheduler_class.__name__} command: {cmd}"
    )


class _RcclPreloadSystemWithEnv(_RcclPreloadSystem):
    """A preload-setting system that also has environment variables to export."""

    def environment_variables(self):
        return [("NCCL_MIN_NCHANNELS", "24"), ("MIOPEN_DISABLE_CACHE", "0")]


def test_slurm_preload_and_environment_share_one_export():
    """
    srun and sbatch accept a single ``--export``; a second occurrence
    replaces the first rather than adding to it. So it is not enough for the
    preload to be *emitted* -- it has to be emitted into the same token as
    the environment, or whichever producer runs second silently erases the
    other.

    This is the shape that actually occurs in production: every system that
    sets ``ld_preloads`` (the El Capitan family, Corona) also exports a large
    tuning block, so the two producers always collide there. A system with a
    preload and no environment, as the tests above use, cannot show it.
    """
    system = _RcclPreloadSystemWithEnv()
    scheduler = SlurmScheduler(nodes=1, procs_per_node=4, gpus_per_proc=1)

    cmd = scheduler.launch_command(system, blocking=True, cli_env_only=True)

    exports = [token for token in cmd if token.startswith("--export")]
    assert len(exports) == 1, (
        f"srun honors only the last --export, so these collide: {exports}"
    )
    assert f"LD_PRELOAD={_RCCL}" in exports[0], exports[0]
    assert "NCCL_MIN_NCHANNELS=24" in exports[0], exports[0]


@pytest.mark.parametrize("scheduler_class", [FluxScheduler, SlurmScheduler])
def test_ld_preload_in_batch_script_bg_flow(scheduler_class, tmp_path):
    """
    Non-regression for the path that accidentally worked: ``launch()`` builds
    the submit command and *then* generates the script from the same
    scheduler instance, so the script is written on a second pass over a
    scheduler whose field is by then populated. This is the behavior a
    ``--bg`` user sees today, and it must keep working.
    """
    system = _RcclPreloadSystem()
    scheduler = scheduler_class(nodes=1, procs_per_node=4, gpus_per_proc=1)

    # The two passes ``Scheduler.launch`` makes, in order.
    scheduler.launch_command(system, blocking=False, cli_env_only=False)
    script = scheduler.launcher_script(
        system, "python", ["train.py"], blocking=False, launch_dir=str(tmp_path)
    )

    assert f"LD_PRELOAD={_RCCL}" in script, (
        f"LD_PRELOAD missing from the generated batch script:\n{script}"
    )


@pytest.mark.parametrize("scheduler_class", [FluxScheduler, SlurmScheduler])
def test_ld_preload_in_batch_script_single_pass(scheduler_class, tmp_path):
    """
    ...and the script must not depend on that second pass to be correct. One
    pass over a fresh scheduler is all a caller is contractually owed, and
    it is what ``--bg`` would fall back to if the command were ever built
    from a different instance.
    """
    system = _RcclPreloadSystem()
    scheduler = scheduler_class(nodes=1, procs_per_node=4, gpus_per_proc=1)

    script = scheduler.launcher_script(
        system, "python", ["train.py"], blocking=False, launch_dir=str(tmp_path)
    )

    assert f"LD_PRELOAD={_RCCL}" in script, (
        f"LD_PRELOAD missing from the generated batch script:\n{script}"
    )


def test_customize_scheduler_dict_edits_survive():
    """
    The other half of the asymmetry, pinned so a fix for the field ordering
    cannot regress it: launch arguments the system adds to
    ``common_launch_args`` reach the blocking command line, because
    ``launch_command`` consumes that dict itself.
    """
    system = _RcclPreloadSystem()
    scheduler = FluxScheduler(nodes=1, procs_per_node=4, gpus_per_proc=1)

    cmd = scheduler.launch_command(system, blocking=True, cli_env_only=False)

    assert _FLUX_TUNING_ARG in cmd, (
        f"a system launch-argument customization was lost: {cmd}"
    )
    # The scheduler's own arguments are still there alongside it.
    assert "-N1" in cmd and "-n4" in cmd, cmd


def test_real_producer_sets_the_field(monkeypatch):
    """
    Keep ``_RcclPreloadSystem`` honest: the real producer is
    ``ElCapitan.customize_scheduler`` reading ``LBANN_USE_THIS_RCCL``. Only
    the customization hook is called here (not the full command build), so
    this test does not depend on the host's ROCm/RCCL state.
    """
    monkeypatch.setenv("LBANN_USE_THIS_RCCL", _RCCL)
    scheduler = SlurmScheduler(nodes=1, procs_per_node=4, gpus_per_proc=1)

    ElCapitan("tuolumne").customize_scheduler(scheduler)

    assert scheduler.ld_preloads == [_RCCL]


def test_no_ld_preload_flag_without_the_variable():
    """
    Nothing is preloaded when no system asks for it: the flag must not
    appear for a plain system, in either the command or the script.
    """
    system = GenericSystem()
    scheduler = FluxScheduler(nodes=1, procs_per_node=4, gpus_per_proc=1)

    cmd = scheduler.launch_command(system, blocking=True, cli_env_only=False)

    assert not any("LD_PRELOAD" in token for token in cmd), cmd
