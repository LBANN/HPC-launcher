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
Regression tests for three resource-validation defects, all in
``hpc_launcher/systems/configure.py``:

* **The oversubscription warning.** The GPU-oversubscription check used to
  log a message claiming "Job will not launch" at ``logger.info``
  (invisible at default verbosity) and then return the uncorrected,
  oversubscribed configuration anyway -- an outcome the message did not
  actually enforce. It is now a ``logger.warning`` (visible by default)
  that no longer asserts an outcome it does not enforce. It deliberately
  still does *not* raise or silently clamp the values in this specific
  branch -- see the docstring on
  ``test_gpu_oversubscription_is_now_a_visible_warning`` for why not,
  which is also the reason this fix is a warning rather than a
  ``ValueError`` unlike the division below.

* **Dividing by a zero GPU memory size.** ``--gpumem-at-least`` divided by
  ``system_params.mem_per_gpu`` unconditionally, which is exactly ``0`` on
  the "Generic CPU" autodetect fallback (and on any other
  GPU-less/unrecognized host), producing a bare ``ZeroDivisionError``
  traceback instead of an actionable error.

* **Mutating a shared template.** CLI ``--system-params``/``-p`` overrides wrote directly into the
  ``__dict__`` of whatever ``SystemParams`` instance
  ``system.system_parameters()`` returned. Several of those instances
  are module-level singletons deliberately reused as the value for many
  systems/queues (e.g. ``_mi300a_node`` in ``el_capitan_family.py``
  backs every queue on tuolumne, elcap, rzadams and tenaya, plus
  tioga's ``mi300a`` queue) -- so the override corrupted the shared
  template for the rest of the process's lifetime, not just the one job
  that requested it. This is unreachable through a single normal CLI
  invocation (both CLI entry points call ``configure_launch()`` exactly
  once and exit), but a pytest process -- like this one -- calls it more
  than once, which is exactly the reproducing condition.
"""
import logging

import pytest
from unittest.mock import patch

from hpc_launcher.schedulers.scheduler import Scheduler
from hpc_launcher.systems.configure import configure_launch
from hpc_launcher.systems.lc.el_capitan_family import ElCapitan, _mi300a_node
from hpc_launcher.systems.system import System, SystemParams


class _MockScheduler(Scheduler):
    pass


class _MockGpuSystem(System):
    """
    A small, directly-constructed system with a 3-GPU/node queue, standing
    in for a real GPU system without depending on autodetection. Mirrors
    the "mockq" fixture already used by ``tests/launch_config_test.py``,
    kept local here (rather than imported) since that file is out of
    scope for this change.
    """

    def __init__(self):
        super().__init__("mock-gpu-system")
        self.default_queue = "mockq"
        self.system_params = {
            "mockq": SystemParams(
                cores_per_node=24,
                gpus_per_node=3,
                gpu_arch="sm_00",
                mem_per_gpu=11.0,
                scheduler="MockScheduler",
                numa_domains=3,
            ),
        }

    def environment_variables(self) -> list[tuple[str, str]]:
        return []

    @property
    def preferred_scheduler(self) -> type[Scheduler]:
        return _MockScheduler


class _MockGenericCpuFallbackSystem(System):
    """
    Stands in for ``autodetect.autodetect_current_system()``'s "Generic
    CPU" fallback (``autodetect.py:108`` / ``:224-235``): the queue
    ``SystemParams`` it hands back has ``gpus_per_node=0`` and, critically,
    ``mem_per_gpu=0`` -- reached on any host where no GPUs were
    auto-detected (a laptop, a CPU-only CI runner, an unlisted cluster, or
    a machine missing ``amdsmi``/``pynvml``).
    """

    def __init__(self):
        super().__init__("Generic CPU")
        self.default_queue = "auto"
        self.system_params = {
            "auto": SystemParams(
                cores_per_node=4,
                gpus_per_node=0,
                gpu_arch=None,
                mem_per_gpu=0,
                scheduler=None,
                numa_domains=1,
            ),
        }

    def environment_variables(self) -> list[tuple[str, str]]:
        return []

    @property
    def preferred_scheduler(self) -> type[Scheduler]:
        return _MockScheduler


# ---------------------------------------------------------------------------
# GPU oversubscription: a visible warning, not an invisible false claim
# ---------------------------------------------------------------------------
@patch(
    "hpc_launcher.systems.autodetect.autodetect_current_system",
    return_value=_MockGpuSystem(),
)
def test_gpu_oversubscription_is_now_a_visible_warning(mock_autodetect, caplog):
    """
    Reproduces the exact structural shape of the bug: ``gpus_per_proc``
    is individually valid (<= gpus_per_node), so the auto-correction branch
    at configure.py:112-114 is skipped, but ``procs_per_node * gpus_per_proc``
    still exceeds ``gpus_per_node`` (4 requested vs. 3 available on the
    mock's "mockq" queue).

    Before the fix, this combination was logged at ``logger.info`` --
    invisible at the default ``WARNING`` verbosity floor set by
    ``launch_helpers.setup_logging()`` -- with a message claiming "Job
    will not launch", while ``configure_launch()`` returned the
    uncorrected, oversubscribed tuple regardless. That is a false
    assertion: the job *does* launch.

    This exact input shape is also exercised as a *passing* assertion in
    ``tests/launch_config_test.py::test_launch_config`` (the very first
    case, "User-specified procs_per_node", and again later in the same
    test), which asserts the returned configuration is passed through
    unchanged. That test is out of scope for this change (owned
    elsewhere), so raising here -- or silently clamping the values to
    something that fits -- would turn that pre-existing, currently
    green test red with no way to update it. The chosen fix therefore
    preserves the returned values exactly and only fixes the actual
    problem: the message must stop claiming an outcome it does not
    enforce, and it must be visible at default verbosity.
    """
    with caplog.at_level(logging.WARNING):
        system, nodes, procs_per_node, gpus_per_proc = configure_launch(
            None, 2, 4, 1, 0, 0, None
        )

    # The configuration is intentionally still passed through unchanged --
    # see the docstring above for why this branch does not raise or clamp.
    assert nodes == 2
    assert procs_per_node == 4
    assert gpus_per_proc == 1

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, (
        "the oversubscription message must be visible at the default "
        "WARNING verbosity floor (launch_helpers.setup_logging()'s "
        "non-verbose level) -- previously it was logger.info() only"
    )
    combined = " ".join(r.message for r in warnings)
    assert "will not launch" not in combined.lower(), (
        "the message must not claim an outcome ('Job will not launch') "
        "that configure_launch() does not actually enforce -- the call "
        "above returned normally, i.e. the job does launch"
    )
    assert "4 processes per node" in combined and "3" in combined


@patch(
    "hpc_launcher.systems.autodetect.autodetect_current_system",
    return_value=_MockGpuSystem(),
)
def test_gpu_per_proc_still_auto_clamped_when_individually_invalid(mock_autodetect):
    """
    Sanity check that the warning fix did not touch the pre-existing,
    legitimate auto-correction path: when ``gpus_per_proc`` is *individually* invalid
    (here, 4 requested but only 3 GPUs/node exist), it is still clamped
    down to fit rather than warned about. This is the same scenario
    ``launch_config_test.py::test_launch_config`` covers ("Ask for too
    many GPUs per proc"); duplicated narrowly here as a fast, local check
    that the change didn't regress it.
    """
    system, nodes, procs_per_node, gpus_per_proc = configure_launch(
        None, 2, 1, 4, 0, 0, None
    )
    assert nodes == 2
    assert procs_per_node == 1
    assert gpus_per_proc == 3


# ---------------------------------------------------------------------------
# --gpumem-at-least must not divide by a zero GPU memory size
# ---------------------------------------------------------------------------
@patch(
    "hpc_launcher.systems.autodetect.autodetect_current_system",
    return_value=_MockGenericCpuFallbackSystem(),
)
def test_gpumem_at_least_on_gpu_less_host_raises_clean_error(mock_autodetect):
    """
    The reproducer: on a GPU-less/unrecognized host, ``mem_per_gpu`` is
    exactly ``0``. Before the fix, ``--gpumem-at-least`` unconditionally
    computed ``ceildiv(gpumem_at_least, system_params.mem_per_gpu)``,
    dividing by that zero and raising a bare, unactionable
    ``ZeroDivisionError`` from inside ``ceildiv()`` -- before any script
    generation. It must instead be a clear ``ValueError`` naming the
    actual problem (no GPUs / no GPU memory on this system).
    """
    with pytest.raises(ValueError) as exc_info:
        configure_launch(None, 0, 0, None, 0, 16, None)

    assert "mem_per_gpu" in str(exc_info.value) or "no GPU" in str(exc_info.value)
    assert not isinstance(exc_info.value, ZeroDivisionError)


@patch(
    "hpc_launcher.systems.autodetect.autodetect_current_system",
    return_value=_MockGpuSystem(),
)
def test_gpumem_at_least_unaffected_on_a_real_gpu_system(mock_autodetect):
    """
    Sanity check that the guard is scoped to the ``mem_per_gpu == 0``
    case and does not disturb ``--gpumem-at-least`` on an ordinary GPU
    system (mem_per_gpu=11 on the mock's "mockq" queue, 3 GPUs/node).
    """
    system, nodes, procs_per_node, gpus_per_proc = configure_launch(
        None, 0, 0, 1, 0, 22, None
    )
    assert nodes == 1
    assert procs_per_node == 2
    assert gpus_per_proc == 1


# ---------------------------------------------------------------------------
# -p overrides must not mutate a shared SystemParams template
# ---------------------------------------------------------------------------
@patch("hpc_launcher.systems.autodetect.autodetect_current_system")
def test_system_params_cli_override_does_not_mutate_shared_template(
    mock_autodetect,
):
    """
    Reproduces the bug against a real, in-repo shared instance:
    ``_mi300a_node`` in ``el_capitan_family.py`` is a single
    module-level ``SystemParams`` object reused as the value for pbatch and
    pdebug on tuolumne, elcap, rzadams and tenaya, *and* tioga's mi300a
    queue (``ElCapitan._system_params``, `el_capitan_family.py:157-203`).

    Before the fix, ``configure_launch()``'s ``--system-params`` override
    loop wrote straight into ``system_params.__dict__`` -- and since
    ``system.system_parameters(queue)`` hands back that exact shared
    object (not a copy), a CLI override requested for *one* job/queue
    silently corrupted the template for every other job/queue that reuses
    it, for the rest of the process's lifetime.

    Unreachable through one normal CLI invocation (each entry point calls
    ``configure_launch()`` exactly once and exits) -- it requires the
    interpreter to process more than one job, which is exactly what this
    pytest process does by calling it twice below.
    """
    original_fraction = _mi300a_node.fraction_max_gpu_mem
    assert original_fraction == 0.8  # baseline documented at el_capitan_family.py:159
    try:
        # Job 1: tuolumne/pbatch, with an explicit CLI override.
        tuolumne = ElCapitan("tuolumne")
        mock_autodetect.return_value = tuolumne
        system1, *_ = configure_launch(
            None, 1, 4, 1, 0, 0, {"fraction_max_gpu_mem": "0.111"}
        )
        assert system1.active_system_params.fraction_max_gpu_mem == 0.111

        # The shared module-level template itself must be untouched --
        # mutating it is the actual corruption.
        assert _mi300a_node.fraction_max_gpu_mem == original_fraction, (
            "a --system-params override for one job mutated the shared "
            "_mi300a_node singleton in place; it also backs elcap, "
            "rzadams, tenaya, and tioga's mi300a queue"
        )

        # Job 2: a different system/queue that happens to share the same
        # _mi300a_node template, with no override requested at all.
        elcap = ElCapitan("elcap")
        mock_autodetect.return_value = elcap
        system2, *_ = configure_launch(None, 1, 4, 1, 0, 0, None)
        assert system2.active_system_params.fraction_max_gpu_mem == original_fraction, (
            "job 1's fraction_max_gpu_mem override leaked into job 2's "
            "unrelated system/queue via the shared SystemParams template"
        )

        # Job 3: tioga's mi300a queue, also backed by _mi300a_node,
        # likewise unaffected.
        tioga = ElCapitan("tioga")
        mock_autodetect.return_value = tioga
        system3, *_ = configure_launch("mi300a", 1, 4, 1, 0, 0, None)
        assert (
            system3.active_system_params.fraction_max_gpu_mem == original_fraction
        )
    finally:
        # Belt-and-suspenders restoration of module-global state so a
        # failure of this test (e.g. while it is still reproducing the
        # pre-fix bug) cannot pollute any other test in the same pytest
        # process.
        _mi300a_node.fraction_max_gpu_mem = original_fraction
