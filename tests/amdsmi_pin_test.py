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
Coverage for the ``[rocm]`` extra's amdsmi version pin in ``setup.py``.

amdsmi only talks to the ROCm runtime that PyTorch loads, so when a ROCm
build of torch is already present in the environment, ``setup.py`` pins
amdsmi to that torch's ROCm release (read from the ``+rocmX.Y`` local
version tag on the distribution, or from ``torch.version.hip`` for
source builds, which carry no tag). When there is no torch, or torch is
a CPU or CUDA build, there is nothing to match and amdsmi must stay
unpinned.

These tests exec ``setup.py`` as a module (with ``setuptools.setup``
mocked out so no real build machinery runs) under simulated Python
environments -- the installed torch distribution and the importable
``torch`` module are both faked, so the outcome never depends on what
happens to be installed where the tests run.
"""
import importlib.metadata
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SETUP_PY = _REPO_ROOT / "setup.py"

# Sentinel distinguishing "simulate torch absent" from "hip is None"
_NO_TORCH = object()


def _load_setup(monkeypatch, *, torch_dist_version, torch_hip=_NO_TORCH):
    """
    Exec ``setup.py`` under a simulated Python environment and return
    ``(module, setup_kwargs)``.

    ``torch_dist_version`` is what ``importlib.metadata.version("torch")``
    reports (e.g. ``"2.4.1+rocm6.2"``), or ``None`` to simulate torch not
    being installed. ``torch_hip`` plants a fake importable ``torch``
    module whose ``torch.version.hip`` has that value; leave it at the
    default to simulate ``import torch`` failing outright.
    """
    monkeypatch.chdir(_REPO_ROOT)

    if torch_dist_version is None:
        def fake_dist_version(name):
            raise importlib.metadata.PackageNotFoundError(name)
    else:
        def fake_dist_version(name):
            assert name == "torch"
            return torch_dist_version

    # setup.py imports `version` from importlib.metadata at call time, so
    # patching the module attribute is sufficient.
    monkeypatch.setattr(importlib.metadata, "version", fake_dist_version)

    if torch_hip is _NO_TORCH:
        # Setting a sys.modules entry to None makes `import torch` raise
        # ImportError without touching any real installation.
        monkeypatch.setitem(sys.modules, "torch", None)
    else:
        fake_torch = types.ModuleType("torch")
        fake_torch.version = types.SimpleNamespace(hip=torch_hip)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

    mock_setup = MagicMock()
    spec = importlib.util.spec_from_file_location(
        f"setup_under_test_{id(mock_setup)}", _SETUP_PY
    )
    module = importlib.util.module_from_spec(spec)
    with patch("setuptools.setup", mock_setup):
        spec.loader.exec_module(module)

    assert mock_setup.called, "setup.py did not call setuptools.setup()"
    return module, mock_setup.call_args.kwargs


def _rocm_extra(setup_kwargs):
    deps = setup_kwargs["extras_require"]["rocm"]
    amdsmi_deps = [d for d in deps if d.lower().startswith("amdsmi")]
    assert len(amdsmi_deps) == 1, f"expected exactly one amdsmi dep, got {deps}"
    return amdsmi_deps[0]


@pytest.mark.parametrize(
    "dist_version, expected",
    [
        # ROCm wheels carry a +rocmX.Y(.Z) local version tag. Pre-7 amdsmi
        # releases track ROCm patch releases, so pin to the matching series.
        ("2.4.1+rocm6.2", "amdsmi==6.2.*"),
        ("2.6.0+rocm6.4.2", "amdsmi==6.4.2"),
        # amdsmi releases on PyPI lag GitHub/ROCm, so ROCm >= 7 accepts the
        # whole major up through torch's minor instead of an exact pin.
        ("2.7.0+rocm7.0", "amdsmi>=7,<7.1"),
        ("2.7.0+rocm7.0.1", "amdsmi>=7,<7.1"),
    ],
)
def test_rocm_torch_wheel_pins_amdsmi(monkeypatch, dist_version, expected):
    """A ROCm build of torch pins amdsmi to that torch's ROCm release."""
    _, kwargs = _load_setup(monkeypatch, torch_dist_version=dist_version)
    assert _rocm_extra(kwargs) == expected


def test_source_built_torch_uses_hip_version(monkeypatch):
    """
    Source builds of torch have no ``+rocm`` tag on the distribution, so
    the pin falls back to ``torch.version.hip``. Its value looks like
    ``6.2.41133-dd7f9576``, where only major.minor identify the ROCm
    release (the rest is a build number) -- the pin must not treat
    ``41133`` as a patch version.
    """
    _, kwargs = _load_setup(
        monkeypatch,
        torch_dist_version="2.5.0a0+gitdeadbee",
        torch_hip="6.2.41133-dd7f9576",
    )
    assert _rocm_extra(kwargs) == "amdsmi==6.2.*"


@pytest.mark.parametrize(
    "dist_version, torch_hip",
    [
        # CUDA wheel: importable torch, but hip is None
        ("2.4.1+cu121", None),
        # CPU wheel
        ("2.4.1+cpu", None),
        # torch distribution metadata present but the module isn't importable
        # (broken/half-installed environment) -- must not blow up setup.py
        ("2.5.0a0+gitdeadbee", _NO_TORCH),
    ],
)
def test_non_rocm_torch_leaves_amdsmi_unpinned(monkeypatch, dist_version, torch_hip):
    """A CPU or CUDA torch has no ROCm release to match -- no pin."""
    _, kwargs = _load_setup(
        monkeypatch, torch_dist_version=dist_version, torch_hip=torch_hip
    )
    assert _rocm_extra(kwargs) == "amdsmi"


def test_no_torch_leaves_amdsmi_unpinned(monkeypatch):
    """Without torch in the environment there is nothing to match -- no pin."""
    _, kwargs = _load_setup(monkeypatch, torch_dist_version=None)
    assert _rocm_extra(kwargs) == "amdsmi"


def test_detection_never_imports_torch_when_wheel_tag_present(monkeypatch):
    """
    Reading the distribution's version string is enough for wheel installs;
    ``import torch`` (slow, loads GPU runtimes) must only happen as the
    fallback for untagged source builds.
    """
    class _Exploding(types.ModuleType):
        def __getattr__(self, name):
            raise AssertionError("torch was imported despite a +rocm wheel tag")

    module, kwargs = _load_setup(monkeypatch, torch_dist_version="2.4.1+rocm6.2")
    monkeypatch.setitem(sys.modules, "torch", _Exploding("torch"))
    assert module.get_torch_rocm_version() == "6.2"
    assert _rocm_extra(kwargs) == "amdsmi==6.2.*"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
