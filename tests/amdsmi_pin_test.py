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
Coverage for the ``[rocm-auto]`` extra's amdsmi version pin in ``setup.py``.

An amdsmi that doesn't match the ROCm runtime it talks to is broken at
runtime, so the ``[rocm-auto]`` extra probes the machine running pip
(``$ROCM_PATH/.info/version``, default ``/opt/rocm``) and pins amdsmi as
close to that ROCm release as PyPI allows. amdsmi's release history has
gaps (e.g. no 6.2.3, and 6.2.2 exists only as 6.2.2.post0) and lags
GitHub/ROCm (nothing past 7.0.x while ROCm 7.2 ships), so exact ``==``
pins would be unsatisfiable for real ROCm releases -- the pin styles
here are compatible-release (``~=``) and range constraints instead.

The plain ``[rocm]`` extra must stay unpinned unconditionally: published
wheels freeze their metadata at build time, so a probing pin there would
bake the *build* machine's ROCm into every install.

These tests exec ``setup.py`` as a module (with ``setuptools.setup``
mocked out so no real build machinery runs) with ``ROCM_PATH`` pointed
at scratch directories, so the outcome never depends on the ROCm
installation (or lack of one) of the machine running the tests.
"""
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SETUP_PY = _REPO_ROOT / "setup.py"


def _load_setup(monkeypatch, tmp_path, *, rocm_version):
    """
    Exec ``setup.py`` under a simulated system and return the keyword
    arguments it passed to ``setuptools.setup()``.

    ``rocm_version`` is the contents of ``$ROCM_PATH/.info/version``, or
    ``None`` to simulate a machine with no ROCm installation (``ROCM_PATH``
    is pointed at an empty scratch directory either way, so the real
    host's ``/opt/rocm`` can never leak in).
    """
    monkeypatch.chdir(_REPO_ROOT)

    rocm_root = tmp_path / f"rocm-{rocm_version}"
    rocm_root.mkdir(exist_ok=True)
    if rocm_version is not None:
        info_dir = rocm_root / ".info"
        info_dir.mkdir(exist_ok=True)
        (info_dir / "version").write_text(rocm_version + "\n")
    monkeypatch.setenv("ROCM_PATH", str(rocm_root))

    mock_setup = MagicMock()
    spec = importlib.util.spec_from_file_location(
        f"setup_under_test_{id(mock_setup)}", _SETUP_PY
    )
    module = importlib.util.module_from_spec(spec)
    with patch("setuptools.setup", mock_setup):
        spec.loader.exec_module(module)

    assert mock_setup.called, "setup.py did not call setuptools.setup()"
    return mock_setup.call_args.kwargs


def _amdsmi_dep(setup_kwargs, extra):
    deps = setup_kwargs["extras_require"][extra]
    amdsmi_deps = [d for d in deps if d.lower().startswith("amdsmi")]
    assert len(amdsmi_deps) == 1, f"expected exactly one amdsmi dep, got {deps}"
    return amdsmi_deps[0]


@pytest.mark.parametrize(
    "rocm_version, expected",
    [
        # Pre-7: compatible-release pin. ~=X.Y.Z means >=X.Y.Z,==X.Y.* --
        # it tolerates amdsmi's release gaps (no 6.2.3; 6.2.2 shipped only
        # as 6.2.2.post0) by accepting later releases of the same minor,
        # where an exact ==X.Y.Z would be unsatisfiable.
        ("6.2.2", "amdsmi~=6.2.2"),
        ("6.2.3", "amdsmi~=6.2.3"),
        ("6.4.2", "amdsmi~=6.4.2"),
        # amdsmi releases on PyPI lag GitHub/ROCm (nothing past 7.0.x while
        # ROCm 7.2 ships), so ROCm >= 7 accepts the whole major up through
        # the system's minor instead of pinning to one minor.
        ("7.0.1", "amdsmi>=7,<7.1"),
        ("7.2.0", "amdsmi>=7,<7.3"),
    ],
)
def test_rocm_auto_pins_to_system_rocm(monkeypatch, tmp_path, rocm_version, expected):
    """[rocm-auto] pins amdsmi from $ROCM_PATH/.info/version."""
    kwargs = _load_setup(monkeypatch, tmp_path, rocm_version=rocm_version)
    assert _amdsmi_dep(kwargs, "rocm-auto") == expected


def test_version_file_with_trailing_build_suffix(monkeypatch, tmp_path):
    """
    Some ROCm installs write more than major.minor.patch into the version
    file (e.g. ``6.2.4-66``); only the leading release triple may end up
    in the pin, otherwise pip is handed a version that doesn't exist.
    """
    kwargs = _load_setup(monkeypatch, tmp_path, rocm_version="6.2.4-66")
    assert _amdsmi_dep(kwargs, "rocm-auto") == "amdsmi~=6.2.4"


def test_no_system_rocm_leaves_rocm_auto_unpinned(monkeypatch, tmp_path):
    """With no ROCm installation there is nothing to match -- no pin."""
    kwargs = _load_setup(monkeypatch, tmp_path, rocm_version=None)
    assert _amdsmi_dep(kwargs, "rocm-auto") == "amdsmi"


@pytest.mark.parametrize("rocm_version", ["6.4.2", None])
def test_plain_rocm_extra_is_always_unpinned(monkeypatch, tmp_path, rocm_version):
    """
    [rocm] must stay unpinned whether or not the machine has ROCm: a
    published wheel's metadata is frozen at build time, so any probing
    pin here would bake the build machine's ROCm into every install.
    """
    kwargs = _load_setup(monkeypatch, tmp_path, rocm_version=rocm_version)
    assert _amdsmi_dep(kwargs, "rocm") == "amdsmi"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
