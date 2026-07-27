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
Regression coverage for Q1: ``setup.py`` used to probe the *build* machine
for GPU vendor libraries (``ctypes.util.find_library("amdhip64")`` /
``("cudart")``) and fold the matching hard dependency (``amdsmi`` /
``nvidia-ml-py``) straight into ``install_requires``. That means the exact
same source tree produces a *different* wheel depending on whether it was
built on an AMD node, an NVIDIA node, or a CPU-only CI runner -- a build
reproducibility bug, and a landmine for air-gapped/private-mirror installs
that don't happen to carry the hardware-mismatched package.

These tests load ``setup.py`` as a module (with ``setuptools.setup``
mocked out so the real build/install machinery never runs) under three
simulated build hosts -- AMD-only, NVIDIA-only, and CPU-only -- and check
that the computed package metadata does not depend on which one it was.
"""
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SETUP_PY = _REPO_ROOT / "setup.py"


def _load_setup_kwargs(monkeypatch, tmp_path, *, amdhip64: str | None,
                        cudart: str | None, rocm_version: str | None = None) -> dict:
    """
    Exec ``setup.py`` as a fresh module under a simulated build host and
    return the keyword arguments it passed to ``setuptools.setup()``.

    ``amdhip64``/``cudart`` stand in for what
    ``ctypes.util.find_library(...)`` would return on the build host (a
    library path, or ``None`` if absent). ``rocm_version`` simulates the
    contents of ``$ROCM_PATH/.info/version`` for the AMD case -- when set,
    a scratch directory under ``tmp_path`` is built and ``ROCM_PATH`` is
    pointed at it so the real host's ROCm install (or lack of one) can't
    leak into the test.

    ``setuptools.setup`` is mocked out (rather than e.g. relying on an
    ``if __name__ == "__main__":`` guard in ``setup.py``) so this works
    unmodified against both the pre-fix and post-fix file -- the
    reproducer itself must not depend on a structural change that's part
    of the fix we're about to make.
    """
    monkeypatch.chdir(_REPO_ROOT)

    if rocm_version is not None:
        rocm_root = tmp_path / f"rocm-{id(rocm_version)}-{amdhip64}-{cudart}"
        info_dir = rocm_root / ".info"
        info_dir.mkdir(parents=True, exist_ok=True)
        (info_dir / "version").write_text(rocm_version + "\n")
        monkeypatch.setenv("ROCM_PATH", str(rocm_root))
    else:
        monkeypatch.delenv("ROCM_PATH", raising=False)

    def fake_find_library(name):
        if name == "amdhip64":
            return amdhip64
        if name == "cudart":
            return cudart
        return None

    monkeypatch.setattr("ctypes.util.find_library", fake_find_library)

    mock_setup = MagicMock()
    # A fresh module object per call, so module-level state (the `extras`
    # list) can't leak between simulated hosts.
    spec = importlib.util.spec_from_file_location(
        f"setup_under_test_{id(mock_setup)}", _SETUP_PY
    )
    module = importlib.util.module_from_spec(spec)
    with patch("setuptools.setup", mock_setup):
        spec.loader.exec_module(module)

    assert mock_setup.called, "setup.py did not call setuptools.setup()"
    return mock_setup.call_args.kwargs


def test_install_requires_is_independent_of_build_host(monkeypatch, tmp_path):
    """
    The reproducer for Q1: ``install_requires`` must be identical no
    matter which GPU vendor libraries (if any) happen to be present on
    the machine running ``setup.py``. Before the fix, an AMD build host
    got ``amdsmi`` baked in, an NVIDIA build host got ``nvidia-ml-py``,
    and a CPU-only host got neither -- three different hard dependency
    sets from one commit.
    """
    amd_host = _load_setup_kwargs(
        monkeypatch, tmp_path, amdhip64="libamdhip64.so.7", cudart=None,
        rocm_version="6.4.2",
    )
    nvidia_host = _load_setup_kwargs(
        monkeypatch, tmp_path, amdhip64=None, cudart="libcudart.so.12"
    )
    cpu_only_host = _load_setup_kwargs(
        monkeypatch, tmp_path, amdhip64=None, cudart=None
    )

    assert amd_host["install_requires"] == nvidia_host["install_requires"]
    assert amd_host["install_requires"] == cpu_only_host["install_requires"]


def test_install_requires_never_contains_gpu_vendor_packages(monkeypatch, tmp_path):
    """
    Belt-and-suspenders on top of the identity check above: even on a
    simulated AMD *and* NVIDIA build host, neither ``amdsmi`` nor
    ``nvidia-ml-py`` (nor its import name ``pynvml``) may show up in
    ``install_requires``. Those packages are optional -- both call sites
    (``autodetect.py``'s ``find_AMD_gpus``/``find_NVIDIA_gpus``) already
    guard the import and degrade gracefully -- so they belong in
    ``extras_require``, not as a hard dependency of every install.
    """
    both_host = _load_setup_kwargs(
        monkeypatch, tmp_path,
        amdhip64="libamdhip64.so.7",
        cudart="libcudart.so.12",
        rocm_version="6.4.2",
    )
    install_requires = " ".join(both_host["install_requires"]).lower()
    assert "amdsmi" not in install_requires
    assert "nvidia-ml-py" not in install_requires
    assert "pynvml" not in install_requires


def test_gpu_extras_available_regardless_of_build_host(monkeypatch, tmp_path):
    """
    The fix's destination for the vendor packages is ``extras_require``,
    e.g. ``pip install hpc-launcher[rocm]`` / ``[cuda]`` -- exactly like
    the existing ``torch``/``mpi``/``testing`` groups in this same file.
    Those groups must be populated unconditionally: if they were only
    filled in when the *build* host happened to have the matching vendor
    library, a wheel built on a CPU-only CI runner would ship an empty
    (uninstallable-as-intended) ``[rocm]``/``[cuda]`` extra, which just
    relocates the non-reproducibility bug instead of fixing it.
    """
    cpu_only_host = _load_setup_kwargs(
        monkeypatch, tmp_path, amdhip64=None, cudart=None
    )
    amd_host = _load_setup_kwargs(
        monkeypatch, tmp_path, amdhip64="libamdhip64.so.7", cudart=None,
        rocm_version="6.4.2",
    )

    extras_require = cpu_only_host["extras_require"]
    assert extras_require == amd_host["extras_require"]
    assert any("amdsmi" in dep.lower() for dep in extras_require.get("rocm", [])), (
        "extras_require['rocm'] should carry amdsmi regardless of the build "
        "host's detected GPU vendor"
    )
    assert any(
        "nvidia-ml-py" in dep.lower() for dep in extras_require.get("cuda", [])
    ), (
        "extras_require['cuda'] should carry nvidia-ml-py regardless of the "
        "build host's detected GPU vendor"
    )


if __name__ == "__main__":
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__, "-v"]))
