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
Tier A tests for the El Capitan family ROCm / RCCL environment block
(review findings G2, E7, E8).

No real torch and no real ``/collab`` plugin trees are used: ``torch`` is
replaced in ``sys.modules`` with a stub exposing ``version.hip`` (the
profile imports it lazily), the aws-ofi-rccl probe root is pointed at
``tmp_path``, and every relevant environment variable is monkeypatched.
"""
import logging
import sys
import types

import pytest

from hpc_launcher.systems.lc import el_capitan_family
from hpc_launcher.systems.lc.el_capitan_family import ElCapitan


def _fake_torch(monkeypatch, hip):
    """
    Install a fake ``torch`` module in ``sys.modules`` exposing
    ``torch.version.hip``. The profile's torch import is lazy, so a
    SimpleNamespace is enough. ``hip=None`` models a torch build without
    a HIP runtime (or, equivalently for the profile, no torch at all).
    """
    fake = types.SimpleNamespace(version=types.SimpleNamespace(hip=hip))
    monkeypatch.setitem(sys.modules, "torch", fake)


def _make_plugin_tree(probe_base, rocm_dirname):
    """Create ``<probe_base>/<rocm_dirname>/install/lib`` with a plugin lib."""
    lib = probe_base / rocm_dirname / "install" / "lib"
    lib.mkdir(parents=True)
    (lib / "librccl-net.so").touch()
    return lib


@pytest.fixture(autouse=True)
def rocm_test_env(monkeypatch, tmp_path):
    """
    Isolate every test from the host's ROCm/RCCL state: clear the ROCm
    environment variables, stub torch to "no HIP runtime", and point the
    aws-ofi-rccl plugin probe at a scratch tree under tmp_path.

    Returns the directory the probe scans for ``rocm-X.Y.Z`` trees.
    """
    for var in (
        "ROCM_PATH",
        "NCCL_NET",
        "NCCL_NET_PLUGIN",
        "LBANN_USE_THIS_OFI_PLUGIN",
        "CRAY_LD_LIBRARY_PATH",
    ):
        monkeypatch.delenv(var, raising=False)
    _fake_torch(monkeypatch, None)
    root = tmp_path / "rccl-plugins"
    # raising=False: the constant is introduced by the E8 rework; harmless
    # before that.
    monkeypatch.setattr(
        el_capitan_family, "_AWS_OFI_RCCL_ROOT", str(root), raising=False
    )
    monkeypatch.setenv("SYS_TYPE", "test_sys_type")
    return root / "test_sys_type"


def _env_pairs(env_list):
    """The (name, value) entries of the env list (comment lines dropped)."""
    return [e for e in env_list if len(e) >= 2]


def _env_names(env_list):
    return [e[0] for e in _env_pairs(env_list)]


def _ld_library_path_values(env_list):
    return [e[1] for e in _env_pairs(env_list) if e[0] == "LD_LIBRARY_PATH"]


def test_unversioned_rocm_path_no_crash_symlink(monkeypatch, tmp_path):
    """G2: an unversioned ROCM_PATH symlink resolves through realpath."""
    real = tmp_path / "rocm-6.4.2"
    real.mkdir()
    link = tmp_path / "rocm"
    link.symlink_to(real)
    monkeypatch.setenv("ROCM_PATH", str(link))

    env_list = ElCapitan("tuolumne").environment_variables()

    names = _env_names(env_list)
    assert "MIOPEN_USER_DB_PATH" in names
    # The symlink resolves to a versioned tree, so the version-dependent
    # configuration runs: ROCm 6.4 takes the pre-7.1 branch.
    assert "NCCL_NET_PLUGIN" not in names


def test_unversioned_rocm_path_no_crash_no_version_anywhere(
    monkeypatch, tmp_path, caplog
):
    """G2 reproducer: no version in ROCM_PATH and no torch must not crash."""
    plain = tmp_path / "rocm"
    plain.mkdir()
    monkeypatch.setenv("ROCM_PATH", str(plain))

    with caplog.at_level(logging.WARNING):
        env_list = ElCapitan("tuolumne").environment_variables()

    names = _env_names(env_list)
    assert "MIOPEN_USER_DB_PATH" in names
    # The version-dependent configuration is skipped, with a warning.
    assert "NCCL_NET" not in names
    assert "NCCL_NET_PLUGIN" not in names
    assert any("ROCm" in record.message for record in caplog.records)


def test_version_gate_tuple_compare(monkeypatch, tmp_path):
    """G2: ROCm 8.0 must take the >=7.1 branch (tuple comparison)."""
    rocm = tmp_path / "rocm-8.0.1"
    rocm.mkdir()
    monkeypatch.setenv("ROCM_PATH", str(rocm))
    # Provide a plugin via the explicit override so the NCCL knobs are
    # emitted regardless of probe results.
    override = tmp_path / "plugin-override"
    override.mkdir()
    monkeypatch.setenv("LBANN_USE_THIS_OFI_PLUGIN", str(override))

    env_list = ElCapitan("tuolumne").environment_variables()

    env = {e[0]: e[1] for e in _env_pairs(env_list)}
    assert env.get("NCCL_NET") == "libfabric"
    assert env.get("NCCL_NET_PLUGIN") == "librccl-net.so"
