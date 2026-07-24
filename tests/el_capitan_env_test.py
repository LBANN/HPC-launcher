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
Tests for the El Capitan family ROCm / RCCL environment block.

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
    # raising=False: harmless if the constant does not exist.
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
    """An unversioned ROCM_PATH symlink resolves through realpath."""
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
    """No version in ROCM_PATH and no torch available must not crash."""
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
    """ROCm 8.0 must take the >=7.1 branch (tuple comparison)."""
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


def test_version_prefers_torch_hip(monkeypatch, tmp_path, rocm_test_env, caplog):
    """
    Torch's bundled ROCm (7.2) wins over a mismatched ROCM_PATH
    (6.4.2): the probe looks in the 7.2 tree, the >=7.1 NCCL branch is
    taken, a mismatch warning is logged, and the mismatched ROCM_PATH's
    llvm/lib is NOT prepended to LD_LIBRARY_PATH.
    """
    rocm = tmp_path / "rocm-6.4.2"
    rocm.mkdir()
    monkeypatch.setenv("ROCM_PATH", str(rocm))
    _fake_torch(monkeypatch, "7.2.24191-cf58cf3856")
    plugin_lib = _make_plugin_tree(rocm_test_env, "rocm-7.2.0")

    system = ElCapitan("tuolumne")
    system.job_comm_protocol = "RCCL"
    with caplog.at_level(logging.WARNING):
        env_list = system.environment_variables()

    ld_paths = _ld_library_path_values(env_list)
    assert any(str(plugin_lib) in v for v in ld_paths)
    env = {e[0]: e[1] for e in _env_pairs(env_list)}
    assert env.get("NCCL_NET") == "libfabric"
    assert env.get("NCCL_NET_PLUGIN") == "librccl-net.so"
    # The 6.4.2 llvm/lib prepend is an ABI hazard next to a 7.2 torch.
    assert not any("llvm" in v for v in ld_paths)
    assert any("mismatch" in record.message.lower() for record in caplog.records)


def test_nccl_net_not_forced_without_plugin(monkeypatch, rocm_test_env, caplog):
    """
    With no plugin anywhere and no override, NCCL_NET and
    NCCL_NET_PLUGIN must not be forced (forcing them hard-crashes RCCL
    init on plugin-less wheels); a warning naming the remedy is logged.
    """
    _fake_torch(monkeypatch, "7.2.0")
    # rocm_test_env points the probe at an empty scratch tree.

    system = ElCapitan("tuolumne")
    system.job_comm_protocol = "RCCL"
    with caplog.at_level(logging.WARNING):
        env_list = system.environment_variables()

    names = _env_names(env_list)
    assert "NCCL_NET" not in names
    assert "NCCL_NET_PLUGIN" not in names
    assert any(
        "LBANN_USE_THIS_OFI_PLUGIN" in record.message for record in caplog.records
    )


def test_nccl_net_set_when_plugin_present(monkeypatch, rocm_test_env):
    """
    With a matching plugin tree the NCCL knobs are set -- and this
    works with ROCM_PATH entirely unset (the torch wheel case).
    """
    _fake_torch(monkeypatch, "7.2.0")
    plugin_lib = _make_plugin_tree(rocm_test_env, "rocm-7.2.0")

    system = ElCapitan("tuolumne")
    system.job_comm_protocol = "RCCL"
    env_list = system.environment_variables()

    env = {e[0]: e[1] for e in _env_pairs(env_list)}
    assert env.get("NCCL_NET") == "libfabric"
    assert env.get("NCCL_NET_PLUGIN") == "librccl-net.so"
    assert any(str(plugin_lib) in v for v in _ld_library_path_values(env_list))


def test_nccl_net_plugin_fuzzy_tree_match(monkeypatch, rocm_test_env):
    """
    ROCm 7.2.1 requested but only a rocm-7.2.0 plugin tree
    exists -- the probe must accept the same-major.minor sibling.
    """
    _fake_torch(monkeypatch, "7.2.1")
    plugin_lib = _make_plugin_tree(rocm_test_env, "rocm-7.2.0")

    system = ElCapitan("tuolumne")
    system.job_comm_protocol = "RCCL"
    env_list = system.environment_variables()

    env = {e[0]: e[1] for e in _env_pairs(env_list)}
    assert env.get("NCCL_NET") == "libfabric"
    assert env.get("NCCL_NET_PLUGIN") == "librccl-net.so"
    assert any(str(plugin_lib) in v for v in _ld_library_path_values(env_list))


def test_explicit_plugin_override(monkeypatch, tmp_path, rocm_test_env):
    """
    LBANN_USE_THIS_OFI_PLUGIN bypasses probing entirely, even when a
    probe tree exists.
    """
    _fake_torch(monkeypatch, "7.2.0")
    probe_lib = _make_plugin_tree(rocm_test_env, "rocm-7.2.0")
    override = tmp_path / "my-plugin"
    override.mkdir()
    monkeypatch.setenv("LBANN_USE_THIS_OFI_PLUGIN", str(override))

    system = ElCapitan("tuolumne")
    system.job_comm_protocol = "RCCL"
    env_list = system.environment_variables()

    ld_paths = _ld_library_path_values(env_list)
    assert any(str(override) in v for v in ld_paths)
    assert not any(str(probe_lib) in v for v in ld_paths)
    env = {e[0]: e[1] for e in _env_pairs(env_list)}
    assert env.get("NCCL_NET") == "libfabric"
    assert env.get("NCCL_NET_PLUGIN") == "librccl-net.so"
