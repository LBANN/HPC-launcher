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
import pytest
from unittest.mock import patch

from hpc_launcher.systems.lc.corona import Corona
from hpc_launcher.systems.lc.corona import _system_params as _corona_system_params
from hpc_launcher.systems.lc.cts2 import CTS2
from hpc_launcher.systems.lc.cts2 import _system_params as _cts2_system_params
from hpc_launcher.systems.lc.el_capitan_family import ElCapitan
from hpc_launcher.systems.lc.el_capitan_family import (
    _system_params as _el_capitan_system_params,
)
from hpc_launcher.systems.lc.sierra_family import Sierra
from hpc_launcher.systems.lc.sierra_family import _system_params as _sierra_system_params
from hpc_launcher.systems.system import GenericSystem
from hpc_launcher.systems.autodetect import (
    system,
    autodetect_current_system,
    clear_autodetected_system,
)


@patch("socket.gethostname", return_value="linux123")
def test_system(mock_gethostname):
    clear_autodetected_system()
    assert system() == "linux"


@patch("socket.gethostname", return_value="tuolumne0001")
def test_autodetect_el_capitan(mock_gethostname):
    clear_autodetected_system()
    assert isinstance(autodetect_current_system(), ElCapitan)


@patch("socket.gethostname", return_value="lassen001")
def test_autodetect_sierra(mock_gethostname):
    clear_autodetected_system()
    assert isinstance(autodetect_current_system(), Sierra)


@patch("socket.gethostname", return_value="linux")
def test_autodetect_generic(mock_gethostname):
    clear_autodetected_system()
    assert system() == "linux"
    assert isinstance(autodetect_current_system(), GenericSystem)


# ---------------------------------------------------------------------------
# G1 -- anti-drift: every family table's hostname keys must match the
# hostnames autodetect.py routes to that family, so a typo like
# `rzanzel`/`rzansel` can never silently reappear.
# ---------------------------------------------------------------------------
_FAMILY_TABLES = [
    (_el_capitan_system_params, ElCapitan),
    (_cts2_system_params, CTS2),
    (_sierra_system_params, Sierra),
    (_corona_system_params, Corona),
]

_FAMILY_HOSTNAMES = [
    (hostname, expected_class)
    for table, expected_class in _FAMILY_TABLES
    for hostname in table
]


@pytest.mark.parametrize("hostname,expected_class", _FAMILY_HOSTNAMES)
def test_family_keys_match_autodetect(hostname, expected_class):
    """
    Every hostname key registered in a family's known-systems table
    (sierra_family, el_capitan_family, cts2, corona) must autodetect to
    that family's ``System`` subclass, under its own name. This is the
    regression guard for the ``rzanzel``/``rzansel`` typo (finding G1): if
    a hostname string in ``autodetect.py``'s table ever drifts from the
    corresponding family table's key again, the mismatched case fails
    here instead of silently falling back to ``GenericSystem``.
    """
    with patch("socket.gethostname", return_value=f"{hostname}0001"):
        clear_autodetected_system()
        detected = autodetect_current_system(quiet=True)
    assert isinstance(detected, expected_class), (
        f"hostname {hostname!r} did not autodetect as "
        f"{expected_class.__name__}; got {type(detected).__name__} "
        f"({detected.system_name!r}) instead -- check autodetect.py's "
        "hostname table against this family table's keys"
    )
    assert detected.system_name == hostname


if __name__ == "__main__":
    test_system()
    test_autodetect_el_capitan()
    test_autodetect_sierra()
    test_autodetect_generic()
