# Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
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
from unittest.mock import patch
from hpc_launcher.systems.lc.el_capitan_family import ElCapitan
from hpc_launcher.systems.system import GenericSystem
from hpc_launcher.systems.autodetect import system, autodetect_current_system


@patch('socket.gethostname', return_value='linux123')
def test_system(mock_gethostname):
    assert system() == 'linux'


@patch('socket.gethostname', return_value='tuolumne0001')
def test_autodetect(mock_gethostname):
    assert isinstance(autodetect_current_system(), ElCapitan)


@patch('socket.gethostname', return_value='linux')
def test_autodetect_generic(mock_gethostname):
    assert system() == 'linux'
    assert isinstance(autodetect_current_system(), GenericSystem)


if __name__ == '__main__':
    test_system()
    test_autodetect()
    test_autodetect_generic()
