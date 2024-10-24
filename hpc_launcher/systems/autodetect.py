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
import warnings
from hpc_launcher.systems.system import System, GenericSystem
from hpc_launcher.systems.lc.el_capitan_family import ElCapitan
#from hpc_launcher.systems.lc.cts2 import CTS2

"""Default settings for LC systems."""
import socket
import re

# Detect system
_system = re.sub(r'\d+', '', socket.gethostname())

# ==============================================
# Access functions
# ==============================================

def system():
    """Name of system.

    Hostname with trailing digits removed.

    """
    return _system

def autodetect_current_system() -> System:
    """
    Tries to detect the current system based on information such
    as the hostname and HPC center.
    """

    sys = system()
    if sys == 'tioga' or sys == 'tuolumne':
        return ElCapitan(sys)

    # if sys == 'ipa':
    #     return CTS2()

    # TODO: Try to find current system
    warnings.warn('Could not auto-detect current system, defaulting '
                  'to generic system')
    return GenericSystem()
