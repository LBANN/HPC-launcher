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
from hpc_launcher.systems.system import System, GenericSystem
from hpc_launcher.systems.lc.el_capitan_family import ElCapitan
from hpc_launcher.systems.lc.cts2 import CTS2
from hpc_launcher.systems.lc.sierra_family import Sierra
import logging
import socket
import re

logger = logging.getLogger(__name__)

# Detect system lazily
_system = None

# ==============================================
# Access functions
# ==============================================


def system():
    """Name of system.

    Hostname with trailing digits removed.

    """
    global _system
    if _system is None:
        _system = re.sub(r'\d+', '', socket.gethostname())
    return _system


def clear_autodetected_system():
    """
    Clears the autodetected system. Used for testing.
    """
    global _system
    _system = None


def autodetect_current_system(quiet: bool = False) -> System:
    """
    Tries to detect the current system based on information such
    as the hostname and HPC center.
    """

    sys = system()
    if sys in ('tioga', 'tuolumne', 'elcap'):
        return ElCapitan(sys)

    if sys == 'ipa':
        return CTS2(sys)

    if sys == 'lassen' or sys == 'sierra' or sys == 'rzadams':
        return Sierra(sys)

    # TODO(later): Try to find current system via other means

    if not quiet:
        logger.warning('Could not auto-detect current system, defaulting '
                       'to generic system')

    return GenericSystem()
