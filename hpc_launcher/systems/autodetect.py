import warnings
from hpc_launcher.systems.system import System, GenericSystem
from hpc_launcher.systems.lc.el_capitan_family import ElCapitan

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
    if sys == 'tioga':
        return ElCapitan()

    # TODO: Try to find current system
    warnings.warn('Could not auto-detect current system, defaulting '
                  'to generic system')
    return GenericSystem()

