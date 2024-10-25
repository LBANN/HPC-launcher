from hpc_launcher.systems.system import System, GenericSystem
from hpc_launcher.systems.lc.el_capitan_family import ElCapitan
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


def autodetect_current_system() -> System:
    """
    Tries to detect the current system based on information such
    as the hostname and HPC center.
    """

    sys = system()
    if sys == 'tioga':
        return ElCapitan()

    # TODO: Try to find current system
    logger.warning('Could not auto-detect current system, defaulting '
                   'to generic system')
    return GenericSystem()
