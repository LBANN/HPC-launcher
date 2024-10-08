import warnings
from hpc_launcher.systems.system import System, GenericSystem
from hpc_launcher.systems.lc.el_capitan_family import ElCapitan

"""Default settings for LC systems."""
import socket
import re

# ==============================================
# Set system parameters
# ==============================================

class SystemParams:
    """Simple data structure to describe an LC system."""
    def __init__(self, cores_per_node, gpus_per_node, scheduler):
        self.cores_per_node = cores_per_node
        self.gpus_per_node = gpus_per_node
        self.scheduler = scheduler

# Supported LC systems
_system_params = {
    'corona':   SystemParams(48, 8, 'flux'),
    'lassen':   SystemParams(44, 4, 'lsf'),
    'pascal':   SystemParams(36, 2, 'slurm'),
    'rzansel':  SystemParams(44, 4, 'lsf'),
    'rzvernal': SystemParams(64, 8, 'flux'),
    'sierra':   SystemParams(44, 4, 'lsf'),
    'tioga':    SystemParams(64, 8, 'flux'),
}

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

