import warnings
from hpc_launcher.systems.system import System, GenericSystem

def autodetect_current_system() -> System:
    """
    Tries to detect the current system based on information such
    as the hostname and HPC center.
    """
    # TODO: Try to find current system
    warnings.warn('Could not auto-detect current system, defaulting '
                  'to generic system')
    return GenericSystem()

