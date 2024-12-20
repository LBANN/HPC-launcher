def get_schedulers():
    from .local import LocalScheduler
    from .flux import FluxScheduler
    from .slurm import SlurmScheduler
    from .lsf import LSFScheduler

    return {
        'local': LocalScheduler,
        'flux': FluxScheduler,
        'FluxScheduler': FluxScheduler,
        'slurm': SlurmScheduler,
        'SlurmScheduler': SlurmScheduler,
        'lsf': LSFScheduler,
        'LSFScheduler': LSFScheduler,
    }
