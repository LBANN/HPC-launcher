def get_schedulers():
    from .local import LocalScheduler
    from .flux import FluxScheduler

    return {
        'local': LocalScheduler,
        'flux': FluxScheduler,
        'FluxScheduler': FluxScheduler,
    }
