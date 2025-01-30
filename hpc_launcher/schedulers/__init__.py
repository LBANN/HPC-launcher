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


def get_schedulers():
    from .local import LocalScheduler
    from .flux import FluxScheduler
    from .slurm import SlurmScheduler
    from .lsf import LSFScheduler

    return {
        None: LocalScheduler,
        "local": LocalScheduler,
        "LocalScheduler": LocalScheduler,
        "flux": FluxScheduler,
        "FluxScheduler": FluxScheduler,
        "slurm": SlurmScheduler,
        "SlurmScheduler": SlurmScheduler,
        "lsf": LSFScheduler,
        "LSFScheduler": LSFScheduler,
    }
