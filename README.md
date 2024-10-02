# LBANN: Livermore Big Artificial Neural Network Toolkit

The Livermore Big Artificial Neural Network toolkit (LBANN) is an
open-source, HPC-centric, deep learning training framework that is
optimized to compose multiple levels of parallelism.

LBANN provides model-parallel acceleration through domain
decomposition to optimize for strong scaling of network training.  It
also allows for composition of model-parallelism with both data
parallelism and ensemble training methods for training large neural
networks with massive amounts of data.  LBANN is able to advantage of
tightly-coupled accelerators, low-latency high-bandwidth networking,
and high-bandwidth parallel file systems.

## HPC-launcher Repository

The HPC launcher repository contains a set of helpful scripts and
Python bindings for launching LBANN 2.0 (PyTorch-core) on multiple
leadership-class HPC systems.  There are optimized routines for FLUX,
SLURM, and LSF launchers.  Currently there are supported systems at:
 - LLNL Livermore Computing (LC)
 - LBL NERSC
 - ORNL OLCF
 - RIKEN
 
## Publications

A list of publications, presentations and posters are shown
[here](https://lbann.readthedocs.io/en/latest/publications.html).

## Reporting issues
Issues, questions, and bugs can be raised on the [Github issue
tracker](https://github.com/LBANN/lbann/issues).
