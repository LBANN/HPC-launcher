import os
from setuptools import find_packages, setup

with open("README.md", "r") as fp:
    long_description = fp.read()

with open(os.path.join("hpc_launcher", "version.py"), "r") as fp:
    version = fp.read().strip().split(" ")[-1][1:-1]

# GPU vendor libraries (amdsmi, nvidia-ml-py) are optional -- both call
# sites (hpc_launcher/systems/autodetect.py's find_AMD_gpus/find_NVIDIA_gpus)
# already guard the import and degrade gracefully at runtime. They must
# NOT be computed by probing *this* (build) machine's installed GPU
# libraries: doing so made the same commit produce a different wheel
# depending on whether it happened to be built on an AMD node, an NVIDIA
# node, or a CPU-only CI runner, which breaks build reproducibility and
# is a landmine for air-gapped/private-mirror installs that don't carry
# the hardware-mismatched package. They belong in extras_require, listed
# unconditionally, exactly like the torch/mpi/testing groups below --
# users opt in with `pip install hpc-launcher[rocm]` / `[cuda]`.
setup(
    name="hpc-launcher",
    version=version,
    license="Apache-2.0",
    url="https://github.com/LBANN/HPC-launcher",
    author="Lawrence Livermore National Laboratory",
    author_email="lbann@llnl.gov",
    description="LBANN Launcher utilities for distributed jobs on HPC clusters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    entry_points={
        "console_scripts": [
            "torchrun-hpc = hpc_launcher.cli.torchrun_hpc:main",
            "launch = hpc_launcher.cli.launch:main",
        ],
    },
    install_requires=["psutil"],
    extras_require={
        "torch": ["torch", "numpy"],
        "mpi": ["mpi4py>=3.1.4", "mpi_rdv"],
        "testing": ["pytest"],
        "e2e_testing": ["accelerate"],
        "rocm": ["amdsmi"],
        "cuda": ["nvidia-ml-py"],
    },
)
