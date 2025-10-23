import os
from setuptools import find_packages, setup
import ctypes.util
import re

def get_rocm_version():
    """Detect installed ROCm version."""
    # Try reading from ROCm installation
    rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
    version_file = os.path.join(rocm_path, '.info', 'version')

    if os.path.exists(version_file):
        with open(version_file) as f:
            version = f.read().strip()
            # Extract major.minor.patch
            match = re.match(r'(\d+\.\d+.\d+)', version)
            if match:
                return match.group(1)

    return None

with open("README.md", "r") as fp:
    long_description = fp.read()

with open(os.path.join("hpc_launcher", "version.py"), "r") as fp:
    version = fp.read().strip().split(" ")[-1][1:-1]

extras = []
path = ctypes.util.find_library("amdhip64")
if path:
    rocm_version = get_rocm_version()
    if rocm_version:
        # Constrain ROCm-dependent packages
        major, minor, patch = rocm_version.split('.')
        extras.append(f"amdsmi=={major}.{minor}.{patch}")
    else:
        # Fallback or raise error
        raise RuntimeError("ROCm installation not found!")

path = ctypes.util.find_library("cudart")
if path:
    extras.append("nvidia-ml-py")

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
    install_requires=["psutil"] + extras,
    extras_require={
        "torch": ["torch", "numpy"],
        "mpi": ["mpi4py>=3.1.4", "mpi_rdv"],
        "testing": ["pytest"],
        "e2e_testing": ["accelerate"],
    },
)
