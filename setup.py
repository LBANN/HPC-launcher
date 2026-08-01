import os
import re
from setuptools import find_packages, setup

def get_torch_rocm_version():
    """ROCm version the PyTorch in this environment was built against.

    Returns None when torch is absent or is a CPU/CUDA build, so the
    caller can leave amdsmi unpinned.
    """
    try:
        from importlib.metadata import version as pkg_version
        torch_version = pkg_version("torch")
    except Exception:
        return None

    # ROCm wheels carry a local version tag, e.g. "2.4.1+rocm6.2"
    match = re.search(r'\+rocm(\d+\.\d+(?:\.\d+)?)', torch_version)
    if match:
        return match.group(1)

    # Source builds lack the tag; torch.version.hip is e.g.
    # "6.2.41133-dd7f9576" where only major.minor identify the ROCm release
    try:
        import torch
        hip_version = getattr(torch.version, "hip", None)
        if hip_version:
            match = re.match(r'\d+\.\d+', hip_version)
            if match:
                return match.group(0)
    except Exception:
        pass

    return None

def amdsmi_requirement():
    """Pin amdsmi to the ROCm release PyTorch uses, when known."""
    rocm_version = get_torch_rocm_version()
    if not rocm_version:
        return "amdsmi"

    parts = rocm_version.split('.')
    major, minor = int(parts[0]), int(parts[1])
    # Releases of amdsmi on PyPI lag the GitHub/ROCm releases, so for
    # ROCm >= 7 accept anything in the major up through torch's release
    if major >= 7:
        return f"amdsmi>={major},<{major}.{minor + 1}"
    if len(parts) >= 3:
        return f"amdsmi=={rocm_version}"
    return f"amdsmi=={major}.{minor}.*"

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
#
# The [rocm] extra does inspect the *Python environment* (not the
# hardware): amdsmi only talks to the ROCm runtime PyTorch loads, so if
# a ROCm build of torch is already installed the pin follows its ROCm
# release. With no torch, or a CPU/CUDA torch, amdsmi stays unpinned.
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
        "rocm": [amdsmi_requirement()],
        "cuda": ["nvidia-ml-py"],
    },
)
