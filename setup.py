import os
from setuptools import find_packages, setup

with open("README.md", "r") as fp:
    long_description = fp.read()

with open(os.path.join("hpc_launcher", "version.py"), "r") as fp:
    version = fp.read().strip().split(' ')[-1][1:-1]

setup(
    name='hpc-launcher',
    version=version,
    url='https://github.com/LBANN/HPC-launcher',
    author='Lawrence Livermore National Laboratory',
    author_email='lbann@llnl.gov',
    description='LBANN Launcher utilities for distributed jobs on HPC clusters',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    entry_points={
        'console_scripts': [
            'torchrun-hpc = hpc_launcher.cli.torchrun_hpc:main',
            'launch = hpc_launcher.cli.launch:main',
        ],
    },
    install_requires=["psutil"],
    extras_require={
        'torch': ['torch', 'numpy'],
        'mpi': ['mpi4py>=3.1.4', 'mpi_rdv'],
        'testing': ['pytest'],
    },
)
