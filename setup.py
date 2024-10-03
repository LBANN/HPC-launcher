import glob
import os
import shutil
import subprocess

from setuptools import find_packages, setup

with open("README.md", "r") as fp:
    long_description = fp.read()

with open(os.path.join("hpc_launcher", "version.py"), "r") as fp:
    version = fp.read().strip().split(' ')[-1][1:-1]

setup(name='hpc-launcher',
      version=version,
      url='https://github.com/LBANN/HPC-launcher',
      author='Lawrence Livermore National Laboratory',
      author_email='lbann@llnl.gov',
      description='Launcher utilities for distributed jobs on HPC clusters',
      long_description=long_description,
      long_description_content_type='text/markdown',
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
      ],
      python_requires='>=3.9, <3.13',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      entry_points={
          'console_scripts': [
              'torchrun-hpc = hpc_launcher.cli.torchrun:main',
              'launch = hpc_launcher.cli.launch:main',
          ],
      }
)
