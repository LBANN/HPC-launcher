#!/bin/sh

set -e

# Install dependencies
pip install --upgrade twine

# Erase old distribution, if exists
rm -rf dist hpc_launcher.egg-info

# Make tarball
python -m build

# Upload to PyPI
twine upload dist/*
