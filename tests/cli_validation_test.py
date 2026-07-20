# Copyright (c) 2014-2025, Lawrence Livermore National Security, LLC.
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
"""
CLI validation and usability regression tests (review findings H1-H5).

- H1: a missing command produces a clean validation error instead of a
      ``TypeError`` crash deep in the scheduler.
- H2: the ``-x/--xargs`` override grammar (undashed keys normalized to dashed,
      attached forms verbatim, ``~key`` removal, values containing ``=``).
- H3: ``--out/--err/-o/--save-hostlist`` are accepted when the run gets a launch
      directory (torchrun-hpc always does), but still rejected for a genuinely
      ephemeral blocking ``launch``.
- H4: re-running ``-l . -o name`` regenerates the script in place instead of
      raising ``shutil.SameFileError``.
- H5: a relative-path argument that would not resolve from the launch directory
      produces a warning.
"""
import argparse
import os
import subprocess
import sys

import pytest

from hpc_launcher.cli import common_args

from conftest import require_torch


LAUNCH = [sys.executable, "-m", "hpc_launcher.cli.launch"]
TORCHRUN = [sys.executable, "-m", "hpc_launcher.cli.torchrun_hpc"]


def _override_parser() -> argparse.ArgumentParser:
    """A parser exposing just the common arguments, for exercising ``-x``."""
    parser = argparse.ArgumentParser()
    common_args.setup_arguments(parser)
    return parser


# ---------------------------------------------------------------------------
# H1 -- missing command is a clean error, not a TypeError crash
# ---------------------------------------------------------------------------
def test_missing_command_is_clean_error():
    """
    ``launch --local -N1 --dry-run`` with no command must fail with a clean,
    command-mentioning validation error rather than the ``TypeError:
    sequence item 0: expected str instance, NoneType found`` crash the dead
    guard used to allow through (finding H1).
    """
    proc = subprocess.run(
        LAUNCH + ["--local", "-N1", "--dry-run"], capture_output=True
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode != 0, f"expected failure, got 0\n{stderr}"
    assert "TypeError" not in stderr, f"crashed with a TypeError:\n{stderr}"
    assert "command" in stderr.lower(), (
        f"error did not mention the missing command:\n{stderr}"
    )


