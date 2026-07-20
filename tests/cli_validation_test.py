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


# ---------------------------------------------------------------------------
# H2 -- the -x/--xargs grammar
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tokens,expected",
    [
        # undashed multi-character key -> --key
        (["-x", "ntasks=8"], {"--ntasks": "8"}),
        # multiple space-separated tokens after a single -x
        (["-x", "k1=v1", "k2=v2"], {"--k1": "v1", "--k2": "v2"}),
        # removal: ~key -> normalized ~--key (empty value)
        (["-x", "~ntasks"], {"~--ntasks": ""}),
        # attached short form is kept verbatim
        (["-x--ntasks=8"], {"--ntasks": "8"}),
        # attached long form via = is kept verbatim
        (["--xargs=--ntasks=8"], {"--ntasks": "8"}),
        # value may contain '=' (split on the first '=' only)
        (["-x", "foo=a=b"], {"--foo": "a=b"}),
        # undashed single-character key -> -k
        (["-x", "q=pbatch"], {"-q": "pbatch"}),
    ],
)
def test_xargs_dashed_key_forms(tokens, expected):
    """The parsed ``override_args`` dict matches the decided ``-x`` grammar."""
    args = _override_parser().parse_args(tokens)
    assert args.override_args == expected


def test_xargs_bad_token_is_clean_error():
    """A token that is neither ``key=value`` nor ``~key`` is a clean error."""
    with pytest.raises(SystemExit):
        _override_parser().parse_args(["-x", "notakeyvalue"])


def test_xargs_flag_lands_in_generated_script(tmp_path):
    """
    End-to-end: an undashed ``-x`` override reaches the generated batch script
    with its normalized (dashed) spelling (finding H2). Uses ``--scheduler
    slurm --bg --setup-only`` so the override is emitted as a header directive
    and the internal run command, without submitting anything.
    """
    proc = subprocess.run(
        LAUNCH
        + [
            "--scheduler", "slurm",
            "-N1", "-n1",
            "--bg", "--setup-only",
            "-l", str(tmp_path),
            "-x", "myflag=myval",
            "--", "echo", "hi",
        ],
        capture_output=True,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, stderr
    script = (tmp_path / "launch.sh").read_text()
    assert "--myflag=myval" in script, (
        f"normalized override did not land in the script:\n{script}"
    )


