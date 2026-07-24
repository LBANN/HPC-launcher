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
CLI validation and usability regression tests.

- A missing command produces a clean validation error instead of a
  ``TypeError`` crash deep in the scheduler.
- The ``-x/--xargs`` override grammar (keys passed through verbatim, attached
  dashed forms, ``~key`` removal, values containing ``=``).
- ``--out/--err/-o/--save-hostlist`` are accepted when the run gets a launch
  directory (torchrun-hpc always does), but still rejected for a genuinely
  ephemeral blocking ``launch``.
- Re-running ``-l . -o name`` regenerates the script in place instead of
  raising ``shutil.SameFileError``.
- A relative-path argument that would not resolve from the launch directory
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
# Missing command is a clean error, not a TypeError crash
# ---------------------------------------------------------------------------
def test_missing_command_is_clean_error():
    """
    ``launch --local -N1 --dry-run`` with no command must fail with a clean,
    command-mentioning validation error rather than the ``TypeError:
    sequence item 0: expected str instance, NoneType found`` crash the dead
    guard used to allow through.
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
# The -x/--xargs grammar
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tokens,expected",
    [
        # undashed key is kept verbatim (no dashes added)
        (["-x", "ntasks=8"], {"ntasks": "8"}),
        # multiple space-separated tokens after a single -x
        (["-x", "k1=v1", "k2=v2"], {"k1": "v1", "k2": "v2"}),
        # removal: ~key is kept verbatim (empty value)
        (["-x", "~--ntasks"], {"~--ntasks": ""}),
        # a dashed key can be removed space-separated (the ~ hides the dash
        # from argparse)
        (["-x", "~-nnodes"], {"~-nnodes": ""}),
        # attached short form is kept verbatim
        (["-x--ntasks=8"], {"--ntasks": "8"}),
        # attached long form via = is kept verbatim
        (["--xargs=--ntasks=8"], {"--ntasks": "8"}),
        # value may contain '=' (split on the first '=' only)
        (["-x", "foo=a=b"], {"foo": "a=b"}),
        # single-character key is also kept verbatim
        (["-x", "q=pbatch"], {"q": "pbatch"}),
    ],
)
def test_xargs_verbatim_key_forms(tokens, expected):
    """The parsed ``override_args`` dict matches the decided ``-x`` grammar."""
    args = _override_parser().parse_args(tokens)
    assert args.override_args == expected


def test_xargs_bad_token_is_clean_error():
    """A token that is neither ``key=value`` nor ``~key`` is a clean error."""
    with pytest.raises(SystemExit):
        _override_parser().parse_args(["-x", "notakeyvalue"])


@pytest.mark.parametrize(
    "override_tokens,expected",
    [
        # attached dashed form: the exact flag spelling lands in the script
        (["-x--myflag=myval"], "#SBATCH --myflag=myval"),
        # undashed key: passed through verbatim, no dashes added
        (["-x", "myflag=myval"], "#SBATCH myflag=myval"),
    ],
)
def test_xargs_flag_lands_in_generated_script(tmp_path, override_tokens, expected):
    """
    End-to-end: an ``-x`` override reaches the generated batch script with the
    exact spelling the user gave. Uses ``--scheduler slurm --bg --setup-only``
    so the override is emitted as a header directive and the internal run
    command, without submitting anything.
    """
    proc = subprocess.run(
        LAUNCH
        + [
            "--scheduler", "slurm",
            "-N1", "-n1",
            "--bg", "--setup-only",
            "-l", str(tmp_path),
        ]
        + override_tokens
        + ["--", "echo", "hi"],
        capture_output=True,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, stderr
    script = (tmp_path / "launch.sh").read_text()
    assert expected in script, (
        f"override did not land verbatim in the script:\n{script}"
    )


# ---------------------------------------------------------------------------
# out/err/etc. allowed when a launch directory is (auto-)provided, but
# still rejected for a genuinely ephemeral blocking launch.
# ---------------------------------------------------------------------------
def test_out_err_allowed_with_auto_launch_dir(tmp_path):
    """
    ``torchrun-hpc`` always runs from a launch directory (it auto-defaults
    ``-l``), so ``--out`` must be accepted -- not rejected as an ephemeral
    interactive job. The genuinely ephemeral rejection must still
    fire for a blocking ``launch`` with no ``-l``.
    """
    require_torch()

    driver = tmp_path / "trivial.py"
    driver.write_text("print('hello')\n")

    # torchrun-hpc auto-provides a launch dir: --out is accepted.
    proc = subprocess.run(
        TORCHRUN
        + [
            "--local", "-N1", "-n1",
            "--setup-only",
            "--out", "t.log",
            str(driver),
        ],
        cwd=str(tmp_path),
        capture_output=True,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, (
        f"torchrun-hpc --out with an auto launch dir should succeed:\n{stderr}"
    )

    # A blocking launch with no -l is genuinely ephemeral: --out is rejected.
    proc = subprocess.run(
        LAUNCH + ["--local", "-N1", "-n1", "--out", "t.log", "echo", "hi"],
        cwd=str(tmp_path),
        capture_output=True,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode != 0, "ephemeral --out should have been rejected"
    assert "ephemeral" in stderr.lower(), (
        f"expected an ephemeral-job rejection message:\n{stderr}"
    )


# ---------------------------------------------------------------------------
# Re-running -l . -o name must not crash with SameFileError
# ---------------------------------------------------------------------------
def test_rerun_same_output_script(tmp_path):
    """
    Running ``launch -l . -o run.sh --setup-only`` twice in the same directory
    must regenerate the script in place rather than aborting with
    ``shutil.SameFileError`` on the second run.
    """
    cmd = LAUNCH + [
        "--local", "-N1", "-n1",
        "-l", ".",
        "-o", "run.sh",
        "--setup-only",
        "echo", "hi",
    ]

    for i in range(2):
        proc = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True)
        stderr = proc.stderr.decode(errors="replace")
        assert proc.returncode == 0, f"run {i} failed:\n{stderr}"
        assert "SameFileError" not in stderr, f"run {i} hit SameFileError:\n{stderr}"

    assert (tmp_path / "run.sh").exists(), "the output script was not regenerated"


# ---------------------------------------------------------------------------
# Relative-path argument warning
# ---------------------------------------------------------------------------
def test_relative_arg_warning(tmp_path):
    """
    ``launch -l outdir python script.py`` where ``./script.py`` exists in the
    invocation directory but not under the launch directory must warn that the
    relative path will not resolve once the job cd's into the launch dir.
    """
    (tmp_path / "script.py").write_text("print(1)\n")

    proc = subprocess.run(
        LAUNCH
        + [
            "--local", "-N1", "-n1",
            "-l", "outdir",
            "--setup-only",
            "python", "script.py",
        ],
        cwd=str(tmp_path),
        capture_output=True,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, stderr
    assert "relative path" in stderr and "script.py" in stderr, (
        f"expected a relative-path warning for script.py:\n{stderr}"
    )


def test_no_relative_arg_warning_when_launch_dir_is_cwd(tmp_path):
    """
    The relative-path warning must not fire when the launch directory is
    the current directory (``-l .``): relative paths still resolve.
    """
    (tmp_path / "script.py").write_text("print(1)\n")

    proc = subprocess.run(
        LAUNCH
        + [
            "--local", "-N1", "-n1",
            "-l", ".",
            "--setup-only",
            "python", "script.py",
        ],
        cwd=str(tmp_path),
        capture_output=True,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, stderr
    assert "relative path" not in stderr, (
        f"unexpected relative-path warning when -l . :\n{stderr}"
    )
