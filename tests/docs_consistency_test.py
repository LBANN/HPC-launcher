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
Mechanically checks documentation claims in ``launch_cli.md`` and
``torchrun-hpc_cli.md`` against the behavior they describe.

Each documented claim gets two kinds of test, and the distinction matters:

- A *code-behavior* test that pins down what the CLI actually does. These do
  not change: the fixes in this batch are doc-only (plus a help-string
  wording fix in ``common_args.py`` that changes no behavior), so these tests
  pass both before and after the fix -- they are characterization tests, not
  regression reproducers.
- A *doc-text* test that inspects the ``.md`` files (and, for the
  ``--save-hostlist`` claim, the argparse ``help=`` string) directly. These
  fail against the original, incorrect documentation and pass once the docs
  are corrected -- they are genuine reproducers for the documentation bug.
"""
import argparse
import os
import re
import subprocess
import sys

from hpc_launcher.cli import common_args

from conftest import require_torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCH = [sys.executable, "-m", "hpc_launcher.cli.launch"]
TORCHRUN = [sys.executable, "-m", "hpc_launcher.cli.torchrun_hpc"]

LAUNCH_CLI_MD = os.path.join(REPO_ROOT, "launch_cli.md")
TORCHRUN_HPC_CLI_MD = os.path.join(REPO_ROOT, "torchrun-hpc_cli.md")


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# --batch-script is documented for torchrun-hpc but cannot be invoked.
# ---------------------------------------------------------------------------
def test_batch_script_is_not_invocable_on_torchrun_hpc():
    """
    Code-behavior characterization (unaffected by the doc-only fix; passes
    before and after).

    ``torchrun_hpc.py`` declares its ``command`` positional with no
    ``nargs='?'`` (unlike ``launch.py``'s ``nargs='?', default=None``), so it
    is always mandatory. Combined with the shared
    ``common_args.validate_arguments`` check (``if args.batch_script and
    args.command: raise ValueError(...)``), there is no way to supply
    ``--batch-script`` to ``torchrun-hpc`` and have it accepted: giving both
    a command and ``--batch-script`` is a ``ValueError``, and omitting the
    command to avoid that is a hard argparse error ("required: command").
    This pins down that code-side behavior.
    """
    # --batch-script together with a command: rejected deep inside
    # validate_arguments with a raw ValueError/traceback, not a clean CLI
    # error -- but rejected all the same.
    proc = subprocess.run(
        TORCHRUN
        + ["--batch-script", "batch.sh", "-N1", "-n1", "--local", "--dry-run", "echo"],
        capture_output=True,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode != 0, "expected --batch-script + command to fail"
    assert "invalid combination" in stderr, (
        f"expected the batch-script/command conflict error:\n{stderr}"
    )

    # --batch-script alone, omitting the command to avoid the conflict above:
    # argparse itself refuses, since `command` is mandatory for torchrun-hpc.
    proc = subprocess.run(
        TORCHRUN + ["--batch-script", "batch.sh", "-N1", "-n1"],
        capture_output=True,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode == 2, f"expected an argparse usage error:\n{stderr}"
    assert "required" in stderr.lower() and "command" in stderr.lower(), (
        f"expected argparse to complain that 'command' is required:\n{stderr}"
    )


def test_torchrun_hpc_doc_does_not_advertise_batch_script():
    """
    Doc-text reproducer: fails against the original docs (which list
    ``--batch-script`` in both the synopsis and the options table for
    ``torchrun-hpc``, identically to the fully-functional ``launch`` version,
    with no caveat) and passes once the doc is corrected to match
    ``test_batch_script_is_not_invocable_on_torchrun_hpc`` above.
    """
    text = _read(TORCHRUN_HPC_CLI_MD)
    assert "--batch-script BATCH_SCRIPT" not in text, (
        "torchrun-hpc_cli.md synopsis still advertises --batch-script, "
        "which torchrun-hpc cannot actually accept"
    )
    assert not re.search(r"\|\s*`--batch-script`\s*\|", text), (
        "torchrun-hpc_cli.md options table still lists --batch-script as a "
        "supported option"
    )


# ---------------------------------------------------------------------------
# "--launch-dir not set + blocking job -> no files" is false for
# torchrun-hpc.
# ---------------------------------------------------------------------------
def test_torchrun_hpc_blocking_without_launch_dir_still_creates_a_folder(tmp_path):
    """
    Code-behavior characterization (unaffected by the doc-only fix).

    ``torchrun_hpc.py`` unconditionally defaults ``args.launch_dir = ""``
    whenever it is ``None``, regardless of ``--bg``, so a timestamped launch
    directory is always created -- even for a blocking run with neither
    ``-l`` nor ``--bg``. The doc's own callout says this explicitly
    ("torchrun-hpc always runs from a launch directory"); only the
    copy-pasted bullet list contradicted it.
    """
    require_torch()
    driver = tmp_path / "trivial.py"
    driver.write_text("print('hello')\n")

    before = set(os.listdir(tmp_path))
    proc = subprocess.run(
        TORCHRUN + ["--local", "-N1", "-n1", "--setup-only", str(driver)],
        cwd=str(tmp_path),
        capture_output=True,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, stderr

    new_entries = set(os.listdir(tmp_path)) - before
    assert any(name.startswith("torchrun_hpc-") for name in new_entries), (
        f"expected a blocking, -l/--bg-less torchrun-hpc run to still create "
        f"a torchrun_hpc-* launch folder; new entries were: {new_entries}"
    )


def test_launch_blocking_without_launch_dir_creates_no_files(tmp_path):
    """
    Contrast case, also a characterization test: the equivalent ``launch``
    invocation (blocking, no ``-l``, no ``--bg``, no ``--batch-script``)
    genuinely creates nothing, which is what the shared bullet list
    describes correctly for ``launch`` -- the bullet list is wrong for
    ``torchrun-hpc`` only.
    """
    before = set(os.listdir(tmp_path))
    proc = subprocess.run(
        LAUNCH + ["--local", "-N1", "-n1", "--setup-only", "echo", "hi"],
        cwd=str(tmp_path),
        capture_output=True,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, stderr
    assert set(os.listdir(tmp_path)) == before, (
        "a blocking `launch` with no -l/--bg should create no files"
    )


def test_torchrun_hpc_doc_launch_dir_bullets_do_not_claim_no_files_when_blocking():
    """
    Doc-text reproducer: the original torchrun-hpc_cli.md bullet list said
    "Not set + blocking job: Runs without creating files", directly
    contradicting both the actual behavior and the doc's own callout two
    paragraphs later. Fails before the fix, passes after.
    """
    text = _read(TORCHRUN_HPC_CLI_MD)
    assert "Not set + blocking job" not in text, (
        "torchrun-hpc_cli.md still carries the generic (and false, for "
        "torchrun-hpc) 'Not set + blocking job: Runs without creating "
        "files' bullet"
    )


# ---------------------------------------------------------------------------
# "--launch-dir not set + non-blocking -> current directory" contradicts the
# --bg row above it.
# ---------------------------------------------------------------------------
def test_launch_bg_without_launch_dir_uses_timestamped_dir_not_cwd(tmp_path):
    """
    Code-behavior characterization (unaffected by the doc-only fix).

    ``launch --bg`` with no ``-l`` hits ``args.launch_dir = ""`` (same as an
    explicit ``-l`` with no argument), which
    ``scheduler.create_launch_folder_name`` turns into a timestamped
    ``launch-<name>_<timestamp>_<uuid>/`` folder -- not the current
    directory. This matches the ``--bg`` row's "uses timestamped directory
    by default" and contradicts the old "Not set + non-blocking job:
    Creates launch file and logs in current directory" bullet.
    """
    before = set(os.listdir(tmp_path))
    proc = subprocess.run(
        LAUNCH + ["--scheduler", "slurm", "-N1", "-n1", "--bg", "--setup-only", "echo", "hi"],
        cwd=str(tmp_path),
        capture_output=True,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, stderr

    assert not (tmp_path / "launch.sh").exists(), (
        "launch.sh must not be written directly into cwd for a --bg run "
        "with no -l"
    )
    new_dirs = [
        name for name in (set(os.listdir(tmp_path)) - before)
        if (tmp_path / name).is_dir()
    ]
    assert any(name.startswith("launch-") for name in new_dirs), (
        f"expected a timestamped launch-* directory; new entries were: "
        f"{set(os.listdir(tmp_path)) - before}"
    )


def test_launch_doc_launch_dir_bullets_do_not_claim_cwd_for_non_blocking():
    """
    Doc-text reproducer: both launch_cli.md and torchrun-hpc_cli.md said
    "Not set + non-blocking job: Creates launch file and logs in current
    directory", contradicting the "--bg ... uses timestamped directory by
    default" row a few dozen lines above. Fails before the fix, passes
    after, in both files.
    """
    for path in (LAUNCH_CLI_MD, TORCHRUN_HPC_CLI_MD):
        text = _read(path)
        assert "Creates launch file and logs in current directory" not in text, (
            f"{path} still claims a non-blocking run with no -l lands in "
            f"the current directory, contradicting the --bg row"
        )


# ---------------------------------------------------------------------------
# -v/--verbose claims to save the hostlist; nothing implements it.
# ---------------------------------------------------------------------------
def test_verbose_alone_does_not_save_hostlist(tmp_path):
    """
    Code-behavior characterization (unaffected by the doc-only fix).

    Both entry points compute ``save_hostlist`` as
    ``args.launch_dir != None and args.save_hostlist``, with no reference to
    ``args.verbose`` at all. So ``-v`` alone must not emit the
    ``HPC_LAUNCHER_HOSTLIST`` export/echo that ``--save-hostlist`` does.
    """
    proc = subprocess.run(
        LAUNCH
        + [
            "--scheduler", "slurm", "-N1", "-n1", "--bg", "--setup-only",
            "-l", ".", "--verbose", "echo", "hi",
        ],
        cwd=str(tmp_path),
        capture_output=True,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, stderr
    script = (tmp_path / "launch.sh").read_text()
    assert "hostlist" not in script.lower(), (
        f"-v alone should not save the hostlist, but it did:\n{script}"
    )


def test_save_hostlist_flag_alone_does_save_hostlist(tmp_path):
    """
    Contrast case for the previous test: ``--save-hostlist`` by itself (no
    ``-v``) does produce the hostlist export and the
    ``echo ... > hpc_launcher_hostlist.txt`` line. Establishes that the two
    flags are genuinely independent in the code, which is exactly what the
    ``-v`` help string incorrectly claims is not the case.
    """
    proc = subprocess.run(
        LAUNCH
        + [
            "--scheduler", "slurm", "-N1", "-n1", "--bg", "--setup-only",
            "-l", ".", "--save-hostlist", "echo", "hi",
        ],
        cwd=str(tmp_path),
        capture_output=True,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, stderr
    script = (tmp_path / "launch.sh").read_text()
    assert "HPC_LAUNCHER_HOSTLIST" in script and "hpc_launcher_hostlist.txt" in script, (
        f"--save-hostlist alone should save the hostlist:\n{script}"
    )


def test_verbose_help_string_does_not_claim_to_save_hostlist():
    """
    Doc-text reproducer, at the source: ``-v/--verbose``'s ``help=`` string
    in ``common_args.py`` is the single authoritative source that feeds both
    ``-h`` output and the two markdown docs (they are generated from it).
    Fails against the original "Also save the hostlist as if
    --save-hostlist is set" wording; passes once that claim is removed.
    """
    parser = argparse.ArgumentParser()
    common_args.setup_arguments(parser)
    verbose_action = next(
        a for a in parser._actions if a.dest == "verbose"
    )
    assert "hostlist" not in verbose_action.help.lower(), (
        f"--verbose help string still claims to save the hostlist: "
        f"{verbose_action.help!r}"
    )


def test_cli_docs_do_not_claim_verbose_saves_hostlist():
    """
    Doc-text reproducer for the two markdown surfaces fed by the help
    string above. Fails against the original docs, passes after the fix.
    """
    for path in (LAUNCH_CLI_MD, TORCHRUN_HPC_CLI_MD):
        text = _read(path)
        assert "Also save the hostlist" not in text, (
            f"{path} still claims --verbose saves the hostlist"
        )
