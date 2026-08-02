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
Regression tests for the LSF backend.

An earlier round of fixes covered:

- bsub flags were built as single dict keys with an embedded space
  (``f"-nnodes {n}"``), which both launch paths execute as argv *without* a
  shell, so bsub saw the single literal token ``-nnodes 2`` instead of
  ``-nnodes`` and ``2`` as separate arguments. The ``-W`` value also had a
  trailing newline baked into the (buggy) embedded key.
- bsub-only flags lived in ``common_launch_args``, which leaked them
  into the internal ``jsrun`` line written into the batch script.
- ``LSFScheduler.get_job_id`` raised ``NotImplementedError`` instead of
  parsing bsub's "Job <N> is submitted..." output.

The defects below were all still live while this file passed 4/4, because
every test above uses a ``GenericSystem`` (whose ``environment_variables()``
is empty, so ``cli_env_arg`` is never called) and deletes ``LSB_HOSTS`` (so
the "already inside an allocation" launch command is never constructed). The
two stubs added below -- ``EnvStubSystem`` and the ``lsb_hosts`` fixture --
exist specifically to close those two holes:

- **The ``--env`` token.** ``cli_env_arg`` baked the flag *and* its value
  into a single dict key (``--env "ALL, A=1, B=2"`` with a ``None`` value).
  That is the same tokenization bug as above, at a site the earlier fix
  never reached, so bsub received one long argv element full of spaces,
  commas and literal double quotes.
- **``--chdir`` instead of ``-cwd``.** The blocking path set the working
  directory with ``--chdir``, which is a jsrun option; bsub spells it
  ``-cwd``, as this file's own batch path already did for the identical
  value.
- **bsub-only flags on jsrun.** ``launch_command`` appends
  ``submit_only_args`` unconditionally, but when ``LSB_HOSTS`` is set the
  blocking launch command is ``jsrun``, not ``bsub``. Every bsub-only flag
  therefore landed on jsrun -- unconditionally, so this broke the primary
  interactive Lassen workflow 100% of the time.
- **No jsrun at all.** The ephemeral blocking path (no launch directory,
  hence no launch script to put the internal run command in) never inserted
  a ``jsrun``, so an 8-rank request silently ran the command once. This one
  is masked by the ``--env`` bug -- until that token is fixed the malformed
  ``--env`` makes the same command a hard bsub error -- which is why the two
  are fixed together.

No torch or scheduler binaries needed. Commands and
scripts are constructed directly against stub systems.
"""
import pytest

from hpc_launcher.schedulers.lsf import LSFScheduler
from hpc_launcher.systems.system import GenericSystem

# bsub-only flags that must never reach jsrun (they live in
# submit_only_args, not common_launch_args/run_only_args).
BSUB_ONLY_FLAGS = ("-nnodes", "-q", "-J", "--shared-launch", "-W", "-G", "-cwd", "-env")

# jsrun flags the scheduler is expected to still emit on the internal run
# line (unaffected by the bsub-flag restructuring).
JSRUN_FLAGS = (
    "--nrs",
    "--rs_per_host",
    "--tasks_per_rs",
    "--launch_distribution",
    "--cpu_per_rs",
    "--gpu_per_rs",
)


class EnvStubSystem(GenericSystem):
    """
    A stub system that actually *has* environment variables.

    ``conftest.py``'s ``stub_system`` is a bare ``GenericSystem``, whose
    ``environment_variables()`` returns ``[]``. ``cli_env_arg`` -- the code
    path that folds environment variables onto a blocking launch command --
    is only called when that list is non-empty, so with the shared fixture it
    is dead code and the malformed ``--env`` token it produced was invisible
    to the tests. The values mirror the real Sierra list closely
    enough to be representative while staying independent of
    ``sierra_family.py``.
    """

    SYSTEM_ENV = [
        ("IBV_FORK_SAFE", 1),
        ("HCOLL_ENABLE_SHARP", 0),
        ("NVSHMEM_MPI_LIB_NAME", "libmpi_ibm.so"),
    ]
    PASSTHROUGH_ENV = [("MY_PASSTHROUGH", "42")]

    def environment_variables(self) -> list[tuple[str, str]]:
        return list(self.SYSTEM_ENV) + list(self._aux_env_list)

    def passthrough_environment_variables(self) -> list[tuple[str, str]]:
        return list(self.PASSTHROUGH_ENV)


@pytest.fixture
def no_lsb_hosts(monkeypatch):
    """
    Force the "not already inside an LSF allocation" code paths so command
    construction is deterministic regardless of the host running the test.
    """
    monkeypatch.delenv("LSB_HOSTS", raising=False)


@pytest.fixture
def lsb_hosts(monkeypatch):
    """
    Force the "already inside an LSF allocation" code paths -- the standard
    Lassen workflow of running ``launch``/``torchrun-hpc`` from inside an
    ``lalloc``/``bsub -Is`` shell. In this state the blocking launch command
    is ``jsrun``, not ``bsub``.
    """
    monkeypatch.setenv("LSB_HOSTS", "host1 host1 host2 host2")


def _make_scheduler(**kwargs):
    params = dict(
        nodes=2,
        procs_per_node=4,
        gpus_per_proc=1,
        job_name="myjob",
        queue="pbatch",
        time_limit=90,
    )
    params.update(kwargs)
    return LSFScheduler(**params)


def _assert_argv_is_tokenized(cmd):
    """
    Shared invariant for every LSF argv: both launch paths ``exec`` these
    lists without a shell, so nothing ever splits an element. A flag and its
    value must therefore be separate elements, and no element may smuggle in
    shell syntax (quotes) that no parser will ever remove.
    """
    for token in cmd:
        assert " " not in token, f"argv token contains a space: {token!r} in {cmd}"
        assert "\n" not in token, f"argv token contains a newline: {token!r} in {cmd}"
        assert '"' not in token, f"argv token contains a quote: {token!r} in {cmd}"


def _values_after(cmd, flag):
    """All argv elements that immediately follow an occurrence of ``flag``."""
    return [cmd[i + 1] for i, tok in enumerate(cmd) if tok == flag and i + 1 < len(cmd)]


@pytest.mark.parametrize(
    "make_system, cli_env_only",
    [(GenericSystem, False), (EnvStubSystem, True)],
    ids=["no-env-vars", "with-env-vars"],
)
def test_bsub_argv_tokens_are_split(make_system, cli_env_only, no_lsb_hosts):
    """
    ``launch_command`` must return the flag and its value as adjacent,
    separate argv elements (e.g. ``..., "-nnodes", "2", ...``), and no
    element may contain a space or a newline (the ``-W`` value used to have
    a trailing ``\\n`` baked in).

    The ``with-env-vars`` parametrization is the later addition: it drives
    ``cli_env_arg``, which built ``--env "ALL, A=1, B=2"`` as a single dict
    *key* and so re-introduced exactly the bug this test was written to
    prevent, at a site the shared ``GenericSystem`` fixture could not reach.
    """
    system = make_system()

    for blocking in (True, False):
        scheduler = _make_scheduler()
        cmd = scheduler.launch_command(system, blocking=blocking, cli_env_only=cli_env_only)

        _assert_argv_is_tokenized(cmd)

        expected_pairs = {
            "-nnodes": "2",
            "-J": "myjob",
            "-q": "pbatch",
            "-W": "1:30",
        }
        for flag, value in expected_pairs.items():
            assert flag in cmd, f"{flag} missing from {cmd} (blocking={blocking})"
            idx = cmd.index(flag)
            assert cmd[idx + 1] == value, (
                f"{flag} not immediately followed by {value!r} in {cmd} "
                f"(blocking={blocking})"
            )

        # --shared-launch is a bare flag (no value).
        assert "--shared-launch" in cmd


def test_bsub_env_is_a_flag_and_value_pair(no_lsb_hosts):
    """
    The ``--env`` half of the tokenization bug. On the ephemeral blocking
    path the system's
    environment variables are moved onto the bsub command line. bsub's option
    is ``-env`` (single dash) and takes its comma-separated list as a
    *separate* argv element; the quotes usually seen around it belong to the
    shell, which is not involved here.
    """
    scheduler = _make_scheduler()
    cmd = scheduler.launch_command(EnvStubSystem(), blocking=True, cli_env_only=True)

    _assert_argv_is_tokenized(cmd)
    assert not any(
        tok.startswith("--env") for tok in cmd
    ), f"bsub has no --env option: {cmd}"

    values = _values_after(cmd, "-env")
    assert len(values) == 1, f"expected exactly one -env value in {cmd}"
    env_value = values[0]

    # "ALL" (inherit the submitting environment) plus one NAME=value entry per
    # variable, comma separated, all inside one argv element.
    entries = env_value.split(",")
    assert entries[0] == "ALL", f"-env list must start with ALL: {env_value!r}"
    assert "IBV_FORK_SAFE=1" in entries
    assert "HCOLL_ENABLE_SHARP=0" in entries
    assert "NVSHMEM_MPI_LIB_NAME=libmpi_ibm.so" in entries
    # cli_env_arg is called twice per launch (system env vars, then
    # passthrough vars); the second call must merge into the first entry
    # rather than emitting a second -env.
    assert "MY_PASSTHROUGH=42" in entries


def test_blocking_bsub_uses_cwd_not_chdir(no_lsb_hosts, tmp_path):
    """
    ``bsub`` spells the working directory ``-cwd``; ``--chdir`` is a jsrun
    option. This file's non-blocking path
    already emitted ``#BSUB -cwd`` for the identical value, so the blocking
    path emitting ``--chdir`` to the same program could not also be right.

    ``torchrun-hpc`` always forces a launch directory, so ``work_dir`` is
    always set there and every interactive Sierra-family run hit this.
    """
    for blocking in (True, False):
        scheduler = _make_scheduler(work_dir=str(tmp_path))
        cmd = scheduler.launch_command(GenericSystem(), blocking=blocking)

        assert "--chdir" not in cmd, f"bsub has no --chdir option: {cmd}"
        assert not any(
            tok.startswith("--chdir") for tok in cmd
        ), f"bsub has no --chdir option: {cmd}"

        if blocking:
            # Non-blocking puts -cwd in the batch script header, which
            # launch_command discards; blocking must carry it on the argv.
            assert _values_after(cmd, "-cwd") == [str(tmp_path)], (
                f"expected -cwd {tmp_path} on the bsub command line: {cmd}"
            )

    # ... and the batch script still carries the #BSUB -cwd directive.
    scheduler = _make_scheduler(work_dir=str(tmp_path))
    script = scheduler.launcher_script(
        GenericSystem(), "python", ["train.py"], blocking=False, launch_dir=str(tmp_path)
    )
    assert any(
        line.startswith("#BSUB -cwd") for line in script.splitlines()
    ), f"missing '#BSUB -cwd' directive in:\n{script}"


def test_jsrun_launch_command_has_no_bsub_flags(lsb_hosts, tmp_path):
    """
    Inside an existing allocation the blocking launch command is ``jsrun``,
    but ``launch_command`` appended ``submit_only_args`` -- the bucket an
    earlier fix moved every bsub-only flag *into* -- before ever asking
    which program it was building a command for. ``-nnodes`` and
    ``--shared-launch`` are added unconditionally, so this fired on every
    interactive run from inside an ``lalloc`` shell.

    The matching launch script correctly contains no internal jsrun line in
    this configuration, so this outer jsrun is the only task launcher and has
    to be right.
    """
    scheduler = _make_scheduler(work_dir=str(tmp_path), account="myacct")
    cmd = scheduler.launch_command(GenericSystem(), blocking=True)

    assert cmd[0] == "jsrun", f"expected a jsrun command inside an allocation: {cmd}"
    _assert_argv_is_tokenized(cmd)

    for flag in BSUB_ONLY_FLAGS:
        assert flag not in cmd, f"bsub-only flag {flag!r} leaked onto jsrun: {cmd}"
        assert not any(
            tok.startswith(f"{flag}=") for tok in cmd
        ), f"bsub-only flag {flag!r} leaked onto jsrun: {cmd}"
    assert "myjob" not in cmd
    assert "pbatch" not in cmd
    assert "myacct" not in cmd

    # The genuine jsrun options are still there...
    for flag in JSRUN_FLAGS:
        assert any(
            tok == flag or tok.startswith(f"{flag}=") for tok in cmd
        ), f"expected jsrun flag {flag!r} missing from: {cmd}"
    # ... including the working directory, under jsrun's spelling of it.
    assert f"--chdir={tmp_path}" in cmd or _values_after(cmd, "--chdir") == [
        str(tmp_path)
    ], f"expected the launch directory on jsrun's --chdir: {cmd}"


def test_jsrun_launch_command_carries_env_vars(lsb_hosts):
    """
    The interaction between the two fixes above: suppressing the bsub-only
    flags on the jsrun command must not silently drop the environment
    variables that share that bucket -- on the ephemeral path they are the
    only way the system's settings (IBV_FORK_SAFE and friends) reach the
    job. jsrun's environment option is ``-E``, one ``NAME=value`` per
    occurrence.
    """
    scheduler = _make_scheduler()
    cmd = scheduler.launch_command(EnvStubSystem(), blocking=True, cli_env_only=True)

    assert cmd[0] == "jsrun"
    assert _values_after(cmd, "-E") == [
        "IBV_FORK_SAFE=1",
        "HCOLL_ENABLE_SHARP=0",
        "NVSHMEM_MPI_LIB_NAME=libmpi_ibm.so",
        "MY_PASSTHROUGH=42",
    ], f"expected one -E NAME=value pair per environment variable: {cmd}"
    _assert_argv_is_tokenized(cmd)


def test_ephemeral_blocking_command_inserts_jsrun(no_lsb_hosts):
    """
    With no launch directory there is no launch script, so
    ``launcher_script`` -- the only caller of
    ``require_parallel_internal_run_command``/``internal_script_run_command``
    -- never runs and the task-launch step simply disappeared:
    ``launch -N2 -n4 -- python train.py`` asked for 8 ranks and ran
    ``python train.py`` exactly once. Slurm and Flux both produce full 8-way
    parallelism for the identical request.

    The fix splices the internal run command into the argv the same way the
    script does: ``bsub -Is <bsub flags> jsrun <jsrun flags> <command>``.
    """
    scheduler = _make_scheduler()
    cmd = scheduler.launch_command(EnvStubSystem(), blocking=True, cli_env_only=True)

    assert cmd[:2] == ["bsub", "-Is"], f"expected an interactive bsub: {cmd}"
    # Checked before the tokenization invariant on purpose: the missing jsrun
    # is *masked* by the malformed --env token, so while that token is still
    # present this command is a hard bsub error rather than a silent
    # one-rank run.
    assert "jsrun" in cmd, (
        f"no jsrun in the ephemeral blocking command -- the job would run "
        f"once instead of {scheduler.nodes * scheduler.procs_per_node} times: {cmd}"
    )
    _assert_argv_is_tokenized(cmd)

    jsrun_idx = cmd.index("jsrun")
    before, after = cmd[:jsrun_idx], cmd[jsrun_idx + 1:]

    # bsub's own flags belong to bsub, ahead of the command it runs.
    assert "-nnodes" in before and "--shared-launch" in before
    assert "-env" in before
    # ... and jsrun's belong to jsrun, behind it.
    for flag in JSRUN_FLAGS:
        assert any(
            tok == flag or tok.startswith(f"{flag}=") for tok in after
        ), f"jsrun flag {flag!r} not after the jsrun token: {cmd}"
    for flag in BSUB_ONLY_FLAGS:
        assert flag not in after and not any(
            tok.startswith(f"{flag}=") for tok in after
        ), f"bsub-only flag {flag!r} passed to jsrun: {cmd}"
    # The requested rank count has to survive to the program that can act on it.
    assert "--nrs=2" in after and "--tasks_per_rs=4" in after


def test_launch_folder_blocking_command_does_not_double_launch(no_lsb_hosts, tmp_path):
    """
    Guard on the fix above rather than a reproducer: when there *is* a launch
    directory the generated script already contains the internal jsrun line,
    so the outer bsub command must not add a second one (which would nest
    jsrun inside jsrun).
    """
    scheduler = _make_scheduler(work_dir=str(tmp_path))
    cmd = scheduler.launch_command(GenericSystem(), blocking=True)
    assert "jsrun" not in cmd, f"the launch script already runs jsrun: {cmd}"

    scheduler = _make_scheduler(work_dir=str(tmp_path))
    script = scheduler.launcher_script(
        GenericSystem(), "python", ["train.py"], blocking=True, launch_dir=str(tmp_path)
    )
    jsrun_lines = [
        line
        for line in script.splitlines()
        if "jsrun" in line and not line.lstrip().startswith("#")
    ]
    assert len(jsrun_lines) == 1, (
        f"expected exactly one internal jsrun line, found {len(jsrun_lines)}:\n{script}"
    )


def test_jsrun_line_has_no_bsub_flags(stub_system, tmp_path, no_lsb_hosts):
    """
    The internal ``jsrun`` line written into the batch script by
    ``launcher_script`` must not contain any bsub-only flag, and must
    contain the genuine jsrun flags.
    """
    scheduler = _make_scheduler()

    script = scheduler.launcher_script(
        stub_system,
        "python",
        ["train.py"],
        blocking=False,
        launch_dir=str(tmp_path),
    )

    jsrun_lines = [
        line
        for line in script.splitlines()
        if "jsrun" in line and not line.lstrip().startswith("#")
    ]
    assert len(jsrun_lines) == 1, (
        f"expected exactly one internal jsrun line, found {len(jsrun_lines)}:\n{script}"
    )
    jsrun_line = jsrun_lines[0]

    for flag in BSUB_ONLY_FLAGS:
        assert (
            f" {flag} " not in f" {jsrun_line} "
        ), f"bsub-only flag {flag!r} leaked into the jsrun line: {jsrun_line}"

    for flag in JSRUN_FLAGS:
        assert flag in jsrun_line, f"expected jsrun flag {flag!r} missing from: {jsrun_line}"

    # The job name (a submit-only value) must not have leaked into the run
    # line either -- it only belongs on the #BSUB header line.
    assert "myjob" not in jsrun_line

    # Sanity check: the job name *is* present, quoted, on a #BSUB directive.
    directive_lines = [
        line for line in script.splitlines() if line.startswith("#BSUB")
    ]
    assert any("myjob" in line for line in directive_lines)


def test_get_job_id_parses_bsub_output():
    scheduler = LSFScheduler(nodes=1, procs_per_node=1, gpus_per_proc=0)
    output = "Job <123> is submitted to default queue <pbatch>.\n"
    assert scheduler.get_job_id(output) == "123"


def test_get_job_id_returns_none_on_garbage():
    scheduler = LSFScheduler(nodes=1, procs_per_node=1, gpus_per_proc=0)
    assert scheduler.get_job_id("garbage output, no job id here") is None
    assert scheduler.get_job_id("") is None
