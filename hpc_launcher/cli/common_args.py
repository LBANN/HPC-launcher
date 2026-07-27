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
Common arguments for CLI utilities.
"""
import argparse
from hpc_launcher.schedulers import get_schedulers
from hpc_launcher.schedulers.scheduler import Scheduler
from hpc_launcher.systems.system import System, GenericSystem
from hpc_launcher.systems import autodetect, configure
import logging
import os

from dataclasses import fields
from typing import Optional

class ParseKVAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not getattr(namespace, self.dest):
            setattr(namespace, self.dest, dict())
        for each in values:
            try:
                key, value = each.split("=")
                getattr(namespace, self.dest)[key] = value
            except ValueError as ex:
                message = "\nTraceback: {}".format(ex)
                message += "\nError on '{}' || It should be 'key=value'".format(each)
                raise argparse.ArgumentError(self, str(message))


class ParseOverrideKVAction(argparse.Action):
    """
    Parse ``-x/--xargs`` override tokens into the ``{key: value}`` dictionary
    consumed as ``override_launch_args`` by the schedulers.

    Each token is one of:

    - ``key=value`` -> ``{key: value}``. The split is on the *first* ``=``
      only, so values may themselves contain ``=`` (``-x foo=a=b`` sets
      ``foo`` to ``a=b``). A trailing ``=`` (``key=``) yields an empty value,
      i.e. a bare flag.
    - ``~key`` -> ``{~key: ""}``, which removes ``key`` if the scheduler had
      set it.

    Keys are passed through verbatim, dashes included: overriding or removing
    a flag the scheduler sets requires its exact spelling (``--ntasks``,
    LSF's single-dash ``-nnodes``, ...). Because argparse refuses a
    space-separated token that starts with ``-``, dashed keys must use the
    attached forms ``-x--ntasks=8`` or ``--xargs=--ntasks=8`` (removal,
    ``-x ~--ntasks``, needs no such workaround).
    """

    def __call__(self, parser, namespace, values, option_string=None):
        if not getattr(namespace, self.dest):
            setattr(namespace, self.dest, dict())
        overrides = getattr(namespace, self.dest)
        for each in values:
            if "=" in each:
                key, value = each.split("=", 1)
            elif each.startswith("~"):
                key, value = each, ""
            else:
                raise argparse.ArgumentError(
                    self,
                    f"invalid override '{each}': expected 'key=value' to set a "
                    f"key or '~key' to remove one",
                )
            overrides[key] = value


# Values accepted by --comm-backend, matched case-insensitively (argparse's
# `type=str.upper` + `choices=` below normalize the case before comparing).
# Kept here, rather than inline in `setup_arguments`, so `resolve_comm_backend`
# can document the mapping against the same list.
#
# "*CCL" is accepted because the docs used to list it as an option verbatim
# ("Options: MPI, *CCL (NCCL, RCCL)"), so a user may well have typed it; it is
# also the internal spelling every consumer matches on. Now that unrecognized
# values are a usage error rather than a silent no-op, turning a previously
# working invocation into a hard failure would be a poor trade.
COMM_BACKEND_CHOICES = ("MPI", "NCCL", "RCCL", "*CCL")


def resolve_comm_backend(job_comm_protocol: Optional[str]) -> Optional[str]:
    """
    Normalize a validated ``--comm-backend`` value to what
    ``System.job_comm_protocol`` consumers understand.

    By the time this runs, argparse has already restricted and uppercased
    ``job_comm_protocol`` to one of ``COMM_BACKEND_CHOICES`` or left it
    ``None`` -- see the ``--comm-backend`` argument in ``setup_arguments``.
    The only consumer today (``ElCapitan.environment_variables``) matches
    ``"RCCL"`` or the generic ``"*CCL"`` marker case-insensitively; it has no
    notion of ``"NCCL"`` at all. The CLI's own help text already documents
    NCCL and RCCL as interchangeable spellings of "whichever collective
    library is native to this system's accelerators" ("MPI or *CCL (NCCL,
    RCCL)"), so both collapse to ``"*CCL"`` here. This is the one place
    shared by ``launch`` and ``torchrun-hpc`` (both route through
    ``process_arguments``), so the two CLIs agree on what a given
    ``--comm-backend`` value means instead of ``launch`` forwarding ``NCCL``
    verbatim into a consumer that silently ignores it.

    :param job_comm_protocol: The parsed ``--comm-backend`` value (already
                               validated/uppercased by argparse), or ``None``.
    :return: ``job_comm_protocol`` unchanged for ``"MPI"`` or ``None``;
             ``"*CCL"`` for ``"NCCL"``/``"RCCL"``.
    """
    if job_comm_protocol in (None, "MPI"):
        return job_comm_protocol
    return "*CCL"


def create_scheduler_arguments(**kwargs) -> dict[str, str]:
    cmdline_args = {}
    for field in fields(Scheduler):
        if field.name in kwargs:
            if kwargs[field.name] is not None:
                cmdline_args[field.name] = kwargs[field.name]

    return cmdline_args


def setup_arguments(parser: argparse.ArgumentParser):
    """
    Adds common arguments for CLI utilities.

    :param parser: The ``argparse`` parser of the tool.
    """
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Run in verbose mode, logging additional INFO-level messages "
        "about job setup and submission.",
    )

    # Job size arguments
    group = parser.add_argument_group(
        "Job size",
        "Determines the number of nodes, accelerators, and ranks for the job",
    )
    group.add_argument(
        "-N",
        "--nodes",
        type=int,
        default=0,
        help="Specifies the number of requested nodes",
    )
    group.add_argument(
        "-n",
        "--procs-per-node",
        type=int,
        default=None,
        help="Specifies the number of requested processes per node",
    )

    group.add_argument(
        "--gpus-per-proc",
        type=int,
        default=None,  # Internally, if there are GPUs, this will default to 1
        help="Specifies the number of requested GPUs per process (default: 1)",
    )

    group.add_argument("-q", "--queue", default=None, help="Specifies the queue to use")

    group.add_argument(
        "-t",
        "--time-limit",
        type=int,
        default=None,
        help="Set a time limit for the job in minutes",
    )

    # Constraints
    group.add_argument(
        "-g",
        "--gpus-at-least",
        type=int,
        default=0,
        help="Specifies the total number of accelerators requested. Mutually "
        'exclusive with "--procs-per-node" and "--nodes"',
    )

    group.add_argument(
        "--gpumem-at-least",
        type=int,
        default=0,
        help="A constraint that specifies how much accelerator "
        "memory is needed for the job (in gigabytes). If this "
        "flag is specified, the number of nodes and processes "
        "are not necessary. Requires the system to be "
        "registered with the launcher.",
    )

    group.add_argument(
        "--exclusive",
        action="store_true",
        default=False,
        help="Request exclusive access from the scheduler",
    )

    group.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Run locally (i.e., one process without a batch " "scheduler)",
    )

    group.add_argument(
        "--comm-backend",
        dest="job_comm_protocol",
        type=str.upper,
        choices=COMM_BACKEND_CHOICES,
        default=None,
        help="Indicate if the job will primarily use a specific communication protocol and set any relevant environment variables. Case-insensitive; one of MPI or *CCL (NCCL, RCCL). An unrecognized value is a "
        "usage error rather than a silent no-op, since the only consumer of "
        "this flag (the El Capitan RCCL/AWS-OFI setup) matches a closed set "
        "of spellings and otherwise skips its environment configuration "
        "without any other warning.",
    )

    group.add_argument(
        "-x",
        "--xargs",
        dest="override_args",
        nargs='+',
        action=ParseOverrideKVAction,
        help="Override scheduler/launch arguments (repeatable; may take several "
        "space-separated tokens: -x k1=v1 k2=v2). Each token is key=value to set "
        "a key or ~key to remove one. Keys are passed through verbatim, dashes "
        "included; overriding a flag the scheduler sets requires its exact "
        "spelling, via an attached form (-x--ntasks=8 or --xargs=--ntasks=8) "
        "since a space-separated token cannot start with '-'. Values may contain "
        "'=' (split on the first '=' only). Note a double dash -- is needed "
        "before the command if -x is the last option.",
        metavar="KEY=VALUE",
    )

    # Schedule
    group = parser.add_argument_group(
        "Schedule", "Arguments that determine when a job will run"
    )

    # Blocking
    group.add_argument(
        "--bg",
        action="store_true",
        default=False,
        help="If set, the job will be run in the background. Otherwise, the "
        "launcher will wait for the job to start and forward the outputs to "
        "the console.  Additionally, by default, it will run from a generated "
        "timestamped directory (which can be overridden by the -l flag).",
    )

    group.add_argument(
        "--batch-script",
        type=str,
        default="",
        help="Launch a user provided batch script",
    )

    group.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=get_schedulers().keys(),
        help="If set, overrides the default batch scheduler",
    )

    group = parser.add_argument_group("Script", "Batch scheduler script parameters")

    # different behavior for interactive vs batch jobs
    # Add an argument to pick the run directory: tmp, none, self labeled, auto labeled

    group.add_argument(
        "-l",
        "--launch-dir",
        dest="launch_dir",
        nargs="?",
        const="",
        # action="store_true",
        default=None,
        help="If set without argument, the launcher will create a timestamped launch directory. "
        "If set with an argument, the launcher will create a directory named [LAUNCH_DIR]. "
        "If set with argument == \".\", the launcher will create a launch script in the <cwd>. "
        "If not set, it will either run the command without creating any files if "
        "the job is blocking and if it is non-blocking it will create the launch "
        "file and logs in the current working "
        "directory. Also note that a double dash -- is need if this is the last argument",
    )

    group.add_argument(
        "-o",
        "--output-script",
        default=None,
        help="Output job setup script file. If not given, uses a temporary file",
    )

    group.add_argument(
        "--setup-only",
        action="store_true",
        default=False,
        help="If set, the launcher will only write the job setup script file, "
        "without scheduling it.",
    )

    group.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="If set, output the results of the launcher without any side-effects."
    )

    group.add_argument(
        "--account",
        default=None,
        help="Specify the account (or bank) to use fo the job",
    )

    group.add_argument(
        "--dependency",
        default=None,
        help="Specify a scheduler dependency of the submitted job.",
    )

    group.add_argument(
        "-J",
        "--job-name",
        default=None,
        help="Specify a name to use fo the job",
    )

    group.add_argument(
        "--reservation",
        default=None,
        help="Add a reservation arguement to scheduler.  "
        "Typically used for Dedecated Application Time runs (DATs)",
    )

    group.add_argument(
        "--save-hostlist",
        action="store_true",
        default=False,
        help="Write the hostlist to a file: hpc_launcher_hostlist.txt.",
    )

    # System
    group = parser.add_argument_group(
        "System",
        "Provide system parameters from the CLI -- overrides built-in system descriptions and autodetection",
    )
    group.add_argument(
        "-p",
        "--system-params",
        dest="system_params",
        nargs='+',
        action=ParseKVAction,
        help="Specifies some or all of the parameters of a system as a dictionary (note it will override any known or autodetected parameters): -p cores_per_node=<int> gpus_per_node=<int> gpu_arch=<str> mem_per_gpu=<float> numa_domains=<int> scheduler=<str>\n -p cores_per_node=<int> gpus_per_node=<int>. \n Also note that a double dash -- is need if this is the last argument",
        metavar="KEY=VALUE",
    )

    group = parser.add_argument_group("Logging", "Logging parameters")
    group.add_argument(
        "--out",
        default=None,
        dest="out_log_file",
        help="Capture standard output to a log file. If not given, only prints "
        "out logs to the console",
    )
    group.add_argument(
        "--err",
        default=None,
        dest="err_log_file",
        help="Capture standard error to a log file. If not given, only prints "
        "out logs to the console",
    )
    group.add_argument(
        "--color-stderr",
        action="store_true",
        default=False,
        help="If True, uses terminal colors to color the standard error "
        "outputs in red. This does not affect the output files",
    )


def validate_arguments(args: argparse.Namespace):
    """
    Validation checks for the common arguments. Raises exceptions on failure.

    :param args: The parsed arguments.
    """
    # There are four modes of operation:
    # 1. The user specifies the number of nodes and processes per node
    # 2. The user specifies the number of nodes (processes per node use the
    #    current system's default)
    # 3. The user specifies a minimum number of GPUs
    # 4. The user specifies a minimum amount of GPU memory

    args_dict = vars(args)
    # Only CLIs that define a ``command`` positional should run this check. Use
    # key presence rather than ``.get('command') is not None`` so that the
    # forgot-the-command case (key present, value ``None``) is caught here with a
    # clean validation error instead of crashing later when ``None`` reaches the
    # command join.
    if 'command' in args_dict:
        if not args.command and not args.batch_script:
            raise ValueError(
                "Either a command or a batch script has to be provided"
            )

        if args.batch_script and args.command:
            raise ValueError(f"A pre-generated batch script file name was provided and an explicit command {args.command} - invalid combination.")

    # TODO(later): Convert some mutual exclusive behavior to constraints on
    #              number of nodes/ranks
    if not args.nodes and not args.gpus_at_least and not args.gpumem_at_least:
        raise ValueError(
            "One of the following flags has to be set: --nodes, --gpus-at-least, or --gpumem-at-least"
        )
    if args.nodes and args.gpus_at_least:
        raise ValueError(
            "The --nodes and --gpus-at-least flags are mutually " "exclusive"
        )
    if args.nodes and args.gpumem_at_least:
        raise ValueError(
            "The --nodes and --gpumem-at-least flags are mutually " "exclusive"
        )
    if args.gpus_at_least and args.procs_per_node:
        raise ValueError(
            "The --gpus-at-least and --procs-per-node flags " "are mutually exclusive"
        )
    if args.gpumem_at_least and args.procs_per_node:
        raise ValueError(
            "The --gpumem-at-least and --procs-per-node flags " "are mutually exclusive"
        )
    if args.gpumem_at_least and args.gpus_at_least:
        raise ValueError(
            "The --gpumem-at-least and --gpus-at-least flags " "are mutually exclusive"
        )
    if args.local and args.bg:
        raise ValueError('"--local" jobs cannot be run in the background')
    if args.local and args.scheduler:
        raise ValueError("The --local and --scheduler flags are mutually " "exclusive")

    if args.output_script:
        output_script = args.output_script
        if os.path.dirname(output_script):
            raise ValueError(f"User provided output script filename cannot be a absolute or relative path: {output_script}")

    if args.launch_dir == None and not args.bg: # ephemeral interactive job
        if args.output_script:
            raise ValueError("A output script file name was provided for a ephemeral interative job.")

        if args.out_log_file:
            raise ValueError("A output log file name was provided for a ephemeral interative job.")

        if args.err_log_file:
            raise ValueError("A error log file name was provided for a ephemeral interative job.")

        if args.save_hostlist:
            raise ValueError("Saving the hostlist was requested for a ephemeral interative job.")

    # Same rejection as -o/--output-script above, and for the same reason:
    # both are joined onto the launch folder (scheduler.py's
    # create_launch_folder/create_launch_folder_name) whose intermediate
    # directories are never created, so a path-bearing --out/--err would
    # otherwise reach an uncaught FileNotFoundError at `open(..., "wb")`
    # well after the launch directory (and launch.sh) already exist on
    # disk -- a raw traceback plus a half-built launch directory instead of
    # a clean, upfront validation error. Rejecting here (rather than
    # os.makedirs(..., exist_ok=True) at the point of use) keeps --out/--err
    # consistent with -o's existing, documented "no path component" rule
    # instead of quietly adding nested-log-directory support that was never
    # asked for -- and scheduler.py's log-file handling is out of scope for
    # this fix. Checked after the ephemeral-job block above so that an
    # ephemeral job with a path-bearing --out/--err reports the more
    # fundamental "not allowed in this mode at all" error first.
    for flag, log_file in (("--out", args.out_log_file), ("--err", args.err_log_file)):
        if log_file and os.path.dirname(log_file):
            raise ValueError(
                f"User provided {flag} filename cannot be a absolute or relative path: {log_file}"
            )

    if args.output_script and args.batch_script:
        raise ValueError("Cannot specify both an output script name: {args.output_script} and a pre-generated batch script {args.batch_script}.")

    if args.batch_script and not os.path.exists(args.batch_script):
        raise ValueError(f"A pre-generated batch script file name was provided but the file does not exist.")

# See if the system can be autodetected and then process some special arguments
# that can autoselect the number of ranks / GPUs
def process_arguments(args: argparse.Namespace, logger: logging.Logger) -> System:
    validate_arguments(args)

    # Set system and launch configuration based on arguments
    system, args.nodes, args.procs_per_node, args.gpus_per_proc = (
        configure.configure_launch(
            args.queue,
            args.nodes,
            args.procs_per_node,
            args.gpus_per_proc,
            args.gpus_at_least,
            args.gpumem_at_least,
            args.system_params,
            resolve_comm_backend(args.job_comm_protocol),
        )
    )

    return system
