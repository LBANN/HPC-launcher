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
import argparse
from hpc_launcher.cli import common_args, launch_helpers
from hpc_launcher.schedulers import get_schedulers
from hpc_launcher.schedulers.scheduler import Scheduler
from hpc_launcher.schedulers.local import LocalScheduler

try:
    import mpi4py

    mpi = True
except (ImportError, ModuleNotFoundError):
    mpi = None

import logging
import os
import shutil
import sys

logger = logging.getLogger(__name__)

# Torchrun flags that torchrun-hpc derives from the HPC scheduler and therefore
# owns. If the user passes any of these explicitly, we error out rather than
# silently overriding them. Both the dashed and underscored spellings are
# guarded. ``-m``/``--module`` is intentionally NOT included: it is allowed to
# pass through to torchrun and is detected separately for the trampoline.
TORCHRUN_HPC_MANAGED_FLAGS = frozenset({
    # Topology / rendezvous (injected via PET_* env vars)
    "--nnodes",
    "--nproc-per-node", "--nproc_per_node",
    "--rdzv-backend", "--rdzv_backend",
    "--rdzv-endpoint", "--rdzv_endpoint",
    "--rdzv-id", "--rdzv_id",
    "--node-rank", "--node_rank",
    "--master-addr", "--master_addr",
    "--master-port", "--master_port",
    "--standalone",
    # Run mode (incompatible with the wrapper invocation)
    "--no-python", "--no_python",
    "--run-path", "--run_path",
})

# Option strings that indicate Python module execution (python -m ...).
_MODULE_FLAGS = frozenset({"-m", "--module"})


class _HelpWithTorchrun(argparse.Action):
    """Custom -h/--help: print torchrun-hpc's help, then torchrun's flags.

    torchrun's flags are forwarded verbatim rather than registered on our
    parser (they would collide on -r/-t/-m), so they don't appear in the normal
    help output. We append torchrun's own help text as a labeled section so
    users can discover every flag they can pass through.
    """

    def __init__(self, option_strings, dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings=option_strings, dest=dest,
                         default=default, nargs=0, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.print_help()
        print("\n" + "=" * 79)
        print("Forwarded torchrun flags (passed through to `torch.distributed."
              "run`)")
        print("=" * 79)
        print("\nAll of the flags below are accepted and forwarded to torchrun. "
              "The flags\nmarked (managed) are derived from the HPC scheduler by "
              "torchrun-hpc; passing\nthem explicitly is an error. Use "
              "torchrun-hpc's own -N/-n/etc. instead.")
        try:
            from torch.distributed.run import get_args_parser
            # Reuse torchrun's own parser to render its flags; strip its usage
            # line since the flags are what matter here.
            torchrun_help = get_args_parser().format_help()
            _, _, body = torchrun_help.partition("options:\n")
            body = body or torchrun_help
            # Drop torchrun's own -h/--help entry; torchrun-hpc owns --help.
            lines = [
                ln for ln in body.splitlines()
                if not ln.lstrip().startswith("-h, --help")
                and "show this help message and exit" not in ln
            ]
            # Annotate the flags torchrun-hpc manages.
            for i, ln in enumerate(lines):
                stripped = ln.lstrip()
                if stripped.split(",")[0].split()[0:1] and any(
                        stripped.startswith(f) for f in TORCHRUN_HPC_MANAGED_FLAGS):
                    lines[i] = ln.rstrip() + "   (managed)"
            print("\noptions:")
            print("\n".join(lines))
        except (ImportError, ModuleNotFoundError):
            print("\n(PyTorch is not installed; run `torchrun --help` on a "
                  "system with PyTorch to see the forwarded flags.)")
        parser.exit()


def _detect_collision(user_torchrun_flags: list[str]) -> None:
    """Error out if the user passed a torchrun flag that torchrun-hpc manages."""
    hits = sorted({
        tok.split("=")[0]
        for tok in user_torchrun_flags
        if tok.split("=")[0] in TORCHRUN_HPC_MANAGED_FLAGS
    })
    if hits:
        sys.exit(
            "ERROR: torchrun-hpc manages the following flags automatically; do "
            "not pass them explicitly:\n"
            f"  {', '.join(hits)}\n"
            "Use torchrun-hpc's own options (-N/--nodes, -n/--procs-per-node, "
            "etc.) to control job topology and rendezvous.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=
        "A wrapper script that launches and runs distributed PyTorch on HPC "
        "systems by wrapping the real torchrun. All torchrun flags are accepted "
        "and forwarded; run with --help to see them.",
        add_help=False)
    # Replace the default -h/--help with one that also prints torchrun's flags,
    # so users can discover everything that is forwarded to torchrun.
    parser.add_argument(
        "-h", "--help", action=_HelpWithTorchrun,
        help="Show this help message (including forwarded torchrun flags) and "
        "exit.")
    # Do not register a `-t` short form for --time-limit: torchrun uses `-t` for
    # --tee, and torchrun-hpc forwards it through.
    common_args.setup_arguments(parser, time_limit_short=False)
    parser.add_argument(
        "--rdv-protocol",
        dest="rdv_protocol",
        default=None,
        help="Specifies rendezvous protocol to use: mpi | tcp. The default "
        "(tcp) wraps the real torchrun using its c10d rendezvous. 'mpi' uses "
        "the legacy per-rank launcher.",
    )

    parser.add_argument(
        "--fraction-max-gpu-mem",
        type=float,
        default=None,
        help="Use the torch.cuda.set_per_process_memory_fraction "
        "to limit how much GPU memory can be allocated.",
    )

    parser.add_argument(
        "-u",
        "--unswap-rocr-hip-vis-dev",
        action="store_true",
        default=False,
        help=
        "Undo moving ROCR_VISIBLE_DEVICES into the HIP_VISIBLE_DEVICES env variable. "
        "In PyTorch codes HIP_VISIBLE_DEVICES is most similar to CUDA_VISIBLE_DEVICES. "
        "Ensureing that HIP vs ROCR can improve behavior of HF Accelerate and TorchTitan.",
    )

    parser.add_argument(
        "--gpu-visibility",
        dest="gpu_visibility",
        choices=["single", "all"],
        default="single",
        help="Controls how each torchrun worker sees the node's GPUs. 'single' "
        "(default) narrows each worker to its one assigned GPU; 'all' leaves "
        "every GPU visible and sets LOCAL_RANK to the round-robin index.",
    )
    return parser


def main():
    parser = _build_parser()

    # Two-stage parse: peel off the torchrun-hpc flags and leave everything else
    # (torchrun flags + training script + script args, in order) for torchrun.
    args, leftover = parser.parse_known_args()

    launch_helpers.setup_logging(logger, args.verbose)

    if args.rdv_protocol == "mpi":
        return _launch_mpi_legacy(args, leftover)

    if args.rdv_protocol not in (None, "tcp"):
        raise Exception(f"Unknown rendezvous {args.rdv_protocol} requested.")

    return _launch_torchrun(args, leftover)


def _launch_torchrun(args, leftover):
    """Default path: wrap the real torchrun, one process per node."""
    try:
        import torch  # noqa: F401
    except (ModuleNotFoundError, ImportError):
        print(
            "PyTorch is not installed on this system, but is required for torchrun-hpc."
        )
        exit(1)

    # Reuse torchrun's own parser to validate the leftover tokens and to extract
    # the training script and its arguments. This is how we accept ALL torchrun
    # flags without tracking them ourselves.
    from torch.distributed.run import get_args_parser as torchrun_args_parser

    torchrun_ns = torchrun_args_parser().parse_args(leftover)

    # The user's torchrun flags are everything in `leftover` before the training
    # script token and its trailing args.
    num_trailing = 1 + len(torchrun_ns.training_script_args)
    user_torchrun_flags = leftover[:len(leftover) - num_trailing]

    _detect_collision(user_torchrun_flags)

    is_module = any(tok in _MODULE_FLAGS for tok in user_torchrun_flags)

    if args.fraction_max_gpu_mem and args.fraction_max_gpu_mem != 1.0:
        if not args.system_params:
            args.system_params = {}
        args.system_params["fraction_max_gpu_mem"] = args.fraction_max_gpu_mem

    # Process special arguments that can autoselect the number of ranks / GPUs
    system = common_args.process_arguments(args, logger)
    optimize_comm_protocol = ""
    if args.job_comm_protocol:
        optimize_comm_protocol = args.job_comm_protocol
    if optimize_comm_protocol.upper() == "MPI":
        logger.warning(
            f"Using MPI as the primary communication protocol for PyTorch requires additional support"
        )
    else:
        system.job_comm_protocol = "*CCL"

    # Pick batch scheduler and put it into per-node (torchrun) launch mode.
    scheduler = launch_helpers.select_scheduler(args, logger, system)
    scheduler.torchrun_mode = True

    # Reuse the scheduler's existing rendezvous setup to obtain the HPC-matched
    # coordinator address/port, then translate it into torchrun's c10d PET_ env
    # variables. This preserves the HPC-specific rendezvous override.
    env_list = scheduler.setup_rendezvous_protocol("tcp")
    env_list.append(("PET_NNODES", f"{args.nodes}"))
    env_list.append(("PET_NPROC_PER_NODE", f"{args.procs_per_node}"))
    env_list.append(("PET_RDZV_BACKEND", "c10d"))
    env_list.append((
        "PET_RDZV_ENDPOINT",
        "${TORCHRUN_HPC_MASTER_ADDR}:${TORCHRUN_HPC_MASTER_PORT}",
    ))
    env_list.append(("TORCHRUN_HPC_MODE", "torchrun"))
    env_list.append(("TORCHRUN_HPC_GPU_VISIBILITY", args.gpu_visibility))
    if is_module:
        env_list.append(("TORCHRUN_HPC_IS_MODULE", "1"))
    if args.unswap_rocr_hip_vis_dev:
        env_list.append(("TORCHRUN_HPC_UNSWAP_ROCR_HIP_VIS_DEV", "TRUE"))

    system.extend_environment_variables(env_list)

    # Inject the one-task-per-node topology through the existing scheduler
    # override mechanism, so we do not have to modify the Slurm/LSF schedulers.
    override_args = dict(args.override_args) if args.override_args else {}
    _inject_topology_overrides(scheduler, args, override_args)

    if args.bg and args.launch_dir is None:  # or args.batch_script
        # If running a batch job with no launch directory argument,
        # run in the generated timestamped directory
        args.launch_dir = ""
    if args.launch_dir is None and not args.bg:
        args.launch_dir = ""
        logger.info(
            f"torchrun-hpc needs to run jobs from a launch directory -- automatically setting the -l (--launch-dir) CLI argument"
        )

    _, folder_name = scheduler.create_launch_folder_name(
        torchrun_ns.training_script, "torchrun_hpc", args.launch_dir)

    script_file = scheduler.create_launch_folder(folder_name, not args.bg,
                                                 args.output_script,
                                                 args.dry_run)

    trampoline_file = "torchrun_hpc_trampoline.py"

    if os.path.exists(folder_name):
        copied_trampoline_file = folder_name + "/" + trampoline_file
        package_path = os.path.dirname(os.path.abspath(__file__))
        shutil.copy(
            os.path.join(package_path, "..", "torch", trampoline_file),
            copied_trampoline_file,
        )

    # Launch the real torchrun (as a module, to avoid depending on PATH). Its
    # training_script is our trampoline, which in turn runs the user's target.
    # Strip -m/--module from the forwarded flags: torchrun must run the
    # trampoline as a normal script, and the trampoline itself runs the user's
    # target as a module (via TORCHRUN_HPC_IS_MODULE). Forwarding -m would make
    # torchrun try to import the trampoline path as a module.
    forwarded_flags = [f for f in user_torchrun_flags if f not in _MODULE_FLAGS]
    command = sys.executable
    launch_args = ["-m", "torch.distributed.run"]
    launch_args += forwarded_flags
    launch_args.append(f"{os.path.abspath(folder_name)}/{trampoline_file}")
    if is_module:
        launch_args.append(torchrun_ns.training_script)
    else:
        launch_args.append(os.path.abspath(torchrun_ns.training_script))
    launch_args += torchrun_ns.training_script_args

    logger.info(f"Running job in directory: {folder_name}")

    jobid = scheduler.launch(
        system,
        folder_name,
        script_file,
        command,
        launch_args,
        override_args,
        not args.bg,
        args.setup_only,
        args.color_stderr,
        args.dry_run,
        args.launch_dir != None and args.save_hostlist,
    )

    if jobid:
        msg = f"Job ID: {jobid} launched from {folder_name}"
        logger.info(msg)
        if not args.verbose:
            print(msg)


def _inject_topology_overrides(scheduler, args, override_args: dict) -> None:
    """Force one task per node with whole-node GPU access via scheduler overrides.

    Flux bakes its task count into an argument key, so it handles torchrun_mode
    in its own build step; here we only adjust the schedulers whose relevant
    knobs are clean override keys (Slurm, LSF).
    """
    scheduler_name = type(scheduler).__name__
    gpus_per_node = args.procs_per_node * (args.gpus_per_proc or 0)
    if scheduler_name == "SlurmScheduler":
        override_args["--ntasks"] = f"{args.nodes}"
        override_args["--ntasks-per-node"] = "1"
        if gpus_per_node > 0:
            override_args["--gpus-per-task"] = f"{gpus_per_node}"
    elif scheduler_name == "LSFScheduler":
        override_args["--tasks_per_rs"] = "1"


def _launch_mpi_legacy(args, leftover):
    """Legacy path: one process per rank, custom MPI/TCP rendezvous.

    This reproduces the historical torchrun-hpc behavior for
    ``--rdv-protocol mpi``, where the scheduler launches one task per rank and
    the trampoline performs the rendezvous itself.
    """
    if not leftover:
        parser = _build_parser()
        parser.error("a command to execute is required")

    # In legacy mode the first non-flag token is the command; a leading -m/-module
    # indicates Python module execution.
    is_module = leftover[0] in _MODULE_FLAGS
    rest = leftover[1:] if is_module else leftover
    command = rest[0]
    command_args = rest[1:]

    if args.fraction_max_gpu_mem and args.fraction_max_gpu_mem != 1.0:
        if not args.system_params:
            args.system_params = {}
        args.system_params["fraction_max_gpu_mem"] = args.fraction_max_gpu_mem

    system = common_args.process_arguments(args, logger)
    optimize_comm_protocol = ""
    if args.job_comm_protocol:
        optimize_comm_protocol = args.job_comm_protocol
    if optimize_comm_protocol.upper() == "MPI":
        logger.warning(
            f"Using MPI as the primary communication protocol for PyTorch requires additional support"
        )
    else:
        system.job_comm_protocol = "*CCL"
    scheduler = launch_helpers.select_scheduler(args, logger, system)

    if not mpi:
        raise Exception("MPI rendezvous requested but not available")
    env_list = scheduler.setup_rendezvous_protocol("mpi")
    env_list.append(("TORCHRUN_HPC_MODE", "mpi"))

    if args.unswap_rocr_hip_vis_dev:
        env_list.append(("TORCHRUN_HPC_UNSWAP_ROCR_HIP_VIS_DEV", "TRUE"))

    system.extend_environment_variables(env_list)

    try:
        import torch  # noqa: F401
    except (ModuleNotFoundError, ImportError):
        print(
            "PyTorch is not installed on this system, but is required for torchrun-hpc."
        )
        exit(1)

    if args.bg and args.launch_dir is None:
        args.launch_dir = ""
    if args.launch_dir is None and not args.bg:
        args.launch_dir = ""
        logger.info(
            f"torchrun-hpc needs to run jobs from a launch directory -- automatically setting the -l (--launch-dir) CLI argument"
        )

    _, folder_name = scheduler.create_launch_folder_name(
        command, "torchrun_hpc", args.launch_dir)

    script_file = scheduler.create_launch_folder(folder_name, not args.bg,
                                                 args.output_script,
                                                 args.dry_run)

    trampoline_file = "torchrun_hpc_trampoline.py"

    if os.path.exists(folder_name):
        copied_trampoline_file = folder_name + "/" + trampoline_file
        package_path = os.path.dirname(os.path.abspath(__file__))
        shutil.copy(
            os.path.join(package_path, "..", "torch", trampoline_file),
            copied_trampoline_file,
        )

    launch_command = sys.executable
    launch_args = [
        "-u",
        f"{os.path.abspath(folder_name)}/{trampoline_file}",
    ]
    if is_module:
        launch_args += ["-m", command]
    else:
        launch_args.append(os.path.abspath(command))
    launch_args += command_args

    logger.info(f"Running job in directory: {folder_name}")

    jobid = scheduler.launch(
        system,
        folder_name,
        script_file,
        launch_command,
        launch_args,
        args.override_args,
        not args.bg,
        args.setup_only,
        args.color_stderr,
        args.dry_run,
        args.launch_dir != None and args.save_hostlist,
    )

    if jobid:
        msg = f"Job ID: {jobid} launched from {folder_name}"
        logger.info(msg)
        if not args.verbose:
            print(msg)


if __name__ == "__main__":
    main()
