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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
from io import StringIO
import os
import sys
import time
import subprocess
from hpc_launcher.cli.console_pipe import run_process_with_live_output

import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # If type-checking, import the other class
    from hpc_launcher.systems.system import System


@dataclass
class Scheduler:
    """
    An instance of a batch job scheduler that can launch jobs on a given
    system. Produces command line arguments and scripts to support job
    launching, and provides functionality to interactively or asynchronously
    launch a job.
    """

    # Number of nodes to use
    nodes: int
    # Processes per node
    procs_per_node: int
    # GPUs per Process (or task) if any
    gpus_per_proc: int
    # Job name
    job_name: Optional[str] = None
    # Working directory (by default, uses current working directory)
    work_dir: Optional[str] = None
    # File for logging output stream (stdout)
    out_log_file: Optional[str] = None
    # File for logging error stream (stderr)
    err_log_file: Optional[str] = None
    # Time limit (in minutes), default is no limit
    time_limit: Optional[int] = None
    # The partition or queue to use with the scheduler
    queue: Optional[str] = None
    # The account to use for the scheduler
    account: Optional[str] = None
    # The reservation to use for the scheduler
    reservation: Optional[str] = None
    # Additional launcher flags
    launcher_flags: Optional[list[str]] = None
    # Hijack preload commands into a scheduler
    ld_preloads: Optional[list[str]] = None
    # Capture the original command so that it can be added to the launch script
    command_line: Optional[list[str]] = None

    def select_interactive_or_batch(
        self,
        tmp: list[str],
        header: StringIO,
        cmd_args: list[str],
        blocking: bool = True,
    ) -> type(None):
        """
        Given a specific string "tmp" write it either in a command line argument
        or a batch shell argument.

        :param tmp: String to package up
        :param header: StringIO that will be prepended to the final script
        :param cmd_args: Mutable list of strings that will be added to the command line
        :param blocking: Flag to indicate if the temporary string is being wrapped for
                         a batch or interactive command.
        :return: None
        """
        return None

    def build_command_string_and_batch_script(
        self, system: "System"
    ) -> (str, list[str]):
        """
        Returns the strings used for a launch command as well as a batch script
        full launcher script, which can be saved as a batch
        script, for the given system and launcher configuration.
        This script usually performs node/resource allocation and manages I/O.

        :param system: The system to use.
        :return: A tuple of (shell script as a string, list of command-line arguments).
        """
        return ("", [])

    def launch_command(self, system: "System", blocking: bool = True) -> list[str]:
        """
        Returns the launch command for this scheduler. Returns the
        command prefix before the program to run.

        :param blocking: Whether to launch a command that waits for the
                         command to finish (True), or launch a batch
                         script that immediately returns (False).
        :return: The command prefix as a list of strings (one per argument).
        """
        raise NotImplementedError

    def launcher_script(
        self,
        system: "System",
        command: str,
        args: Optional[list[str]] = None,
        blocking: bool = True,
        save_hostlist: bool = False,
        launch_dir: str = "",
    ) -> str:
        """
        Returns the full launcher script, which can be saved as a batch
        script, for the given system and launcher configuration.
        This script usually performs node/resource allocation and manages I/O.

        :param system: The system to use.
        :param command: The command to launch
        :param args: Optional list of argument for the command to launch
        :param blocking: Launch the comamnd interactively if true, else in a batch job
        :params save_hostlist: Add local scripting to capture the list of hosts the command is launched on
        :return: A shell script as a string.
        """
        raise NotImplementedError

    def internal_script(self, system: "System") -> Optional[str]:
        """
        Returns the script that runs on each process within the allocated job.
        This script is optional, and usually sets up additional elements (e.g.,
        environment variables, working directory, profiler) in case external
        variables are cleaned up by the job scheduler.

        :param system: The system to use.
        :return: A shell script as a string, or None if no internal script is
                 required.
        """
        # By default, no internal script is required
        return None

    def get_job_id(self, output: str) -> Optional[str]:
        """
        Parses and returns the job ID from a batch job submission (running in
        the background). Returns ``None`` if parsing cannot be performed.

        :param output: Console outputs of the batch submission.
        :return: A string containing the job ID, or None if the output cannot
                 be parsed.
        """
        return None

    @classmethod
    def num_nodes_in_allocation(cls) -> tuple[int]:
        """
        When running under an allocation, check how many nodes are available

        :return: Number of nodes in an allocation
        """
        raise NotImplementedError

    @classmethod
    def get_parallel_configuration(cls) -> tuple[int, int, int, int]:
        """
        Using scheduler environment variables report the parallel configuration
        of the run.

        :return: A tuple of integers in the format
                 (world_size, rank, local_world_size, local_rank)
        """
        raise NotImplementedError

    def dynamically_configure_rendezvous_protocol(self, protocol: str) -> list[str]:
        """
        Configure the rendezvous protocol at runtime for a tool like PyTorch to establish
        distributed communication.

        :param protocol: Field to select which protocol to use for the rendezvous
        :return: An init_method string that conforms to
                 https://pytorch.org/docs/stable/distributed.html.
        """
        raise NotImplementedError

    def setup_rendezvous_protocol(self, protocol: str) -> list[str]:
        """
        Setup a protocol for a tool like PyTorch to use to establish
        distributed communication.

        :param protocol: Field to select which protocol to use for the rendezvous
        :return: A list of strings that are added to the torchrun-hpc launch environment.
        """
        env_list = []
        env_list.append(("TORCHRUN_HPC_SCHEDULER", type(self).__name__))
        env_list.extend(self.dynamically_configure_rendezvous_protocol(protocol))
        if protocol.lower() == "tcp":
            env_list.append(
                (
                    "TORCHRUN_HPC_RDV_PROTOCOL",
                    '"tcp://${TORCHRUN_HPC_MASTER_ADDR}:${TORCHRUN_HPC_MASTER_PORT}"',
                )
            )
        elif protocol.lower() == "mpi":
            env_list.append(("TORCHRUN_HPC_RDV_PROTOCOL", "mpi://"))
        else:
            msg = f"Unsupported rendezvous protocol {protocol}"
            raise Exception(msg)
        return env_list

    def create_launch_folder_name(
        self,
        command: str,
        folder_prefix: str = "launch",
        no_launch_dir: bool = False,
    ) -> (str, str):
        """
        Create a folder name for the launcher based on the command.

        :param command: The command line to run.
        :param folder_prefix: Specializable prefix for the folder name
        :return: A tuple of strings with the the command as a possible folder name, and the folder name.
        """
        # Remove spaces and semi-colons from the command sequence
        command_as_folder_name = (
            os.path.basename(command).replace(" ", "_").replace(";", "-")
        )
        # Create a folder for the output and error logs
        # Timestamp is of the format YYYY-MM-DD_HHhMMmSSs
        if no_launch_dir:
            folder_name = os.getcwd()
        else:
            folder_name = f'{folder_prefix}-{self.job_name or command_as_folder_name}_{time.strftime("%Y-%m-%d_%Hh%Mm%Ss")}'
        return (command_as_folder_name, folder_name)

    def create_launch_folder(
        self,
        folder_name: str,
        blocking: bool = True,
        script_file: Optional[str] = None,
        run_from_launch_dir: bool = False,
    ) -> (str, str):
        """
        Create a folder and associated launch script if approrpiate.

        :param folder_name: The name of the folder for containing all of the launch artifacts.
        :param blocking: If True, the job should run from the launch folder.
        :param script_file: If given, saves the output script to this file.
        :param run_from_launch_dir: If True, runs the command from the launch folder.
        :return: The filename for the launch script as a string.
        """

        should_make_folder = blocking or run_from_launch_dir

        # Create a temporary file or a script file, if given
        if script_file is not None:
            if os.path.dirname(script_file):
                os.makedirs(os.path.dirname(script_file), exist_ok=True)

            # Warn if this file exists
            if os.path.exists(script_file):
                logger.warning(f"Overwriting existing file {script_file}")

            filename = os.path.abspath(script_file)
        else:
            should_make_folder = True
            filename = os.path.abspath(os.path.join(folder_name, "launch.sh"))

        if self.out_log_file is None:
            self.out_log_file = os.path.abspath(os.path.join(folder_name, "out.log"))
            should_make_folder = True
        if self.err_log_file is None:
            self.err_log_file = os.path.abspath(os.path.join(folder_name, "err.log"))
            should_make_folder = True

        stub_file = ""
        if should_make_folder:
            os.makedirs(folder_name, exist_ok=True)

        return filename

    def launch(
        self,
        system: "System",
        folder_name: str,
        filename: str,
        command: str,
        args: Optional[list[str]] = None,
        blocking: bool = True,
        setup_only: bool = False,
        color_stderr: bool = False,
        run_from_launch_dir: bool = False,
        save_hostlist: bool = False,
    ) -> str:
        """
        Launches the given command and arguments uaing this launcher.

        :param system: The system to use for launching the job.
        :param folder_name: The name of the folder for containing all of the launch artifacts.
        :param filename: The filename for the launch script
        :param command: The command line to run.
        :param args: The arguments to use for the command.
        :param blocking: If True, blocks until the job is complete
                         and redirects/duplicates outputs to the terminal.
        :param setup_only: If True, only sets up the job and does not launch it.
        :param color_stderr: If True, colors stderr terminal outputs in red.
        :param run_from_launch_dir: If True, runs the command from the launch directory.
        :params save_hostlist: Add local scripting to capture the list of hosts the command is launched on
        :return: The queued job ID as a string.
        """

        # If the command is run from a directory
        if run_from_launch_dir:
            # Change the working directory to the launch folder
            if not self.work_dir:
                self.work_dir = os.path.abspath(folder_name)
            # There is no need to use the following at the moment:
            # elif shutil.which(command):
            #     command = os.path.abspath(shutil.which(command))

        # If the command exists as a file, use its absolute path
        if os.path.isfile(command):
            command = os.path.abspath(command)

        cmd = self.launch_command(system, blocking)
        full_cmdline = cmd + [filename]

        logger.info(f"Script filename: {filename}")
        with open(filename, "w") as fp:
            fp.write(
                self.launcher_script(system, command, args, blocking, save_hostlist, os.path.dirname(filename))
            )

            fp.write(f"\n# Launch command: " + " ".join(full_cmdline) + "\n")
            if self.command_line:
                fp.write(
                    f"# User command invoked: " + " ".join(self.command_line) + "\n"
                )
        os.chmod(filename, 0o700)

        if setup_only:
            logger.warning(f'To launch, run: {" ".join(full_cmdline)}')
            return ""

        logger.info(f'Launching {" ".join(full_cmdline)}')

        if blocking:  # Launch job and trace outputs live
            with open(os.path.join(folder_name, "out.log"), "wb") as out_file:
                with open(os.path.join(folder_name, "err.log"), "wb") as err_file:

                    run_process_with_live_output(
                        full_cmdline,
                        out_file=out_file,
                        err_file=err_file,
                        color_stderr=color_stderr,
                    )
            # In this mode, there is no job ID
            return None
        else:
            # Run batch script and get job ID
            process = subprocess.run(full_cmdline, capture_output=True)
            if process.returncode or process.stderr:
                logging.error(
                    f"Batch scheduler exited with error code {process.returncode}"
                )
                sys.stderr.buffer.write(process.stderr)
                return None
            return self.get_job_id(process.stdout.decode())
