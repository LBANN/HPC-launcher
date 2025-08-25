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
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
from io import StringIO
import os
import sys
import time
import tempfile
import subprocess
import shutil
from hpc_launcher.cli.console_pipe import run_process_with_live_output
from hpc_launcher.schedulers import parse_env_list

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
    # Request exclusive access to the resources
    exclusive: Optional[bool] = None
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
    # Dependency str
    dependency: Optional[str] = None
    # Hijack preload commands into a scheduler
    ld_preloads: Optional[list[str]] = None
    # Capture the original command so that it can be added to the launch script
    command_line: Optional[list[str]] = None

    # Command line flags given to a batch or interactive submit command
    submit_only_args = OrderedDict()
    # Commands given to active run command
    run_only_args = OrderedDict()
    # Flags given to both submit and run commands
    common_launch_args = OrderedDict()

    # CLI flags for override
    override_launch_args: Optional[dict] = None

    def build_scheduler_specific_arguments(
            self, system: "System", blocking: bool = True
    ):
        return NotImplementedError

    def build_command_string_and_batch_script(
            self, system: "System", blocking: bool = True, cli_env_only: bool = False,
            for_launch_cmd: bool = True,
    ) -> (str, list[str]):
        """
        Returns the strings used for a launch command as well as a batch script
        full launcher script, which can be saved as a batch
        script, for the given system and launcher configuration.
        This script usually performs node/resource allocation and manages I/O.

        :param system: The system to use.
        :param blocking: Is the job interactive of blocking
        :param cli_env_only: Append environment variables to CLI not a launch script
        :param for_launch_cmd:  Some args should not be in both the header and launch cmnd. Ex: flux --dependency=afterany:XXX
        :return: A tuple of (shell script as a string, list of command-line arguments).
        """
        env_vars = system.environment_variables()
        passthrough_env_vars = system.passthrough_environment_variables()

        header = StringIO()
        header.write("#!/bin/sh\n")
        cmd_args = []

        self.build_scheduler_specific_arguments(system, blocking)

        # Enable the system to apply some customization to the scheduler instance
        system.customize_scheduler(self)

        if self.override_launch_args:
            for k,v in self.override_launch_args.items():
                arg_overridden = False
                remove_arg = False
                if "~" in k:
                    k = k.replace("~", "")
                    remove_arg = True
                if k in self.common_launch_args:
                    if remove_arg:
                        self.common_launch_args.pop(k, None)
                    else:
                        tmp = self.common_launch_args[k]
                        self.common_launch_args[k] = v
                        arg_overridden = True

                if k in self.run_only_args:
                    if remove_arg:
                        self.run_only_args.pop(k, None)
                    else:
                        tmp = self.run_only_args[k]
                        self.run_only_args[k] = v
                        arg_overridden = True

                if k in self.submit_only_args:
                    if remove_arg:
                        self.submit_only_args.pop(k, None)
                    else:
                        tmp = self.submit_only_args[k]
                        self.submit_only_args[k] = v
                        arg_overridden = True

                if not arg_overridden and not remove_arg:
                    self.common_launch_args[k] = v

        if not blocking: # Only add batch script header items on non-blocking calls
            prefix = self.batch_script_prefix()
            for k,v in self.submit_only_args.items():
                if not for_launch_cmd and k == '--dependency':
                    continue
                if not v:
                    header.write(f"{prefix} {k}\n")
                else:
                    header.write(f"{prefix} {k}={v}\n")
            for k,v in self.common_launch_args.items():
                if not for_launch_cmd and k == '--dependency':
                    continue
                if not v:
                    header.write(f"{prefix} {k}\n")
                else:
                    header.write(f"{prefix} {k}={v}\n")

        if len(env_vars):
            if blocking and cli_env_only:
                self.cli_env_arg(env_vars)
            else:
                for e in env_vars:
                    header.write(parse_env_list(*e))

        if len(passthrough_env_vars):
            if blocking:
                self.cli_env_arg(passthrough_env_vars)
            else:
                for k, v in passthrough_env_vars:
                    header.write(f"export {k}={v}\n")

        return (header.getvalue(), cmd_args)

    def batch_script_prefix(self) -> str:
        """
        Returns scheduler specific prefix for batch scripts
        :return: scheduler specific prefix for batch scripts
        """
        raise NotImplementedError

    def blocking_launch_command(self) -> list[str]:
        """
        Returns scheduler specific command for interactive (blocking) jobs
        :return: scheduler specific command for interactive (blocking) jobs
        """
        raise NotImplementedError

    def nonblocking_launch_command(self) -> list[str]:
        """
        Returns scheduler specific command for non-blocking batch jobs
        :return: scheduler specific command for non-blocking batch jobs
        """
        raise NotImplementedError

    def cli_env_arg(self, env_list: list[tuple[str,str]]) -> None:
        """
        How should environment variables be passed to launched command.
        Append them the to the submit_only_args
        :return: None
        """
        raise NotImplementedError

    def launch_command(
            self, system: "System", blocking: bool = True, cli_env_only: bool = False
    ) -> list[str]:
        """
        Returns the launch command for this scheduler. Returns the
        command prefix before the program to run.

        :param blocking: Whether to launch a command that waits for the
                         command to finish (True), or launch a batch
                         script that immediately returns (False).
        :return: The command prefix as a list of strings (one per argument).
        """
        (header_lines, cmd_args) = self.build_command_string_and_batch_script(
            system, blocking, cli_env_only,
        )

        # Both commands get the submit args
        for k,v in self.common_launch_args.items():
            if not v:
                cmd_args += [k]
            else:
                cmd_args += [f"{k}={v}"]
        for k,v in self.submit_only_args.items():
            if not v:
                cmd_args += [k]
            else:
                cmd_args += [f"{k}={v}"]
        if not blocking:
            return self.nonblocking_launch_command() + cmd_args

        # For interactive jobs add the run args (if the scheduler permits it)
        if self.enable_run_args_on_launch_command():
            for k,v in self.run_only_args.items():
                if not v:
                    cmd_args += [k]
                else:
                    cmd_args += [f"{k}={v}"]
        return self.blocking_launch_command() + cmd_args

    def export_hostlist(self) -> str:
        """
        Returns a shell cmmand to set the hostlist of the job to an environment variable.
        :return: string that exports hostlist to environment variable
        """
        raise NotImplementedError

    def require_parallel_internal_run_command(self, blocking: bool) -> bool:
        """
        Returns scheduler specific command for use in a batch or interactive submitted script
        :return: bool indicating if a special run command is required
        """
        if not blocking:
            return True
        else:
            return False

    def enable_run_args_on_launch_command(self) -> bool:
        """
        Allow scheduler to explicitly enable or disable appending the runtime
        arguments to the launch command.
        :return: bool indicating if run arguments are appended to launch command
        """
        return True

    def internal_script_run_command(self) -> str:
        """
        Returns scheduler specific command for use in a batch or interactive submitted script
        :return: string scheduler specific command for use in a batch or interactive submitted script
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
        :params launch_dir: Folder used for running the command
        :return: A shell script as a string.
        """
        script = ""
        # Launch command only use the cmd_args to construct the shell script to be launched
        (header_lines, cmd_args) = self.build_command_string_and_batch_script(
            system, blocking, False, for_launch_cmd=False
        )
        # For batch jobs add any common args to the internal command
        if not blocking:
            for k,v in self.common_launch_args.items():
                if not v:
                    cmd_args += [k]
                else:
                    cmd_args += [f"{k}={v}"]
        # For jobs that require a parallel internal command add any run args
        if self.require_parallel_internal_run_command(blocking):
            for k,v in self.run_only_args.items():
                if not v:
                    cmd_args += [k]
                else:
                    cmd_args += [f"{k}={v}"]

        # Configure header and command line with scheduler job options
        script += header_lines
        script += "\n"
        if launch_dir != os.getcwd():
            callee_directory = os.path.dirname(launch_dir)
            logger.info(f"Callee directory: {callee_directory} - and {launch_dir}")
            script += f"export PYTHONPATH={callee_directory}:" + "${PYTHONPATH}\n"
        if save_hostlist:
            script += f'export RANK={self.get_parallel_rank_env_variable()}\n'
            script += self.export_hostlist()
            script += 'if [ "${RANK}" = "0" ]; then\n'
            script += "    echo ${HPC_LAUNCHER_HOSTLIST} > " + os.path.join(launch_dir, f"hpc_launcher_hostlist.txt\n")
            script += "fi\n\n"

        if system.active_system_params:
            system_params = system.active_system_params
            if system_params.fraction_max_gpu_mem and system_params.fraction_max_gpu_mem != 1.0:
                script += f'export HPC_LAUNCHER_MAX_GPU_MEM={system_params.fraction_max_gpu_mem}\n'

        if self.require_parallel_internal_run_command(blocking):
            script += self.internal_script_run_command()
            script += " ".join(cmd_args)
            script += " "

        script += f"{command}"

        for arg in args:
            script += f" {arg}"

        script += "\n"

        return script

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
    def get_parallel_rank_env_variable(cls) -> str:
        """
        When running under an allocation, return the environment variable to get the current rank

        :return: environment variable for rank in an allocation
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
        launch_dir: Optional[str] = None,
    ) -> (str, str):
        """
        Create a folder name for the launcher based on the command.

        :param command: The command line to run.
        :param folder_prefix: Specializable prefix for the folder name
        :param launch_dir: [Optional] Name of launch directory, None, or "." for <cwd>
        :return: A tuple of strings with the the command as a possible folder name, and the folder name.
        """
        # Remove spaces and semi-colons from the command sequence
        # command_as_folder_name = "batch_script"
        # if command:
        command_as_folder_name = (
            os.path.basename(command).replace(" ", "_").replace(";", "-")
        )

        if launch_dir == ".":
            folder_name = os.getcwd()
        elif launch_dir == "":
            # Create a folder for the output and error logs
            # Timestamp is of the format YYYY-MM-DD_HHhMMmSSs
            folder_name = f'{folder_prefix}-{self.job_name or command_as_folder_name}_{time.strftime("%Y-%m-%d_%Hh%Mm%Ss")}'
        else:
            folder_name = launch_dir

        return (command_as_folder_name, folder_name)

    def create_launch_folder(
        self,
        folder_name: str,
        blocking: bool = True,
        script_file: Optional[str] = None,
        dry_run: bool = False,
    ) -> (str, str):
        """
        Create a folder and associated launch script if approrpiate.

        :param folder_name: The name of the folder for containing all of the launch artifacts.
        :param blocking: If True, the job should run from the launch folder.
        :param script_file: If given, saves the output script to this file.
        :param dry_run: If True, only sets up the job and does not launch it.
        :return: The filename for the launch script as a string.
        """

        should_make_folder = folder_name != None
        copy_script_to_filename = False

        # Create a temporary file or a script file, if given
        if script_file is not None:
            if os.path.exists(script_file):
                # A batch file was provided
                filename = os.path.abspath(os.path.join(folder_name, os.path.basename(script_file)))
                # Plan to copy provided file into the launch directory
                copy_script_to_filename = True
            else:
                # Create a new batch file with the provided name
                if os.path.dirname(script_file) and not dry_run:
                    msg = f"Unsupported script_file {script_file} - cannot be a full path"
                    raise Exception(msg)
                else:
                    filename = os.path.abspath(os.path.join(folder_name, script_file))

            # Warn if this file exists
            if os.path.exists(filename):
                logger.warning(f"Overwriting existing file {filename}")

        else:
            filename = os.path.abspath(os.path.join(folder_name, "launch.sh"))

        if self.out_log_file is None:
            self.out_log_file = os.path.abspath(os.path.join(folder_name, "out.log"))
        else:
            if not os.path.isabs(self.out_log_file):
                log_file = os.path.abspath(os.path.join(folder_name, self.out_log_file))
                self.out_log_file = log_file

        if self.err_log_file is None:
            self.err_log_file = os.path.abspath(os.path.join(folder_name, "err.log"))
        else:
            if not os.path.isabs(self.err_log_file):
                log_file = os.path.abspath(os.path.join(folder_name, self.err_log_file))
                self.err_log_file = log_file

        stub_file = ""
        if should_make_folder and not dry_run:
            os.makedirs(folder_name, exist_ok=True)
            if copy_script_to_filename:
                shutil.copy(script_file, filename)

        return filename

    def launch(
        self,
        system: "System",
        folder_name: Optional[str],
        filename: Optional[str],
        command: str,
        args: Optional[list[str]] = None,
        override_launch_args: Optional[dict] = None,
        blocking: bool = True,
        setup_only: bool = False,
        color_stderr: bool = False,
        dry_run: bool = False,
        save_hostlist: bool = False,
        immutable_launch_script: bool = False,
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
        :params immutable_launch_script: It True, do not modify the script and put any system env arguments on the CLI command
        :return: The queued job ID as a string.
        """

        self.override_launch_args = override_launch_args

        # If the command is run from a directory
        if folder_name:
            # Change the working directory to the launch folder
            if not self.work_dir:
                self.work_dir = os.path.abspath(folder_name)
            # There is no need to use the following at the moment:
            # elif shutil.which(command):
            #     command = os.path.abspath(shutil.which(command))

        # If the command exists as a file, use its absolute path
        if command and os.path.isfile(command):
            command = os.path.abspath(command)

        use_launch_folder = folder_name or filename
        cmd = self.launch_command(system, blocking, not use_launch_folder or immutable_launch_script)

        if not use_launch_folder: # Launch job and trace outputs live
            # Run interactive script
            full_cmdline = cmd + [command]

            for arg in args:
                full_cmdline += [arg]

            if setup_only:
                logger.warning(f'To launch, run: {" ".join(full_cmdline)}')
                return ""

            logger.info(f'Launching {" ".join(full_cmdline)}')

            if not dry_run:
                process = subprocess.run(full_cmdline, capture_output=True)
                sys.stdout.buffer.write(process.stdout)
                sys.stderr.buffer.write(process.stderr)
                if process.returncode or process.stderr:
                    logging.error(
                        f"Interactive scheduler exited with error code {process.returncode}"
                    )
            return None
        else:
            full_cmdline = cmd + [filename]
            logger.info(f"Script filename: {filename}")
            if not dry_run and not immutable_launch_script:
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

            if dry_run:
                return None

            if blocking:  # Launch job and trace outputs live
               with open(self.out_log_file, "wb") as out_file:
                   with open(self.err_log_file, "wb") as err_file:

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
