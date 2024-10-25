# Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
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
import os
import tempfile
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
    partition: Optional[str] = None
    # The account to use for the scheduler
    account: Optional[str] = None
    # Additional launcher flags
    launcher_flags: Optional[str] = None
    # Hijack preload commands into a scheduler
    ld_preloads: Optional[list[str]] = None

    def launch_command(self, blocking: bool = True) -> list[str]:
        """
        Returns the launch command for this scheduler. Returns the
        command prefix before the program to run.

        :param blocking: Whether to launch a command that waits for the
                         command to finish (True), or launch a batch
                         script that immediately returns (False).
        :return: The command prefix as a list of strings (one per argument).
        """
        raise NotImplementedError

    def launcher_script(self,
                        system: 'System',
                        command: str,
                        args: Optional[list[str]] = None) -> str:
        """
        Returns the full launcher script, which can be saved as a batch
        script, for the given system and launcher configuration.
        This script usually performs node/resource allocation and manages I/O.

        :param system: The system to use.
        :return: A shell script as a string.
        """
        raise NotImplementedError

    def internal_script(self, system: 'System') -> Optional[str]:
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

    def launch(self,
               system: 'System',
               command: str,
               args: Optional[list[str]] = None,
               blocking: bool = True,
               script_file: Optional[str] = None,
               setup_only: bool = False,
               color_stderr: bool = False) -> str:
        """
        Launches the given command and arguments uaing this launcher.

        :param system: The system to use for launching the job.
        :param command: The command line to run.
        :param args: The arguments to use for the command.
        :param blocking: If True, blocks until the job is complete
                         and redirects/duplicates outputs to the terminal.
        :param script_file: If given, saves the output script to this file.
        :param verbose: If True, prints more information about the job details.
        :param setup_only: If True, only sets up the job and does not launch it.
        :param color_stderr: If True, colors stderr terminal outputs in red.
        :return: The queued job ID as a string.
        """
        # Create a temporary file or a script file, if given
        if script_file is not None:
            if os.path.dirname(script_file):
                os.makedirs(os.path.dirname(script_file), exist_ok=True)

            # TODO: Should we warn if this file exists or fail without "-f"?
            file = open(script_file, 'w')
            filename = os.path.abspath(script_file)
        else:
            file = tempfile.NamedTemporaryFile('w',
                                               prefix='launch-',
                                               suffix='.sh',
                                               delete=False)
            filename = file.name

        logger.info(f'Script filename: {filename}')
        with file as fp:
            fp.write(self.launcher_script(system, command, args))
        os.chmod(filename, 0o700)

        if setup_only:
            return ''

        cmd = self.launch_command(blocking)
        full_cmdline = cmd + [filename]
        logger.info(f'Launching {" ".join(full_cmdline)}')

        try:
            if blocking:  # Launch job and trace outputs live
                run_process_with_live_output(full_cmdline,
                                             color_stderr=color_stderr)
                # In this mode, there is no job ID
                return None
            else:
                # Run batch script and get job ID
                process = subprocess.run(cmd, capture_output=True)
                return self.get_job_id(process.stdout)
        finally:
            if script_file is None:  # Erase temporary file
                os.unlink(filename)
