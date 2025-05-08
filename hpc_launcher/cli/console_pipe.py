"""
Implements utilities to pipe outputs from stdout and stderr to the console
and potentially to output files, similarly to the ``tee`` application.
Works best with unbuffered Python (``python -u``).

The implementation is loosely based on https://stackoverflow.com/a/25960956
but modernized to use ``async def``.
"""

import asyncio
import io
import sys
import subprocess
from typing import Optional


async def replicate_output(
    input_stream, out1, out2=None, prefix=b"", suffix=b"", buffer_size=32
):
    """
    Reads a stream, ``buffer_size`` characters at a time, and replicates
    outputs to ``out1`` and ``out2``.
    """
    while True:
        line = await input_stream.read(buffer_size)
        if not line:  # EOF
            break
        out1.write(prefix + line + suffix)
        out1.flush()
        if out2 is not None:
            out2.write(line)
            out2.flush()


async def _run_process(
    command: list[str],
    out_file: Optional[io.FileIO] = None,
    err_file: Optional[io.FileIO] = None,
    color_stderr: bool = False,
    buffer_size: int = 32,
) -> int:
    """
    Runs a process asynchronously and pipes its stdout and stderr to up to two
    streams.

    :param command: The command to run and its arguments.
    :param out_file: An optional handle to a file to pipe ``stdout`` to. Note
                     that the file must be opened in binary mode.
    :param err_file: An optional handle to a file to pipe ``stderr`` to. Note
                     that the file must be opened in binary mode.
    :param color_stderr: If True, colors the standard error output in red.
    :param buffer_size: Output buffer size in characters.
    :return: The command's exit code.
    """
    # Create the subprocess
    args = [] if len(command) == 1 else command[1:]
    process = await asyncio.create_subprocess_exec(
        command[0], *args, bufsize=0, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    # Read the stdout and stderr concurrently
    try:
        await asyncio.gather(
            replicate_output(
                process.stdout, sys.stdout.buffer, out_file, buffer_size=buffer_size
            ),
            replicate_output(
                process.stderr,
                sys.stderr.buffer,
                err_file,
                prefix=(b"\033[31m" if color_stderr else b""),
                suffix=(b"\033[0m" if color_stderr else b""),
                buffer_size=buffer_size,
            ),
        )
    except Exception:
        process.kill()
        raise
    finally:
        rc = await process.wait()
    return rc


def run_process_without_files(command: list[str]) -> int:
    """
    Runs a process "clasically" (i.e., without redirecting output and error
    streams).

    :param command: The command to run and its arguments.
    :return: The command's exit code.
    """
    result = subprocess.run(" ".join(command), shell=True)
    return result.returncode


def run_process_with_live_output(
    command: list[str],
    out_file: Optional[io.FileIO] = None,
    err_file: Optional[io.FileIO] = None,
    color_stderr: bool = False,
    buffer_size: int = 32,
) -> int:
    """
    Runs a process asynchronously and pipes its stdout and stderr to up to two
    streams.

    :param command: The command to run and its arguments.
    :param out_file: An optional handle to a file to pipe ``stdout`` to. Note
                     that the file must be opened in binary mode.
    :param err_file: An optional handle to a file to pipe ``stderr`` to. Note
                     that the file must be opened in binary mode.
    :param color_stderr: If True, colors the standard error output in red.
    :param buffer_size: Output buffer size in characters.
    :return: The command's exit code.
    """
    if not command:
        return 0
    if out_file is not None or err_file is not None or color_stderr:
        return asyncio.run(
            _run_process(command, out_file, err_file, color_stderr, buffer_size)
        )
    return run_process_without_files(command)


if __name__ == "__main__":
    code = run_process_with_live_output(sys.argv[1:], color_stderr=True)
    print("Process finished with exit code", code)
