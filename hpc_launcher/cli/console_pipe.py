"""
Implements utilities to pipe outputs from stdout and stderr to the console
and potentially to output files, similarly to the ``tee`` application.
Works best with unbuffered Python (``python -u``).

The implementation is loosely based on https://stackoverflow.com/a/25960956
but modernized to use ``async def``.
"""

import asyncio
import io
import os
import signal
import sys
import subprocess
from typing import Optional

# Bounded time (in seconds) to wait for a child to exit after we forward a
# termination signal to it before escalating to SIGKILL. Keeps Ctrl-C from
# wedging the launcher when the child ignores the first signal.
_CHILD_SHUTDOWN_TIMEOUT = 5.0


def _signal_child_process(process: "asyncio.subprocess.Process", signum: int) -> None:
    """
    Forward ``signum`` to a child process. When the child leads its own process
    group (created with ``start_new_session=True``), signal the whole group so
    that any grandchildren (e.g. the ``python`` a scheduler script spawns) are
    terminated too; otherwise fall back to signalling the child pid directly.
    """
    try:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signum)
        return
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        process.send_signal(signum)
    except ProcessLookupError:
        pass


async def _reap(process: "asyncio.subprocess.Process", timeout: Optional[float]) -> bool:
    """
    Wait for ``process`` to exit, up to ``timeout`` seconds (``None`` waits
    indefinitely). Returns True if the process exited, False on timeout.
    """
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False


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
    # Create the subprocess in its own session/process group so that we can
    # (a) forward Ctrl-C to the whole child tree without also signalling
    # ourselves, and (b) reliably kill grandchildren the scheduler script may
    # spawn.
    args = [] if len(command) == 1 else command[1:]
    process = await asyncio.create_subprocess_exec(
        command[0],
        *args,
        bufsize=0,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    # Read the stdout and stderr concurrently.
    gather_task = asyncio.gather(
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

    # Intercept SIGINT/SIGTERM at the event-loop level. This replaces Python's
    # default SIGINT->KeyboardInterrupt behavior for the duration of the run, so
    # a Ctrl-C is handled deterministically here instead of surfacing (possibly)
    # inside asyncio.run(): we forward the signal to the child's process group
    # and cancel the readers, guaranteeing the child dies within a bounded time.
    loop = asyncio.get_running_loop()
    received_sig: Optional[int] = None

    def _on_signal(signum: int) -> None:
        nonlocal received_sig
        received_sig = signum
        _signal_child_process(process, signum)
        gather_task.cancel()

    installed = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal, sig)
            installed.append(sig)
        except (NotImplementedError, RuntimeError, ValueError):
            # Signal handlers are unavailable (e.g. not the main thread or an
            # unsupported platform); fall back to plain execution.
            pass

    try:
        try:
            await gather_task
        except asyncio.CancelledError:
            # If this cancellation was not driven by a signal we handled,
            # propagate it (still making sure the child is not left running).
            if received_sig is None:
                _signal_child_process(process, signal.SIGKILL)
                await _reap(process, timeout=None)
                raise
        except BaseException:
            # Any other error: never leave the child running.
            _signal_child_process(process, signal.SIGKILL)
            await _reap(process, timeout=None)
            raise

        if received_sig is not None:
            # We already forwarded the signal to the child; give it a bounded
            # window to exit, then escalate to SIGKILL.
            if not await _reap(process, timeout=_CHILD_SHUTDOWN_TIMEOUT):
                _signal_child_process(process, signal.SIGKILL)
                await _reap(process, timeout=_CHILD_SHUTDOWN_TIMEOUT)
            # Report the death-by-signal using the shell convention 128 + signo
            # so callers propagate a non-zero exit status.
            return 128 + received_sig

        return await process.wait()
    finally:
        for sig in installed:
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError, ValueError):
                pass


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
