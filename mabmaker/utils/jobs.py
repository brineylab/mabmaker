# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

import io
import queue
import subprocess as sp
import threading
from contextlib import redirect_stderr, redirect_stdout
from typing import Callable, Iterable

import torch


def gpu_worker(
    cmd: str | Callable,
    gpu_queue: queue.Queue,
) -> None:
    """
    A worker function for running commands on a GPU.

    Parameters
    ----------
    cmd : str
        The command to run, either as a string (which will be called using ``subprocess.run()``) or as a callable function.
        If a string is supplied, it should not include the ``CUDA_VISIBLE_DEVICES`` environment variable, as this will
        be added by the `gpu_worker` function. If a callable is supplied, it should take a single argument: ``device``,
        which accepts the GPU ID formatted as a string (e.g. ``"cuda:0"``).

    gpu_queue : queue.Queue
        A queue of GPU IDs which will be used to determine which GPU to use for the command.

    Returns
    -------
    result : subprocess.CompletedProcess
        The result of the command.

    """
    gpu_id = gpu_queue.get()
    try:
        if isinstance(cmd, str):
            cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} {cmd}"
            result = sp.run(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
        else:
            result = cmd(device=f"cuda:{gpu_id}")
    finally:
        gpu_queue.put(gpu_id)
    return result


def quiet_gpu_worker(
    cmd: str | Callable,
    gpu_queue: queue.Queue,
    return_stdout: bool = False,
    return_stderr: bool = False,
):
    """
    Redirect the stdout and stderr of the current process to an in-memory buffer. If desirted,
    return the stdout and stderr as strings together with the result.

    Parameters
    ----------
    fn : callable
        The function to run.

    *args : tuple
        The arguments to pass to the function.

    return_stdout : bool, optional
        If ``True``, return the stdout as a string.

    return_stderr : bool, optional
        If ``True``, return the stderr as a string.

    **kwargs : dict
        Keyword arguments to pass to the function.

    Returns
    -------
    result : object
        The result of the function. Note that if ``return_stdout`` and ``return_stderr`` are both ``False``,
        the result will be returned as a single item (not a tuple). If either ``return_stdout`` or ``return_stderr``
        are ``True``, the result will be returned as a tuple containing the result and the stdout and/or stderr.

    stdout : str, optional
        The stdout of the function.

    stderr : str, optional
        The stderr of the function.

    """
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        gpu_id = gpu_queue.get()
        try:
            if isinstance(cmd, str):
                cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} {cmd}"
                result = sp.run(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
            else:
                result = cmd(device=f"cuda:{gpu_id}")
        finally:
            gpu_queue.put(gpu_id)
    # if not returning stdout or stderr, return the result as a single item (not a tuple)
    if not return_stdout and not return_stderr:
        return result
    # otherwise, return a tuple of the result and the stdout and/or stderr
    result = (result,)
    if return_stdout:
        result += (stdout.getvalue(),)
    if return_stderr:
        result += (stderr.getvalue(),)
    return result


# class ThreadSilencer:
#     def __init__(
#         self,
#         real_stream,
#         exempt_threads: threading.Thread | Iterable[threading.Thread],
#     ):
#         """
#         Silence a stream (e.g. ``sys.stdout``) for all threads except for the "exempt" threads specified.

#         Parameters
#         ----------
#         real_stream : io.TextIOWrapper
#             The stream to silence.

#         exempt_threads : threading.Thread | Iterable[threading.Thread]
#             The threads to exempt from being silenced. Typically the main thread.

#         """
#         self._real = real_stream
#         self._exempt = (
#             [exempt_threads]
#             if isinstance(exempt_threads, threading.Thread)
#             else exempt_threads
#         )

#     # -- File-like protocol ------------------------------------------------
#     def write(self, data):
#         if threading.current_thread() in self._exempt:
#             self._real.write(data)

#     def flush(self):
#         if threading.current_thread() in self._exempt:
#             self._real.flush()

#     def __getattr__(self, name):  # isatty, fileno, etc.
#         return getattr(self._real, name)


class SubThreadSilencer:
    """
    Forward writes from the main thread to the real stream,
    but capture writes from worker threads in an in-memory buffer.
    """

    def __init__(self, real_stream: io.TextIOWrapper, main_thread: threading.Thread):
        self._real = real_stream  # the real sys.stdout / sys.stderr
        self._main = main_thread  # usually threading.current_thread()
        self._local = threading.local()  # per-thread storage for buffers
        self._all_buffers = {}  # {thread_id: StringIO}

    # ------------------------------------------------------------------ file-like API
    def write(self, data):
        if threading.current_thread() is self._main:
            self._real.write(data)
        else:  # worker thread
            buf = self._get_buffer()
            buf.write(data)

    def flush(self):
        if threading.current_thread() is self._main:
            self._real.flush()
        # workers don't need flushing; StringIO keeps everything in RAM

    # let tqdm / logging / others ask for fileno, isatty, encoding, …
    def __getattr__(self, name):
        return getattr(self._real, name)

    # ------------------------------------------------------------------ helpers
    def _get_buffer(self):
        """Return (and create if needed) this thread's capture buffer."""
        buf = getattr(self._local, "buf", None)
        if buf is None:
            buf = io.StringIO()
            self._local.buf = buf
            self._all_buffers[threading.get_ident()] = buf
        return buf

    def get_logs(self, thread_id=None):
        """
        Return the captured text.

        • If *thread_id* is None, returns {id: text, …} for every thread.
        • Otherwise returns only the text for the chosen thread (or '').
        """
        if thread_id is None:
            return {tid: b.getvalue() for tid, b in self._all_buffers.items()}
        buf = self._all_buffers.get(thread_id)
        return "" if buf is None else buf.getvalue()


def get_gpu_queue(gpus: int | Iterable[int | str] | str | None = None) -> queue.Queue:
    """
    Get a queue of GPU IDs.

    Parameters
    ----------
    gpus : int | Iterable[int] | str | None, optional
        A single GPU ID, a list of GPU IDs, or a string of GPU IDs separated by commas.
        If ``None``, all GPUs will be used.

    Returns
    -------
    gpu_queue : queue.Queue
        A queue of GPU IDs.

    Examples
    --------
    >>> get_gpu_queue(0)
    >>> get_gpu_queue([0, 1])
    >>> get_gpu_queue("0,1")
    >>> get_gpu_queue(None)

    """
    # parse gpu IDs
    if gpus is None:
        gpus = list(range(torch.cuda.device_count()))
    elif isinstance(gpus, str):
        gpus = [int(gpu) for gpu in gpus.split(",")]
    elif isinstance(gpus, int):
        gpus = [gpus]
    else:
        gpus = [int(gpu) for gpu in gpus]
    if not gpus:
        raise ValueError("No GPUs specified.")

    # create the GPU queue
    gpu_queue = queue.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)
    return gpu_queue
