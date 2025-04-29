# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import os
import queue
import subprocess as sp
import sys
import threading
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


def silence_worker():
    """
    Silence the stdout and stderr of the current process. Designed to be used as an initializer for
    a `concurrent.futures.ProcessPoolExecutor`.
    """
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull


class ThreadSilencer:
    def __init__(self, real_stream, main_thread):
        self._real = real_stream
        self._main = main_thread

    # -- File-like protocol ------------------------------------------------
    def write(self, data):
        if threading.current_thread() is self._main:
            self._real.write(data)

    def flush(self):
        if threading.current_thread() is self._main:
            self._real.flush()

    def __getattr__(self, name):  # isatty, fileno, etc.
        return getattr(self._real, name)


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
