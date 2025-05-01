# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import concurrent.futures as cf
import multiprocessing as mp
import os
import sys
import threading
import warnings
from functools import partial
from pathlib import Path
from typing import Iterable

import abutils
from chai_lab.chai1 import run_inference
from tqdm.auto import tqdm

from ..utils.inputs import setup_structure_prediction_run
from ..utils.jobs import SubThreadSilencer, get_gpu_queue, gpu_worker
from ..utils.outputs import process_chai_output
from .msa import precompute_chai_msas

warnings.filterwarnings("ignore", category=DeprecationWarning)

__all__ = ["chai"]


def chai(
    json_path: str,
    output_path: str,
    gpus: int | Iterable[int] | None = None,
    use_msa_server: bool = True,
    msa_server_url: str = "https://api.colabfold.com",
    use_msa_cache: bool = True,
    msa_cache_dir: str = "~/.mabmaker/msa_cache",
    recycle_msa_subsample: int = 0,
    use_templates_server: bool = False,
    template_hits_path: str | None = None,
    msa_directory: str | None = None,
    num_trunk_recycles: int = 3,
    num_trunk_samples: int = 1,
    num_diffusion_timesteps: int = 200,
    num_diffusion_samples: int = 5,
    low_memory: bool = False,
) -> None:
    """
    Structure prediction with `Chai-1`_.

    Parameters
    ----------
    json_path : str
        The path to the JSON file containing the input parameters, or a folder containing
        one or more JSON files. Each JSON file should follow the schema of the
        `AlphaFold3 input JSON file`_, which allows for multiple runs to be specified in
        a single file.

    output_path : str
        The path to the output directory. If it does not exist, it will be created.

    gpus : Union[int, Iterable, None], optional, default=None
        GPU(s) to use. Can be provided as:
            - a single integer: ``0``
            - a comma-separated string of integers: ``"0,1"``
            - a list or tuple of integers: ``[0, 1]``
        If not provided, all available GPUs will be used.

    use_msa_server : bool, optional, default=True
        Whether to use the MSA server. If ``False``, ESM embeddings will be used instead.

    msa_server_url : str, optional, default="https://api.colabfold.com"
        The URL of the MSA server.

    use_msa_cache : bool, optional, default=True
        Whether to use the MSA cache. If ``True``, the cache will be checked for existing
        MSAs before running ``mmseqs2``. If a sequence is not present in the cache, the
        resulting MSA will be saved to the cache. If ``False``, ``mmseqs2`` will be run
        for each sequence and the resulting MSAs will not be cached.

    msa_cache_dir : str, optional, default="~/.mabmaker/msa_cache"
        The path to the MSA cache directory.

    recycle_msa_subsample : int, optional, default=None
        Whether to subsample the MSA for each trunk recycle. If ``0``, no subsampling
        will be performed. If ``>0``, the MSA will be subsampled.

    use_templates_server : bool, optional, default=False
        Whether to use the templates server.

    template_hits_path : str, optional
        The path to the template hits file.

    msa_directory : str, optional
        The path to the directory containing the MSAs.

    num_trunk_recycles : int, optional, default=3
        The number of trunk recycles to perform.

    num_trunk_samples : int, optional, default=1
        The number of trunk samples to generate.

    num_diffusion_timesteps : int, optional, default=200
        The number of diffusion timesteps to use.

    num_diffusion_samples : int, optional, default=5
        The number of diffusion samples to generate.

    low_memory : bool, optional, default=False
        Whether to use low memory mode.

    .. _Chai-1: https://github.com/chaidiscovery/chai-lab/tree/main?tab=readme-ov-file
    .. _AlphaFold3 input JSON file: https://github.com/google-deepmind/alphafold/tree/main/server

    """
    # setup runs
    runs = setup_structure_prediction_run(json_path, output_path)

    # precompute MSAs
    if use_msa_server and msa_directory is None:
        runs = precompute_chai_msas(
            runs=runs,
            base_output_path=output_path,
            msa_server_url=msa_server_url,
            use_msa_cache=use_msa_cache,
            msa_cache_dir=msa_cache_dir,
        )

    # get GPU queue
    gpu_queue = get_gpu_queue(gpus)
    num_gpus = gpu_queue.qsize()

    # silence stdout and stderr for all threads except the main thread
    # because Chai-1 prints a lot of stuff to stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    main_thread = threading.current_thread()
    sys.stdout = SubThreadSilencer(sys.stdout, main_thread)
    sys.stderr = SubThreadSilencer(sys.stderr, main_thread)

    # run predictions
    futures = []
    output_paths = []
    with cf.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        for run in runs:
            run_output_path = os.path.join(output_path, run.name, "raw_output")
            fasta_path, constraint_path = run.build_chai_input(run_output_path)
            msa_directory = msa_directory or getattr(run, "msa_directory", None)
            for seed in run.seeds:
                _output_path = os.path.join(run_output_path, f"seed_{seed}")
                cmd = _build_chai_command(
                    fasta_path=fasta_path,
                    output_path=_output_path,
                    seed=int(seed),
                    constraint_path=constraint_path,
                    use_msa_server=use_msa_server,
                    msa_server_url=msa_server_url,
                    recycle_msa_subsample=recycle_msa_subsample,
                    use_templates_server=use_templates_server,
                    template_hits_path=template_hits_path,
                    msa_directory=msa_directory,
                    num_trunk_recycles=num_trunk_recycles,
                    num_trunk_samples=num_trunk_samples,
                    num_diffusion_timesteps=num_diffusion_timesteps,
                    num_diffusion_samples=num_diffusion_samples,
                    low_memory=low_memory,
                )
                futures.append(
                    executor.submit(
                        gpu_worker,
                        cmd,
                        gpu_queue,
                    )
                )
                output_paths.append(_output_path)

        # monitor progress
        with tqdm(
            total=len(futures),
            desc="Chai-1: ",
            bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}",
        ) as pbar:
            for _ in cf.as_completed(futures):
                pbar.update(1)

    # process outputs
    run_idx = 0
    for run in runs:
        for seed in run.seeds:
            run_path = output_paths[run_idx]
            result = futures[run_idx].result()
            process_chai_output(
                result=result,
                original_path=run_path,
                processed_path=os.path.join(output_path, run.name),
                run_name=run.name,
                seed=seed,
            )
            run_idx += 1

    # stdout and stderr logs
    stdout_logs = sys.stdout.get_logs()
    stderr_logs = sys.stderr.get_logs()

    # restore stdout and stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr

    # write stdout and stderr logs
    log_path = os.path.join(output_path, "logs")
    abutils.io.make_dir(log_path)
    with open(os.path.join(log_path, "stdout.log"), "w") as fh:
        for tid, text in stdout_logs.items():
            fh.write(f"--- output from thread {tid} ---\n{text}\n")
    with open(os.path.join(log_path, "stderr.log"), "w") as fh:
        for tid, text in stderr_logs.items():
            fh.write(f"--- output from thread {tid} ---\n{text}\n")


def _build_chai_command(
    fasta_path: str,
    output_path: str,
    seed: int = 42,
    constraint_path: str | None = None,
    use_msa_server: bool = True,
    msa_server_url: str = "https://api.colabfold.com",
    recycle_msa_subsample: int | None = None,
    use_templates_server: bool = False,
    template_hits_path: str | None = None,
    msa_directory: str | None = None,
    num_trunk_recycles: int = 3,
    num_trunk_samples: int = 1,
    num_diffusion_timesteps: int = 200,
    num_diffusion_samples: int = 5,
    low_memory: bool = False,
) -> str:
    """
    Build a command for running `Chai-1`_.

    Parameters
    ----------
    fasta_path : str
        The path to the FASTA file.

    output_path : str
        The path to the output directory.

    seed : int, optional, default=42
        The seed to use for the prediction.

    constraint_path : str, optional
        The path to the constraint file.

    use_msa_server : bool, optional, default=True
        Whether to use the MSA server.

    msa_server_url : str, optional, default="https://api.colabfold.com"
        The URL of the MSA server.

    recycle_msa_subsample : int, optional, default=None
        Whether to subsample the MSA for each trunk recycle. If ``0``, no subsampling
        will be performed. If ``>0``, the MSA will be subsampled.

    use_templates_server : bool, optional, default=False
        Whether to use the templates server.

    template_hits_path : str, optional
        The path to the template hits file.

    msa_directory : str, optional
        The path to the directory containing the MSAs.

    num_trunk_recycles : int, optional, default=3
        The number of trunk recycles to perform.

    num_trunk_samples : int, optional, default=1
        The number of trunk samples to generate.

    num_diffusion_timesteps : int, optional, default=200
        The number of diffusion timesteps to use.

    num_diffusion_samples : int, optional, default=5
        The number of diffusion samples to generate.

    low_memory : bool, optional, default=False
        Whether to use low memory mode.

    Returns
    -------
    command : str
        The command to run.


    .. _Chai-1: https://github.com/chaidiscovery/chai-lab/tree/main?tab=readme-ov-file

    """
    # embeddings vs MSA server vs MSA directory
    if use_msa_server or msa_directory is not None:
        use_esm_embeddings = False
        # msa_directory is exclusive with use_msa_server
        if msa_directory is not None:
            use_msa_server = False
    else:
        use_esm_embeddings = True

    # build partial function call (without device arg)
    cmd = partial(
        run_inference,
        fasta_file=Path(fasta_path),
        output_dir=Path(output_path),
        use_esm_embeddings=use_esm_embeddings,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_directory=msa_directory,
        constraint_path=constraint_path,
        use_templates_server=use_templates_server,
        template_hits_path=template_hits_path,
        recycle_msa_subsample=recycle_msa_subsample,
        num_trunk_recycles=num_trunk_recycles,
        num_diffn_timesteps=num_diffusion_timesteps,
        num_diffn_samples=num_diffusion_samples,
        num_trunk_samples=num_trunk_samples,
        seed=seed,
        low_memory=low_memory,
    )
    return cmd
