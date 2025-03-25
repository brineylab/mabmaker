# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import concurrent.futures as cf
import os
from typing import Iterable

import abutils
from tqdm.auto import tqdm

from ..utils.inputs import StructurePredictionRun, setup_structure_prediction_run
from ..utils.jobs import get_gpu_queue, gpu_worker
from ..utils.outputs import process_protenix_output

PROTENIX_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "Protenix")
)

__all__ = ["protenix"]


def protenix(
    json_path: str,
    output_path: str,
    gpus: int | Iterable[int] | None = None,
    use_msa_server: bool = True,
    num_trunk_recycles: int = 3,
    num_trunk_samples: int = 1,
    num_diffusion_timesteps: int = 200,
    num_diffusion_samples: int = 5,
) -> None:
    """
    Structure prediction with `Protenix`_.

    Parameters
    ----------
    json_path : str
        The path to the JSON file containing the input parameters, or a folder containing
        one or more JSON files. Each JSON file should follow the schema of the
        `AlphaFold3 input JSON file`_, which allows for multiple runs to be specified in
        a single file.

    output_path : str
        The path to the output directory. If it does not exist, it will be created.

    numbering_reference : str | None, optional
        The path to a PDB file containing the numbering reference. If provided, the
        residues in the input JSON files will be renumbered based on the numbering in
        the PDB file.

    gpus : Union[int, Iterable, None], optional, default=None
        GPU(s) to use. Can be provided as:
            - a single integer: ``0``
            - a comma-separated string of integers: ``"0,1"``
            - a list or tuple of integers: ``[0, 1]``
        If not provided, all available GPUs will be used.

    use_msa_server : bool, optional, default=True
        Whether to use the MSA server.

    msa_server_url : str, optional, default="https://api.colabfold.com"
        The URL of the MSA server.

    .. _Protenix: https://github.com/bytedance/Protenix
    .. _AlphaFold3 input JSON file: https://github.com/google-deepmind/alphafold/tree/main/server

    """
    # setup runs
    runs = setup_structure_prediction_run(json_path, output_path)

    # get GPU queue
    gpu_queue = get_gpu_queue(gpus)
    num_gpus = gpu_queue.qsize()

    # run predictions
    futures = []
    output_paths = []
    with cf.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        for run in runs:
            run_output_path = os.path.join(output_path, run.name, "raw_output")
            cmd = _build_protenix_command(
                run=run,
                output_path=run_output_path,
                use_msa_server=use_msa_server,
            )
            futures.append(executor.submit(gpu_worker, cmd, gpu_queue))
            output_paths.append(run_output_path)

        # monitor progress
        with tqdm(
            total=len(futures),
            desc="Protenix",
            bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}",
        ) as pbar:
            for _ in cf.as_completed(futures):
                pbar.update(1)

    # write prediction logs (stdout and stderr)
    abutils.io.make_dir(os.path.join(output_path, run.name))
    for run, run_path, future in zip(runs, output_paths, futures):
        result = future.result()
        stdout = result.stdout.decode("utf-8")
        stderr = result.stderr.decode("utf-8")
        process_protenix_output(
            original_path=run_path,
            processed_path=os.path.join(output_path, run.name),
            run_name=run.name,
            stdout=stdout,
            stderr=stderr,
        )


def _build_protenix_command(
    run: StructurePredictionRun,
    output_path: str,
    use_msa_server: bool = True,
) -> str:
    """
    Build a command for running `Protenix`_.

    Parameters
    ----------
    run : StructurePredictionRun
        The run to build the command for.

    output_path : str
        The path to the output directory.

    use_msa_server : bool, optional, default=True
        Whether to use the MSA server.

    Returns
    -------
    command : str
        The command to run.


    .. _Protenix: https://github.com/bytedance/Protenix

    """
    seeds = ",".join(run.seeds)

    # build Protenix-formatted input JSON file
    json_path = os.path.join(output_path, f"{run.name}.json")
    run.build_protenix_input(json_path)

    # build command
    cmd = "protenix predict"
    cmd += f" --input '{json_path}'"
    cmd += f" --out_dir '{output_path}'"
    cmd += f" --seeds {seeds}"
    if use_msa_server:
        cmd += " --use_msa_server"
    return cmd
