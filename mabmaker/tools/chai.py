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


def chai(
    json_path: str,
    output_path: str,
    gpus: int | Iterable[int] | None = None,
    use_msa_server: bool = True,
    msa_server_url: str = "https://api.colabfold.com",
    use_templates_server: bool = False,
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
        Whether to use the MSA server. If ``False``, ESM embeddings will be used instead.

    msa_server_url : str, optional, default="https://api.colabfold.com"
        The URL of the MSA server.

    .. _Chai-1: https://github.com/chaidiscovery/chai-lab/tree/main?tab=readme-ov-file
    .. _AlphaFold3 input JSON file: https://github.com/google-deepmind/alphafold/tree/main/server

    """
    # setup runs
    runs = setup_structure_prediction_run(json_path, output_path)

    # get GPU queue
    gpu_queue = get_gpu_queue(gpus)
    num_gpus = gpu_queue.qsize()

    # run predictions
    futures = []
    with cf.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        for run in runs:
            for seed in run.seeds:
                cmd = _build_chai_command(
                    run=run,
                    output_path=output_path,
                    seed=seed,
                    use_msa_server=use_msa_server,
                    msa_server_url=msa_server_url,
                    use_templates_server=use_templates_server,
                    msa_directory=msa_directory,
                    num_trunk_recycles=num_trunk_recycles,
                    num_trunk_samples=num_trunk_samples,
                    num_diffusion_timesteps=num_diffusion_timesteps,
                    num_diffusion_samples=num_diffusion_samples,
                    low_memory=low_memory,
                )
                futures.append(executor.submit(gpu_worker, cmd, gpu_queue))

        # monitor progress
        with tqdm(
            total=len(futures),
            desc="Chai-1",
            bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}",
        ) as pbar:
            for _ in cf.as_completed(futures):
                pbar.update(1)

    # write prediction logs (stdout and stderr)
    abutils.io.make_dir(os.path.join(output_path, run.name))
    for run, future in zip(runs, futures):
        result = future.result()
        with open(os.path.join(output_path, run.name, "stdout.log"), "w") as f:
            f.write(result.stdout.decode("utf-8"))
        with open(os.path.join(output_path, run.name, "stderr.log"), "w") as f:
            f.write(result.stderr.decode("utf-8"))


def _build_chai_command(
    run: StructurePredictionRun,
    output_path: str,
    seed: int = 42,
    use_msa_server: bool = True,
    msa_server_url: str = "https://api.colabfold.com",
    use_templates_server: bool = False,
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
    run : StructurePredictionRun
        The run to build the command for.

    output_path : str
        The path to the output directory.

    use_msa_server : bool, optional, default=True
        Whether to use the MSA server.

    msa_server_url : str, optional, default="https://api.colabfold.com"
        The URL of the MSA server.

    msa_directory : str, optional
        The path to the directory containing the MSAs.

    num_trunk_recycles : int, optional, default=3
        The number of trunk recycles to perform.

    num_diffusion_timesteps : int, optional, default=200
        The number of diffusion timesteps to use.

    num_diffusion_samples : int, optional, default=5
        The number of diffusion samples to generate.

    Returns
    -------
    command : str
        The command to run.


    .. _Chai-1: https://github.com/chaidiscovery/chai-lab/tree/main?tab=readme-ov-file

    """
    # build Chai-formatted input files
    fasta_path, constraints_path = run.build_chai_input(output_path)

    # build command
    cmd = "chai-lab fold"
    if constraints_path is not None:
        cmd += f" --constraint-path '{constraints_path}'"
    cmd += f" --seed {seed}"
    if use_msa_server:
        cmd += " --use-msa-server"
        cmd += f" --msa-server-url '{msa_server_url}'"
    if msa_directory is not None:
        cmd += f" --msa-directory '{msa_directory}'"
    if use_templates_server:
        cmd += " --use-templates-server"
    cmd += f" --num-trunk-recycles {num_trunk_recycles}"
    cmd += f" --num-trunk-samples {num_trunk_samples}"
    cmd += f" --num-diffn-timesteps {num_diffusion_timesteps}"
    cmd += f" --num-diffn-samples {num_diffusion_samples}"
    if low_memory:
        cmd += " --low-memory"

    # Chai-1 freaks out if the output dir isn't empty, and ours already has
    # the FASTA and constraint files and maybe data from previous seeds
    #
    # TODO: redo logging so that each run logs into its own directory
    # after the prediction is complete
    chai_output_path = os.path.join(output_path, f"seed_{seed}")
    cmd += f" '{fasta_path}' '{chai_output_path}'"
    return cmd
