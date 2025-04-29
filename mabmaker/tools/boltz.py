# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import concurrent.futures as cf
import os
import sys
from typing import Iterable

import abutils
from tqdm.auto import tqdm

from ..utils.inputs import StructurePredictionRun, setup_structure_prediction_run
from ..utils.jobs import get_gpu_queue, gpu_worker
from ..utils.outputs import process_boltz_output

__all__ = ["boltz"]


def boltz(
    json_path: str,
    output_path: str,
    gpus: int | Iterable[int] | None = None,
    use_msa_server: bool = True,
    msa_server_url: str = "https://api.colabfold.com",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 5,
    step_scale: float = 1.638,
    output_format: str = "mmcif",
    override: bool = False,
    msa_pairing_strategy: str = "greedy",
    write_full_pae: bool = True,
    write_full_pde: bool = True,
    cache: str = "~/.boltz",
) -> None:
    """
    Structure prediction with `Boltz-1`_.

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
        Whether to use the MSA server.

    msa_server_url : str, optional, default="https://api.colabfold.com"
        The URL of the MSA server.

    recycling_steps : int, optional, default=3
        The number of recycling steps.

    sampling_steps : int, optional, default=200
        The number of sampling steps.

    diffusion_samples : int, optional, default=5
        The number of diffusion samples.

    step_scale : float, optional, default=1.638
        The step scale. The step size is related to the temperature at which the diffusion
        process samples the distribution. The lower the step_scale, the higher the diversity
        among samples (recommended between 1 and 2).

    output_format : str, optional, default="mmcif"
        The output format. Options are ``"mmcif"`` and ``"pdb"``.

    override : bool, optional, default=False
        Whether to override existing output files if found.

    msa_pairing_strategy : str, optional, default="greedy"
        The MSA pairing strategy. Used only if ``use_msa_server`` is ``True``. Options are
        ``"greedy"`` and ``"complete"``.

    write_full_pae : bool, optional, default=False
        Whether to write the full predicted aligned error (PAE) matrix as a file.

    write_full_pde : bool, optional, default=False
        Whether to write the full predicted docking error (PDE) matrix as a file.

    cache : str, optional, default="~/.boltz"
        The path to the cache directory, which contains model weights and other resources.

    .. _Boltz-1: https://github.com/jwohlwend/boltz
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
            # Boltz accepts a single seed, so we need a separate job for each seed
            for seed in run.seeds:
                _output_path = os.path.join(run_output_path, f"seed_{seed}")
                cmd = _build_boltz_command(
                    run=run,
                    output_path=_output_path,
                    use_msa_server=use_msa_server,
                    msa_server_url=msa_server_url,
                    seed=seed,
                    recycling_steps=recycling_steps,
                    sampling_steps=sampling_steps,
                    diffusion_samples=diffusion_samples,
                    step_scale=step_scale,
                    output_format=output_format,
                    override=override,
                    msa_pairing_strategy=msa_pairing_strategy,
                    write_full_pae=write_full_pae,
                    write_full_pde=write_full_pde,
                    cache=cache,
                )
                futures.append(executor.submit(gpu_worker, cmd, gpu_queue))
                output_paths.append(_output_path)

    # monitor progress
    with tqdm(
        total=len(futures),
        desc="Boltz-1",
        bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}",
    ) as pbar:
        for _ in cf.as_completed(futures):
            pbar.update(1)

    # assemble the outputs into a standardized directory schema
    run_idx = 0
    for run in runs:
        for seed in run.seeds:
            run_path = output_paths[run_idx]
            result = futures[run_idx].result()
            stdout = result.stdout.decode("utf-8")
            stderr = result.stderr.decode("utf-8")
            process_boltz_output(
                original_path=run_path,
                processed_path=os.path.join(output_path, run.name),
                run_name=run.name,
                seed=seed,
                stdout=stdout,
                stderr=stderr,
            )
            run_idx += 1


def _build_boltz_command(
    run: StructurePredictionRun,
    output_path: str,
    seed: int = 42,
    use_msa_server: bool = True,
    msa_server_url: str = "https://api.colabfold.com",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 5,
    step_scale: float = 1.638,
    output_format: str = "mmcif",
    override: bool = False,
    msa_pairing_strategy: str = "greedy",
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    cache: str = "~/.boltz",
) -> str:
    """
    Build a command for running `Boltz-1`_.

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


    .. _Boltz-1: https://github.com/jwohlwend/boltz

    """
    # build Boltz-formatted input YAML file
    yaml_path = os.path.join(output_path, f"{run.name}.yaml")
    run.build_boltz_input(yaml_path)

    # build command
    cmd = f"boltz predict '{yaml_path}'"
    cmd += f" --out_dir '{output_path}'"
    cmd += f" --seed {seed}"
    if use_msa_server:
        cmd += " --use_msa_server"
        cmd += f" --msa_server_url '{msa_server_url}'"
        cmd += f" --msa_pairing_strategy '{msa_pairing_strategy}'"
    cmd += f" --recycling_steps {recycling_steps}"
    cmd += f" --sampling_steps {sampling_steps}"
    cmd += f" --diffusion_samples {diffusion_samples}"
    cmd += f" --step_scale {step_scale}"
    cmd += f" --output_format {output_format}"
    if override:
        cmd += " --override"
    if write_full_pae:
        cmd += " --write_full_pae"
    if write_full_pde:
        cmd += " --write_full_pde"
    cmd += f" --cache '{cache}'"
    return cmd
