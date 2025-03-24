# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

import concurrent.futures
import json
import os
import re
import shutil

# import subprocess as sp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import abutils
import torch
from magika import Magika
from natsort import natsorted
from tqdm.auto import tqdm

from ..utils.jobs import get_gpu_queue, gpu_worker

__all__ = ["ligandmpnn", "expand_residue_ranges"]

LIGAND_MPNN_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "LigandMPNN")
)


@dataclass
class LigandMPNNParameters:
    pdb_path: Union[str, Iterable]
    output_dir: str
    model_type: str
    model_checkpoint: str
    seed: int
    temperature: float
    fixed_residues: Optional[str]
    redesigned_residues: Optional[str]
    bias_aa: Optional[str]
    bias_aa_per_residue_dict: Optional[Dict[str, float]]
    omit_aa: Optional[str]
    omit_aa_per_residue_dict: Optional[Dict[str, str]]
    chains_to_design: Optional[str]
    parse_these_chains_only: Optional[str]
    use_side_chain_context: bool
    use_atom_context: bool
    batch_size: int
    num_batches: int
    save_stats: bool
    verbose: bool

    def __post_init__(self):
        """
        Post-initialization processing of per-position bias and omit AA data.
        """
        # write AA bias and omit data to JSON files, return file paths
        self.bias_aa_per_residue = self._process_bias_aa_per_residue_dict()
        self.omit_aa_per_residue = self._process_omit_aa_per_residue_dict()

    def _process_bias_aa_per_residue_dict(self) -> Optional[str]:
        """
        Writes the bias_aa_per_residue_dict to a JSON file.

        Returns
        -------
        Optional[str]
            Path to the bias_aa_per_residue JSON file. If `bias_aa_per_residue_dict` is not provided, then the function
            returns ``None``.
        """
        if self.bias_aa_per_residue_dict is not None:
            json_file = os.path.join(self.output_dir, "bias_aa_per_residue.json")
            with open(json_file, "w") as f:
                json.dump(self.bias_aa_per_residue_dict, f)
            return json_file
        return None

    def _process_omit_aa_per_residue_dict(self) -> Optional[str]:
        """
        Writes the omit_aa_per_residue_dict to a JSON file.

        Returns
        -------
        Optional[str]
            Path to the omit_aa_per_residue JSON file. If `omit_aa_per_residue_dict` is not provided, then the function
            returns ``None``.
        """
        if self.omit_aa_per_residue_dict is not None:
            json_file = os.path.join(self.output_dir, "omit_aa_per_residue.json")
            with open(json_file, "w") as f:
                json.dump(self.omit_aa_per_residue_dict, f)
            return json_file
        return None


def ligandmpnn(
    pdb_path: Union[str, Iterable],
    output_dir: str,
    model_type: str = "ligand_mpnn",
    model_checkpoint: Optional[str] = None,
    seed: Union[int, str, Iterable] = 42,
    temperature: Union[float, str, Iterable] = 0.1,
    gpus: Union[int, str, Iterable, None] = None,
    bias_aa: Union[str, Dict[str, float], None] = None,
    bias_aa_per_residue: Union[str, Dict[str, float], None] = None,
    omit_aa: Union[str, Dict[str, str], None] = None,
    omit_aa_per_residue: Union[str, Dict[str, str], None] = None,
    fixed_residues: Union[str, Dict[str, str], None] = None,
    redesigned_residues: Union[str, Iterable, None] = None,
    chains_to_design: Union[str, Iterable, None] = None,
    parse_these_chains_only: Union[str, Iterable, None] = None,
    use_side_chain_context: bool = True,
    use_atom_context: bool = False,
    batch_size: int = 32,
    num_batches: int = 1,
    save_stats: bool = True,
    started_from_cli: bool = False,
    verbose: bool = True,
    debug: bool = False,
):
    """
    Run LigandMPNN_ on a PDB file, a list of PDB files, or a directory of PDB files.

    .. note::
        If multiple temperature or seed values are provided, then each PDB file will be
        processed with every combination of temperature and seed.

    Parameters
    ----------
    pdb_path : Union[str, Iterable]
        Path to a PDB file, a list of PDB files, or a directory of PDB files. Required.

    output_dir : str
        Path to the output directory. Required.

    model_type : str, optional, default="ligand_mpnn"
        Type of model to use. Must be one of:
          - "ligand_mpnn": LigandMPNN model
          - "protein_mpnn": ProteinMPNN model

    model_checkpoint : Optional[str], optional, default=None
        Checkpoint of the model to use, excluding the model name and file extension. For example,
        if the full name of the weights file for the desired model checkpoint is ``"ligandmpnn_v_32_010_25.pt"``,
        then `model_checkpoint` should be ``"v_32_010_25"``. If not provided, the default checkpoints are:
            - ligand_mpnn: ``"v_32_010_25"``
            - protein_mpnn: ``"v_48_020"``

    seeds : Union[int, Iterable], optional, default=42
        Random seed(s) to use. Can be provided as:
            - a single integer: ``42``
            - a comma-separated string of integers: ``"42,43"``
            - a list or tuple of integers: ``[42, 43]``
        If multiple seed values are provided, each PDB file will be processed with every combination of temperature and seed.

    temperature : Union[float, Iterable], optional, default=0.1
        Temperature(s) to use. Can be provided as:
            - a single float: ``0.1``
            - a comma-separated string of floats: ``"0.1,0.2"``
            - a list or tuple of floats: ``[0.1, 0.2]``
        If multiple temperature values are provided, each PDB file will be processed with every combination of temperature and seed.

    gpus : Union[int, Iterable, None], optional, default=None
        GPU(s) to use. Can be provided as:
            - a single integer: ``0``
            - a comma-separated string of integers: ``"0,1"``
            - a list or tuple of integers: ``[0, 1]``
        If not provided, all available GPUs will be used.

    bias_aa : Union[str, Dict[str, float], None], optional, default=None
        Bias the generation of AAs. Can be provided as:
            - a string of comma-separated AA IDs and their biases: ``"A:-1.024,P:2.34,C:-12.34"``
            - a dictionary mapping AA IDs to their biases: ``{"A": -1.024, "P": 2.34, "C": -12.34}``
            - a file path to a JSON file containing a dictionary mapping AA IDs to their biases, which will be applied to all PDB files: ``{"A": -1.024, "P": 2.34, "C": -12.34}``
            - a file path to a JSON file containing a dictionary mapping PDB file names (not the full path) to dictionaries of AA IDs and their biases: ``{"a1b2.pdb": {"A": -1.024, "P": 2.34, "C": -12.34}}``

    bias_aa_per_residue : Union[str, Dict[str, float], None], optional, default=None
        Bias the generation of AAs per residue. Can be provided as:
            - a string of comma-separated residue IDs and their biases: ``"A12:A:-1.024,P:2.34,C:-12.34"``
            - a dictionary mapping residue IDs to their biases: ``{"A12": {"A": -1.024, "P": 2.34, "C": -12.34}}``
            - a file path to a JSON file containing a dictionary mapping residue IDs to their biases, which will be applied to all PDB files: ``{"A12": {"A": -1.024, "P": 2.34, "C": -12.34}}``
            - a file path to a JSON file containing a dictionary mapping PDB file names (not the full path) to dictionaries of residue IDs and their biases: ``{"a1b2.pdb": {"A12": {"A": -1.024, "P": 2.34, "C": -12.34}}}``

    omit_aa : Union[str, Dict[str, str], None], optional, default=None
        Omit the generation of AAs. Can be provided as:
            - a string of concatenated AAs to omit: ``"ACG"``
            - a file path to a text file containing concatenated AAs to omit: ``"ACG"``
            - a file path to a JSON file containing a dictionary mapping PDB file names (not the full path) to dictionaries of concatenated AAs to omit: ``{"a1b2.pdb": "ACG"}``

    omit_aa_per_residue : Union[str, Dict[str, str], None], optional, default=None
        Omit the generation of AAs per residue. Can be provided as:
            - a string of comma-separated residue IDs: ``"A12:ACG"``
            - a dictionary mapping residue IDs to their biases: ``{"A12": "ACG"}``
            - a file path to a JSON file containing a dictionary mapping residue IDs to their biases, which will be applied to all PDB files: ``{"A12": "ACG"}``
            - a file path to a JSON file containing a dictionary mapping PDB file names (not the full path) to dictionaries of per-residue AA omissions: ``{"a1b2.pdb": {"A12": "ACG"}}``

    fixed_residues : Union[str, Dict[str, str], None], optional, default=None
        Fix residues at the provided positions (redesign all other residues). Can be provided as:
            - a string of space- or comma-separated residue IDs: ``"A12-14 B2 B25"`` or ``"A12-14,B2,B25"``
            - a list or tuple of residue IDs: ``["A12", "A13", "A14", "B2", "B25"]``
            - a file path to a text file containing space- or comma-separated residue IDs: ``"A12-14 B2 B25"`` or ``"A12-14,B2,B25"``
            - a file path to a JSON file containing a dictionary mapping PDB file names (not the full path) to space- or comma-separated residue IDs: ``{"a1b2.pdb": "A12-14 B2 B25"}``

    redesigned_residues : Union[str, Iterable, None], optional, default=None
        Residues to redesign (all other residues will be fixed). Can be provided as:
          - a string of space- or comma-separated residue IDs: ``"A12-14 B2 B25"`` or ``"A12-14,B2,B25"``
          - a list or tuple of residue IDs: ``["A12", "A13", "A14", "B2", "B25"]``
          - a file path to a text file containing space- or comma-separated residue IDs: ``"A12-14 B2 B25"`` or ``"A12-14,B2,B25"``
          - a file path to a JSON file containing a dictionary mapping PDB file names (not the full path) to space- or comma-separated residue IDs: ``{"a1b2.pdb": "A12-14 B2 B25"}``

    chains_to_design : Union[str, Iterable, None], optional, default=None
        Chains to design. Can be provided as:
            - a string of comma-separated chain IDs: ``"A,B,C"``
            - a list or tuple of chain IDs: ``["A", "B", "C"]``
            - a file path to a text file containing comma-separated chain IDs: ``"A,B,C"``
            - a file path to a JSON file containing a dictionary mapping PDB file names (not the full path) to comma-separated chain IDs: ``{"a1b2.pdb": "A,B,C"}``

    parse_these_chains_only : Union[str, Iterable, None], optional, default=None
        Chains to parse. Can be provided as:
            - a string of comma-separated chain IDs: ``"A,B,C"``
            - a list or tuple of chain IDs: ``["A", "B", "C"]``
            - a file path to a text file containing comma-separated chain IDs: ``"A,B,C"``
            - a file path to a JSON file containing a dictionary mapping PDB file names (not the full path) to comma-separated chain IDs: ``{"a1b2.pdb": "A,B,C"}``

    use_side_chain_context : bool, optional, default=True
        Whether to use side chain context. Only used if `model_type` is ``"ligand_mpnn"``.

    use_atom_context : bool, optional, default=False
        Whether to use atom context. Only used if `model_type` is ``"ligand_mpnn"``.

    batch_size : int, optional, default=32
        Number of sequences to generate per pass.

    num_batches : int, optional, default=1
        Number of times to design sequences using the chosen `batch size`.

    weights_path : Optional[str], optional, default=None
        Path to the weights file.

    save_stats : bool, optional, default=True
        Whether to save the stats.

    started_from_cli : bool, optional, default=False
        Whether the function was called from the CLI.

    verbose : bool, optional, default=True
        Whether to print verbose output.

    debug : bool, optional, default=False
        Whether to print additional output potentially useful for debugging.


    .. _LigandMPNN: https://www.biorxiv.org/content/10.1101/2023.12.22.573103v1

    """
    # output directory
    abutils.io.make_dir(output_dir)

    # setup logging
    global logger
    if started_from_cli:
        abutils.log.setup_logging(
            logfile=os.path.join(output_dir, "ligandmpnn.log"),
            add_stream_handler=verbose,
            single_line_handler=False,
            print_log_location=False,
            debug=debug,
        )
        logger = abutils.log.get_logger(
            name="ligandmpnn",
            add_stream_handler=verbose,
            single_line_handler=False,
        )
    elif verbose:
        logger = abutils.log.NotebookLogger(verbose=verbose, end="")
    else:
        logger = abutils.log.null_logger()

    if verbose and started_from_cli:
        log_splash()
        log_arguments(
            pdb_path=pdb_path,
            output_dir=output_dir,
            model_type=model_type,
            model_checkpoint=model_checkpoint,
            seed=seed,
            temperature=temperature,
            gpus=gpus,
            bias_aa=bias_aa,
            bias_aa_per_residue=bias_aa_per_residue,
            omit_aa=omit_aa,
            omit_aa_per_residue=omit_aa_per_residue,
            fixed_residues=fixed_residues,
            redesigned_residues=redesigned_residues,
            chains_to_design=chains_to_design,
            parse_these_chains_only=parse_these_chains_only,
            use_side_chain_context=use_side_chain_context,
            use_atom_context=use_atom_context,
            batch_size=batch_size,
            num_batches=num_batches,
            save_stats=save_stats,
            started_from_cli=started_from_cli,
            verbose=verbose,
            debug=debug,
        )
        # logger.info("")
        # logger.info("==============")
        # logger.info("  LIGANDMPNN  ")
        # logger.info("==============")
        # logger.info("")

    # PDB path(s)
    if isinstance(pdb_path, str):
        if os.path.isdir(pdb_path):
            pdbs = abutils.io.list_files(pdb_path, extension=".pdb")
        else:
            pdbs = [pdb_path]
    else:
        pdbs = natsorted(pdb_path)
    pdbs = [os.path.abspath(pdb) for pdb in pdbs if os.path.isfile(pdb)]
    if len(pdbs) == 0:
        raise FileNotFoundError(f"No PDB files found in {pdb_path}")
    pdb_names = [os.path.basename(pdb).rstrip(".pdb") for pdb in pdbs]

    # log PDB file info (only if started from CLI)
    if started_from_cli:
        log_pdb_file_info(pdbs=pdb_names)

    # model
    model_type = model_type.lower()
    if model_type not in ["ligand_mpnn", "protein_mpnn"]:
        raise ValueError(
            f"Invalid model type: {model_type}. Must be one of: ['ligand_mpnn', 'protein_mpnn']"
        )

    # model checkpoint
    if model_checkpoint is None:
        if model_type == "ligand_mpnn":
            model_checkpoint = "v_32_010_25"
        elif model_type == "protein_mpnn":
            model_checkpoint = "v_48_020"
    model_checkpoint = model_checkpoint.lower()
    checkpoint_path = _get_model_checkpoint(model_type, model_checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    # gpus
    if gpus is None:
        if torch.cuda.is_available():
            gpus = list(range(torch.cuda.device_count()))
        else:
            raise ValueError("No GPUs available")
    elif isinstance(gpus, int):
        gpus = [gpus]
    elif isinstance(gpus, str):
        gpus = [int(g) for g in gpus.split(",")]

    # seed(s)
    if isinstance(seed, int):
        seeds = [seed]
    elif isinstance(seed, str):
        if os.path.isfile(seed):
            with open(seed, "r") as f:
                seeds = [int(s) for s in f.read().split(",")]
        else:
            seeds = [int(s) for s in seed.split(",")]
    else:
        seeds = sorted(seed)

    # temperature(s)
    if isinstance(temperature, float):
        temperatures = [temperature]
    elif isinstance(temperature, str):
        if os.path.isfile(temperature):
            with open(temperature, "r") as f:
                temperatures = [float(t) for t in f.read().split(",")]
        else:
            temperatures = [float(t) for t in temperature.split(",")]
    else:
        temperatures = sorted(temperature)

    # chains
    chains_to_design = _process_chain_or_residue_data(
        chains_to_design, pdbs=pdb_names, sep=","
    )
    parse_these_chains_only = _process_chain_or_residue_data(
        parse_these_chains_only, pdbs=pdb_names, sep=","
    )

    # fixed/redesigned residues
    fixed_residues_dict = _process_chain_or_residue_data(
        fixed_residues, pdbs=pdb_names, sep=" "
    )
    for k, v in fixed_residues_dict.items():  # expand residue ranges
        fixed_residues_dict[k] = expand_residue_ranges(v)

    redesigned_residues_dict = _process_chain_or_residue_data(
        redesigned_residues, pdbs=pdb_names, sep=" "
    )
    for k, v in redesigned_residues_dict.items():  # expand residue ranges
        redesigned_residues_dict[k] = expand_residue_ranges(v)

    # bias AA
    if isinstance(bias_aa, dict):
        bias_aa = ",".join([f"{str(k)}:{str(v)}" for k, v in bias_aa.items()])

    # bias AA per residue
    # returns a dict with PDB file paths as keys and per-residue bias AA data as values
    bias_aa_per_residue_dict = _process_per_residue_data(
        bias_aa_per_residue, pdbs=pdb_names
    )
    if bias_aa_per_residue_dict:
        if not all([pdb in bias_aa_per_residue_dict for pdb in pdb_names]):
            raise ValueError(
                f"Supplied bias_aa_per_residue ({bias_aa_per_residue}) does not have information for all input PDB files"
            )

    # omit AA
    if isinstance(omit_aa, (list, tuple)):
        omit_aa = "".join([str(r) for r in omit_aa])

    # omit AA per residue
    # returns a dict with PDB file paths as keys and per-residue omit AA data as values
    omit_aa_per_residue_dict = _process_per_residue_data(
        omit_aa_per_residue, pdbs=pdb_names
    )
    if omit_aa_per_residue_dict:
        if not all([pdb in omit_aa_per_residue_dict for pdb in pdb_names]):
            raise ValueError(
                f"Supplied omit_aa_per_residue ({omit_aa_per_residue}) does not have information for all input PDB files"
            )

    # set up GPU queue and thread pool
    gpu_queue = get_gpu_queue(gpus)

    # run
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        for pdb in pdbs:
            # copy input PDB to the output directory
            pdb_name = os.path.basename(pdb).rstrip(".pdb")
            pdb_output_dir = os.path.join(output_dir, pdb_name)
            abutils.io.make_dir(pdb_output_dir)
            shutil.copy(pdb, pdb_output_dir)
            # run
            for temperature in temperatures:
                for seed in seeds:
                    # set up output directory
                    output = os.path.join(
                        output_dir,
                        pdb_name,
                        f"temperature={temperature}",
                        f"seed={seed}",
                    )
                    abutils.io.make_dir(output)
                    # parameters
                    params = LigandMPNNParameters(
                        pdb_path=pdb,
                        output_dir=output,
                        model_type=model_type,
                        model_checkpoint=checkpoint_path,
                        seed=seed,
                        temperature=temperature,
                        fixed_residues=fixed_residues_dict.get(pdb_name, None),
                        redesigned_residues=redesigned_residues_dict.get(
                            pdb_name, None
                        ),
                        bias_aa=bias_aa,
                        bias_aa_per_residue_dict=bias_aa_per_residue_dict.get(
                            pdb_name, None
                        ),
                        omit_aa=omit_aa,
                        omit_aa_per_residue_dict=omit_aa_per_residue_dict.get(
                            pdb_name, None
                        ),
                        chains_to_design=chains_to_design.get(pdb_name, None),
                        parse_these_chains_only=parse_these_chains_only.get(
                            pdb_name, None
                        ),
                        use_side_chain_context=use_side_chain_context,
                        use_atom_context=use_atom_context,
                        batch_size=batch_size,
                        num_batches=num_batches,
                        save_stats=save_stats,
                        verbose=verbose,
                    )
                    log_params(params)
                    # get cmd
                    cmd = _get_ligandmpnn_cmd(params=params)

                    # submit cmd to thread pool
                    futures.append(executor.submit(gpu_worker, cmd, gpu_queue))
        # monitor progress
        with tqdm(
            total=len(futures),
            desc="LigandMPNN",
            bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}",
        ) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)


def expand_residue_ranges(
    input_source: str, output_file: Optional[str] = None
) -> Optional[str]:
    """
    Expand residue ranges into individual residue identifiers.

    Parameters
    ----------
    input_source : str
        Either a path to a file containing residue ranges or a string containing the ranges.

    output_file : str, optional
        Path to write the expanded residues. If ``None``, returns the expanded residues as a string.

    Returns
    -------
    str or None
        If `output_file` is ``None``, returns the expanded residues as a space-separated string.
        Otherwise, writes to the file and returns ``None``.

    Example
    -------
    Input: "A1-A5,B10,C15-C20" or "A1-5 B10 C15-20"
    Returns: "A1 A2 A3 A4 A5 B10 C15 C16 C17 C18 C19 C20"

    """
    # determine whether input_source is a file path or a string containing residue ranges
    if os.path.isfile(input_source):
        try:
            with open(input_source, "r") as f:
                input_str = f.read().strip()
        except FileNotFoundError:
            print(f"Error: Input file '{input_source}' not found.")
            return None
    else:
        input_str = input_source.strip()

    output = []
    # split the input string by commas or whitespace
    items = re.split(r"[,\s]+", input_str)

    # iterate over each item
    for item in items:
        item = item.strip()
        if not item:  # skip empty items
            continue

        # check if it's a range (contains '-')
        if "-" in item:
            parts = item.split("-")
            if len(parts) != 2:
                print(f"Warning: Invalid range format '{item}'. Skipping.")
                continue

            start_part, end_part = parts

            # extract chain from the start part
            chain_match = re.match(r"^([A-Z])", start_part)
            if not chain_match:
                print(
                    f"Warning: No chain identifier found in '{start_part}'. Skipping."
                )
                continue

            chain = chain_match.group(1)

            # extract start position
            start_num_match = re.search(r"(\d+)", start_part)
            if not start_num_match:
                print(f"Warning: No residue number found in '{start_part}'. Skipping.")
                continue

            start_num = int(start_num_match.group(1))

            # extract end position, handling both "A1-A10" and "A1-10" formats
            # check if the end part starts with a chain identifier
            end_chain_match = re.match(r"^([A-Z])", end_part)

            if end_chain_match:
                # format "A1-A10" - extract only the number
                end_num_match = re.search(r"(\d+)", end_part)
                if not end_num_match:
                    print(
                        f"Warning: No residue number found in '{end_part}'. Skipping."
                    )
                    continue
                end_num = int(end_num_match.group(1))
            else:
                # format "A1-10" - the end part is just the number
                try:
                    end_num = int(end_part)
                except ValueError:
                    print(f"Warning: Invalid end number '{end_part}'. Skipping.")
                    continue

            # generate the sequence and append to output
            for i in range(start_num, end_num + 1):
                output.append(f"{chain}{i}")
        else:
            # single residue case
            if re.match(
                r"^[A-Z]\d+$", item
            ):  # ensure it has a valid format like A1, B22
                output.append(item)
            else:
                print(f"Warning: Invalid residue format '{item}'. Skipping.")
                continue

    # join the output with spaces
    expanded_residues = " ".join(output)

    # write or return
    if output_file:
        try:
            with open(output_file, "w") as f:
                f.write(expanded_residues)
            # print(f"Expanded residues written to {output_file}")
            return None
        except Exception as e:
            print(f"Error writing to output file: {e}")
            return expanded_residues
    else:
        return expanded_residues


def log_params(params: LigandMPNNParameters) -> None:
    param_str = ""
    param_str += f"PDB_PATH: {params.pdb_path}\n"
    param_str += f"OUTPUT_DIR: {params.output_dir}\n"
    param_str += f"MODEL_TYPE: {params.model_type}\n"
    param_str += f"MODEL_CHECKPOINT: {params.model_checkpoint}\n"
    param_str += f"SEED: {params.seed}\n"
    param_str += f"TEMPERATURE: {params.temperature}\n"
    param_str += f"BIAS_AA: {params.bias_aa}\n"
    param_str += f"OMIT_AA: {params.omit_aa}\n"
    param_str += f"CHAINS_TO_DESIGN: {params.chains_to_design}\n"
    param_str += f"PARSE_THESE_CHAINS_ONLY: {params.parse_these_chains_only}\n"
    param_str += f"USE_SIDE_CHAIN_CONTEXT: {params.use_side_chain_context}\n"
    param_str += f"USE_ATOM_CONTEXT: {params.use_atom_context}\n"
    param_str += f"BATCH_SIZE: {params.batch_size}\n"
    param_str += f"NUM_BATCHES: {params.num_batches}\n"
    param_str += f"SAVE_STATS: {params.save_stats}\n"
    param_str += f"VERBOSE: {params.verbose}\n"
    with open(os.path.join(params.output_dir, "params.txt"), "w") as f:
        f.write(param_str)


def log_pdb_file_info(pdbs: Iterable[str]) -> None:
    num_files = len(pdbs)
    plural = "s" if num_files > 1 else ""
    logger.info("")
    logger.info("INPUT FILES")
    logger.info("===========")
    logger.info(f"found {num_files} input PDB file{plural}:")
    if num_files < 6:
        for pdb in pdbs:
            logger.info(f"  {os.path.basename(pdb)}")
    else:
        for pdb in pdbs[:5]:
            logger.info(f"  {os.path.basename(pdb)}")
        logger.info(f"  ... and {num_files - 5} more")
    logger.info("")


def log_arguments(
    pdb_path: str,
    output_dir: str,
    model_type: str,
    model_checkpoint: str,
    seed: Union[int, str, Iterable[int]],
    temperature: Union[float, str, Iterable[float]],
    gpus: Union[int, str, Iterable[int]],
    bias_aa: Union[str, Dict[str, str]],
    bias_aa_per_residue: Union[str, Dict[str, str]],
    omit_aa: Union[str, Iterable[str]],
    omit_aa_per_residue: Union[str, Dict[str, str]],
    fixed_residues: Union[str, Dict[str, str]],
    redesigned_residues: Union[str, Dict[str, str]],
    chains_to_design: Union[str, Dict[str, str]],
    parse_these_chains_only: Union[str, Dict[str, str]],
    use_side_chain_context: bool,
    use_atom_context: bool,
    batch_size: int,
    num_batches: int,
    save_stats: bool,
    started_from_cli: bool,
    verbose: bool,
    debug: bool,
):
    logger.info("")
    logger.info("ARGUMENTS")
    logger.info("=========")
    logger.info(f"PDB_PATH: {str(pdb_path)}")
    logger.info(f"OUTPUT_DIR: {str(output_dir)}")
    logger.info(f"MODEL_TYPE: {str(model_type)}")
    logger.info(f"MODEL_CHECKPOINT: {str(model_checkpoint)}")
    logger.info(f"SEED: {str(seed)}")
    logger.info(f"TEMPERATURE: {str(temperature)}")
    logger.info(f"GPUS: {str(gpus)}")
    logger.info(f"BIAS_AA: {str(bias_aa)}")
    logger.info(f"BIAS_AA_PER_RESIDUE: {str(bias_aa_per_residue)}")
    logger.info(f"OMIT_AA: {str(omit_aa)}")
    logger.info(f"OMIT_AA_PER_RESIDUE: {str(omit_aa_per_residue)}")
    logger.info(f"FIXED_RESIDUES: {str(fixed_residues)}")
    logger.info(f"REDESIGNED_RESIDUES: {str(redesigned_residues)}")
    logger.info(f"CHAINS_TO_DESIGN: {str(chains_to_design)}")
    logger.info(f"PARSE_THESE_CHAINS_ONLY: {str(parse_these_chains_only)}")
    logger.info(f"USE_SIDE_CHAIN_CONTEXT: {str(use_side_chain_context)}")
    logger.info(f"USE_ATOM_CONTEXT: {str(use_atom_context)}")
    logger.info(f"BATCH_SIZE: {str(batch_size)}")
    logger.info(f"NUM_BATCHES: {str(num_batches)}")
    logger.info(f"SAVE_STATS: {str(save_stats)}")
    logger.info(f"STARTED_FROM_CLI: {str(started_from_cli)}")
    logger.info(f"VERBOSE: {str(verbose)}")
    logger.info(f"DEBUG: {str(debug)}")
    logger.info("")


def log_splash():
    splash = """
   __   _                   ____  ______  _  ___  __
  / /  (_)__ ____ ____  ___/ /  |/  / _ \/ |/ / |/ /
 / /__/ / _ `/ _ `/ _ \/ _  / /|_/ / ___/    /    / 
/____/_/\_, /\_,_/_//_/\_,_/_/  /_/_/  /_/|_/_/|_/  
       /___/                                        
"""
    logger.info(splash)


# def gpu_worker(
#     cmd: str,
#     gpu_queue: queue.Queue,
# ) -> None:
#     gpu_id = gpu_queue.get()
#     try:
#         cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} {cmd}"
#         sp.run(cmd, shell=True)
#     finally:
#         gpu_queue.put(gpu_id)


def _get_ligandmpnn_cmd(
    params: LigandMPNNParameters,
) -> str:
    run_path = os.path.join(LIGAND_MPNN_DIR, "run.py")
    cmd = f"python {run_path}"
    cmd += f' --pdb_path "{params.pdb_path}"'
    cmd += f" --model_type {params.model_type}"
    cmd += f' --out_folder "{params.output_dir}"'
    cmd += f" --seed {params.seed}"
    cmd += f" --temperature {params.temperature}"
    cmd += f" --batch_size {params.batch_size}"
    cmd += f" --number_of_batches {params.num_batches}"
    # model checkpoint
    if params.model_type == "ligand_mpnn":
        cmd += f' --checkpoint_ligand_mpnn "{params.model_checkpoint}"'
    elif params.model_type == "protein_mpnn":
        cmd += f' --checkpoint_protein_mpnn "{params.model_checkpoint}"'
    # fixed/redesigned residues
    if params.fixed_residues is not None:
        cmd += f' --fixed_residues "{params.fixed_residues}"'
    if params.redesigned_residues is not None:
        cmd += f' --redesigned_residues "{params.redesigned_residues}"'
    # chains
    if params.chains_to_design is not None:
        cmd += f' --chains_to_design "{params.chains_to_design}"'
    if params.parse_these_chains_only is not None:
        cmd += f' --parse_these_chains_only "{params.parse_these_chains_only}"'
    # all-atom
    cmd += f" --ligand_mpnn_use_side_chain_context {int(params.use_side_chain_context)}"
    cmd += f" --ligand_mpnn_use_atom_context {int(params.use_atom_context)}"
    # misc
    cmd += f" --save_stats {int(params.save_stats)}"
    cmd += " --verbose 0"

    return cmd


def _process_chain_or_residue_data(
    data: Union[str, Dict[str, str], None],
    pdbs: Iterable,
    sep: str = " ",
) -> Dict[str, str]:
    """
    Process chain or residue input.

    Parameters
    ----------
    data : Union[str, Dict[str, str], None]
        Chain or residue data. Can be either:
            - a file path to a JSON file, with PDB file paths as keys and space- or comma-separated chain or residue IDs as values
            - a file path to a plain text file, with space- or comma-separated chain or residue IDs that will be applied to all PDBs
            - a string of space- or comma-separated chain or residue IDs
            - a list or tuple of chain or residue IDs

    pdbs : Iterable
        List of PDB names.

    sep : str, default=" "
        Separator to be used when compiling a string representation of chain or residue data from an Iterable.

    Returns
    -------
    Dict[str, str]
        Residue dictionary.

    """
    magika = Magika()
    if data is not None:
        if isinstance(data, str):
            if os.path.isfile(data):
                filetype = magika.identify_path(Path(data)).output.ct_label
                with open(data, "r") as f:
                    # JSON file, which is a mapping of PDB file paths to residue IDs
                    if filetype == "json":
                        residues_dict = json.load(f)
                    # plain text file, which is just residue IDs that will be applied to all PDBs
                    else:
                        residues_dict = {pdb: f.read().strip() for pdb in pdbs}
            else:
                residues_dict = {pdb: data for pdb in pdbs}
        elif isinstance(data, (list, tuple)):
            data = sep.join([str(r) for r in data])
            residues_dict = {pdb: data for pdb in pdbs}
        else:
            raise ValueError(f"Invalid chain or residue data: {data}")
    else:
        residues_dict = {}
    return residues_dict


def _process_per_residue_data(
    data: Union[str, Dict[str, str], None],
    pdbs: Iterable,
) -> Tuple[Optional[str]]:
    """
    Process per-residue data, supplied as either a JSON file or a dictionary.

    Parameters
    ----------
    data : Union[str, Dict[str, str], None]
        Per-residue data.

    pdbs : Iterable
        List of PDB names.

    Returns
    -------
    Dict[str, str]
        Per-residue data dictionary.

    """
    if data is not None:
        if isinstance(data, str):
            if os.path.isfile(data):
                with open(data, "r") as f:
                    res_data = json.load(f)
            else:
                raise ValueError(f"Supplied per-residue data ({data}) does not exist")
        else:
            res_data = data
        if any(os.path.isfile(k) for k in res_data.keys()):
            return res_data
        else:
            return {pdb: res_data for pdb in pdbs}
    return {}


def _get_model_checkpoint(model_type: str, model_checkpoint: str) -> str:
    """
    Get the path to the model checkpoint.

    Parameters
    ----------
    model_type : str
        Type of model.

    model_checkpoint : str
        Model checkpoint.

    Returns
    -------
    str
        Path to the model checkpoint.
    """
    model_type = model_type.lower().replace("_", "")
    model_params_dir = os.path.join(LIGAND_MPNN_DIR, "model_params")
    model_checkpoint_path = os.path.join(
        model_params_dir, f"{model_type}_{model_checkpoint}.pt"
    )
    return model_checkpoint_path
