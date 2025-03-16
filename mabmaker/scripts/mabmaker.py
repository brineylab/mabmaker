# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Iterable, Optional, Union

import click

from ..tools.ligandmpnn import ligandmpnn as run_ligandmpnn
from ..tools.protenix import protenix as run_protenix


@click.group()
def cli():
    pass


@cli.command()
@click.argument(
    "pdb_path",
    type=str,
    required=True,
    # help="Path to the input PDB file or a directory of PDB files.",
)
@click.argument(
    "output_dir",
    type=str,
    required=True,
    # help="Path to the output directory. It will be created if it does not exist.",
)
@click.option(
    "-m",
    "--model_type",
    type=str,
    default="ligand_mpnn",
    help="Type of model to use. Options are 'ligand_mpnn' or 'protein_mpnn'.",
)
@click.option(
    "--model_checkpoint",
    type=str,
    default=None,
    help="Checkpoint of the model to use, excluding the model name and file extension. For example, if the full name of the weights file for the desired model checkpoint is 'ligandmpnn_v_32_10_25.pt', then `model_checkpoint` should be 'v_32_10_25'.",
)
@click.option(
    "-s",
    "--seed",
    # type=int,
    # default=42,
    help="Random seed(s) to use, for example '42' or '42,43'. If multiple seed values are provided, each PDB file will be processed with every combination of temperature and seed.",
)
@click.option(
    "--gpus",
    type=str,
    default=None,
    help="GPU(s) to use, for example '0' or '0,1'. If not provided, all available GPUs will be used.",
)
@click.option(
    "-t",
    "--temperature",
    # type=float,
    # default=0.1,
    help="Temperature(s) to use, for example '0.1' or '0.1,0.2'. If multiple temperature values are provided, each PDB file will be processed with every combination of temperature and seed.",
)
@click.option(
    "--bias_aa",
    type=str,
    default=None,
    help="Bias the generation of AAs, for example 'A:-1.024,P:2.34,C:-12.34'",
)
@click.option(
    "--bias_aa_per_residue",
    type=str,
    default=None,
    help="Path to a JSON file containing per-residue AA biases, for example {'A12': {'G': -0.3, 'C': -2.0, 'H': 0.8}, 'A13': {'G': -1.3}}. Alternatively, provide a JSON file containing per-pdb-file per-residue AA biases, for example {'/path/to/pdb': {'A12': {'G': -0.3, 'C': -2.0, 'H': 0.8}, 'A13': {'G': -1.3}}}.",
)
@click.option(
    "--omit_aa",
    type=str,
    default=None,
    help="Omit the generation of certain AAs, for example 'ACG'",
)
@click.option(
    "--omit_aa_per_residue",
    type=str,
    default=None,
    help="Path to a JSON file containing per-residue AA omissions, for example {'A12': 'APQ', 'A13': 'QST'}. Alternatively, provide a JSON file containing per-pdb-file per-residue AA omissions, for example {'/path/to/pdb': {'A12': 'QSPC', 'A13': 'AGE'}}.",
)
@click.option(
    "--fixed_residues",
    type=str,
    default=None,
    help="Provide fixed residues. Can be a string of space-separated residue IDs (for example 'A12 A13 A14 B2 B25'), the path to a text file containing space-separated residue IDs that will be aplied to all input PDBs, or a file path to a JSON file mapping PDB file paths to space-separated residue IDs.",
)
@click.option(
    "--redesigned_residues",
    type=str,
    default=None,
    help="Provide redesigned residues. Can be a string of space-separated residue IDs (for example 'A12 A13 A14 B2 B25'), the path to a text file containing space-separated residue IDs that will be aplied to all input PDBs, or a file path to a JSON file mapping PDB file paths to space-separated residue IDs.",
)
@click.option(
    "--chains_to_design",
    type=str,
    default=None,
    help="Provide chains to design. Can be a string of comma-separated chain IDs (for example 'A,B,C'), the path to a text file containing comma-separated chain IDs that will be aplied to all input PDBs, or a file path to a JSON file mapping PDB file paths to comma-separated chain IDs.",
)
@click.option(
    "--parse_these_chains_only",
    type=str,
    default=None,
    help="Provide chains to parse. Can be a string of comma-separated chain IDs (for example 'A,B,C'), the path to a text file containing comma-separated chain IDs that will be aplied to all input PDBs, or a file path to a JSON file mapping PDB file paths to comma-separated chain IDs.",
)
@click.option(
    "--use_side_chain_context/--no_use_side_chain_context",
    default=True,
    help="Use side chain context for generation.",
)
@click.option(
    "--use_atom_context/--no_use_atom_context",
    default=False,
    help="Use atom context for generation.",
)
@click.option(
    "--batch_size",
    type=int,
    default=32,
    help="Number of sequences to generate per pass.",
)
@click.option(
    "--num_batches",
    type=int,
    default=1,
    help="Number of times to design sequences using the chosen `batch size`.",
)
@click.option(
    "--save_stats/--no_save_stats",
    default=True,
    help="Whether to save the stats.",
)
@click.option(
    "--verbose/--quiet",
    default=True,
    help="Print verbose output",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Print debug output.",
)
def ligandmpnn(
    pdb_path: str,
    output_dir: str,
    model_type: str = "ligand_mpnn",
    model_checkpoint: Optional[str] = None,
    seed: Union[int, str] = 42,
    gpus: Optional[str] = None,
    temperature: Union[float, str] = 0.1,
    bias_aa: Optional[str] = None,
    bias_aa_per_residue: Optional[str] = None,
    omit_aa: Optional[str] = None,
    omit_aa_per_residue: Optional[str] = None,
    fixed_residues: Optional[str] = None,
    redesigned_residues: Optional[str] = None,
    chains_to_design: Optional[str] = None,
    parse_these_chains_only: Optional[str] = None,
    use_side_chain_context: bool = True,
    use_atom_context: bool = False,
    batch_size: int = 32,
    num_batches: int = 1,
    save_stats: bool = True,
    verbose: bool = True,
    debug: bool = False,
) -> None:
    run_ligandmpnn(
        pdb_path=pdb_path,
        output_dir=output_dir,
        model_type=model_type,
        model_checkpoint=model_checkpoint,
        seed=seed,
        gpus=gpus,
        temperature=temperature,
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
        verbose=verbose,
        debug=debug,
        started_from_cli=True,
    )


@cli.command()
@click.argument(
    "json_path",
    type=str,
    required=True,
    # help="Path to the input JSON file or a directory of JSON files.",
)
@click.argument(
    "output_path",
    type=str,
    required=True,
    # help="Path to the output directory. It will be created if it does not exist.",
)
@click.option(
    "--gpus",
    type=str,
    default=None,
    help="GPU(s) to use, for example '0' or '0,1'. If not provided, all available GPUs will be used.",
)
@click.option(
    "--use_msa_server/--use_esm_embeddings",
    default=True,
    help="Use the MSA server to get the MSA.",
)
def protenix(
    json_path: str,
    output_path: str,
    gpus: int | Iterable[int] | None = None,
    use_msa_server: bool = True,
) -> None:
    run_protenix(
        json_path=json_path,
        output_path=output_path,
        gpus=gpus,
        use_msa_server=use_msa_server,
    )
