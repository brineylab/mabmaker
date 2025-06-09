# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

import os
from typing import Iterable, Optional, Union

import click

from ..tools.boltz import boltz as run_boltz
from ..tools.chai import chai as run_chai
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
    """
    Structure-based sequence design with LigandMPNN.
    """
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
    "--use_msa_server/--no_msa_server",
    default=True,
    help="Use the MSA server to get the MSA.",
)
@click.option(
    "--msa_server_url",
    type=str,
    default="https://api.colabfold.com",
    help="The URL of the MSA server.",
)
@click.option(
    "--use_msa_cache/--no_use_msa_cache",
    default=True,
    help="Whether to use the MSA cache.",
)
@click.option(
    "--msa_cache_dir",
    type=str,
    default="~/.mabmaker/msa_cache",
    help="The path to the MSA cache directory.",
)
@click.option(
    "--num_trunk_recycles",
    type=int,
    default=4,
    help="The number of trunk recycles to perform.",
)
@click.option(
    "--num_diffusion_timesteps",
    type=int,
    default=200,
    help="The number of diffusion timesteps to use.",
)
@click.option(
    "--num_diffusion_samples",
    type=int,
    default=5,
    help="The number of diffusion samples to generate.",
)
@click.option(
    "--compress_output/--no_compress_output",
    default=False,
    help="Whether to compress the output directory.",
)
def protenix(
    json_path: str,
    output_path: str,
    gpus: int | Iterable[int] | None = None,
    use_msa_server: bool = True,
    msa_server_url: str = "https://api.colabfold.com",
    use_msa_cache: bool = True,
    msa_cache_dir: str = "~/.mabmaker/msa_cache",
    num_trunk_recycles: int = 4,
    num_diffusion_timesteps: int = 200,
    num_diffusion_samples: int = 5,
    compress_output: bool = False,
) -> None:
    """
    Structure prediction with Protenix.
    """
    run_protenix(
        json_path=json_path,
        output_path=output_path,
        gpus=gpus,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        use_msa_cache=use_msa_cache,
        msa_cache_dir=msa_cache_dir,
        num_trunk_recycles=num_trunk_recycles,
        num_diffusion_timesteps=num_diffusion_timesteps,
        num_diffusion_samples=num_diffusion_samples,
        compress_output=compress_output,
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
    "--use_msa_server/--no_msa_server",
    default=True,
    help="Use the MSA server to get the MSA.",
)
@click.option(
    "--msa_server_url",
    type=str,
    default="https://api.colabfold.com",
    help="The URL of the MSA server.",
)
@click.option(
    "--use_msa_cache/--no_msa_cache",
    default=True,
    help="Whether to use the MSA cache.",
)
@click.option(
    "--msa_cache_dir",
    type=str,
    default="~/.mabmaker/msa_cache",
    help="The path to the MSA cache directory.",
)
@click.option(
    "--recycle_msa_subsample",
    type=int,
    default=0,
    help="Whether to subsample the MSA for each trunk recycle. If 0, no subsampling will be performed. If >0, the MSA will be subsampled.",
)
@click.option(
    "--use_templates_server",
    is_flag=True,
    default=False,
    help="Whether to use the templates server.",
)
@click.option(
    "--template_hits_path",
    type=str,
    default=None,
    help="The path to the template hits file.",
)
@click.option(
    "--msa_directory",
    type=str,
    default=None,
    help="The path to the directory containing the MSAs.",
)
@click.option(
    "--num_trunk_recycles",
    type=int,
    default=3,
    help="The number of trunk recycles to perform.",
)
@click.option(
    "--num_trunk_samples",
    type=int,
    default=1,
    help="The number of trunk samples to generate.",
)
@click.option(
    "--num_diffusion_timesteps",
    type=int,
    default=200,
    help="The number of diffusion timesteps to use.",
)
@click.option(
    "--num_diffusion_samples",
    type=int,
    default=5,
    help="The number of diffusion samples to generate.",
)
@click.option(
    "--low_memory",
    is_flag=True,
    default=False,
    help="Whether to use low memory mode.",
)
@click.option(
    "--compress_output/--no_compress_output",
    default=False,
    help="Whether to compress the output directory.",
)
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
    template_hits_path: Optional[str] = None,
    msa_directory: Optional[str] = None,
    num_trunk_recycles: int = 3,
    num_trunk_samples: int = 1,
    num_diffusion_timesteps: int = 200,
    num_diffusion_samples: int = 5,
    low_memory: bool = False,
    compress_output: bool = False,
) -> None:
    """
    Structure prediction with Chai-1.
    """
    run_chai(
        json_path=os.path.expanduser(json_path),
        output_path=os.path.expanduser(output_path),
        gpus=gpus,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        use_msa_cache=use_msa_cache,
        msa_cache_dir=os.path.expanduser(msa_cache_dir),
        recycle_msa_subsample=recycle_msa_subsample,
        use_templates_server=use_templates_server,
        template_hits_path=template_hits_path,
        msa_directory=msa_directory,
        num_trunk_recycles=num_trunk_recycles,
        num_trunk_samples=num_trunk_samples,
        num_diffusion_timesteps=num_diffusion_timesteps,
        num_diffusion_samples=num_diffusion_samples,
        low_memory=low_memory,
        compress_output=compress_output,
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
    "--use_msa_server/--no_msa_server",
    default=True,
    help="Whether to use the MSA server.",
)
@click.option(
    "--msa_server_url",
    type=str,
    default="https://api.colabfold.com",
    help="The URL of the MSA server.",
)
@click.option(
    "--use_msa_cache/--no_msa_cache",
    default=True,
    help="Whether to use the MSA cache.",
)
@click.option(
    "--msa_cache_dir",
    type=str,
    default="~/.mabmaker/msa_cache",
    help="The path to the MSA cache directory.",
)
@click.option(
    "--recycling_steps",
    type=int,
    default=3,
    help="The number of recycling steps.",
)
@click.option(
    "--sampling_steps",
    type=int,
    default=200,
    help="The number of sampling steps.",
)
@click.option(
    "--diffusion_samples",
    type=int,
    default=5,
    help="The number of diffusion samples.",
)
@click.option(
    "--step_scale",
    type=float,
    default=1.638,
    help="The step scale. The step size is related to the temperature at which the diffusion process samples the distribution. The lower the step_scale, the higher the diversity among samples (recommended between 1 and 2).",
)
@click.option(
    "--output_format",
    type=str,
    default="mmcif",
    help="The output format. Options are 'mmcif' and 'pdb'.",
)
@click.option(
    "--override",
    is_flag=True,
    default=False,
    help="Whether to override existing output files if found.",
)
@click.option(
    "--msa_pairing_strategy",
    type=str,
    default="greedy",
    help="The MSA pairing strategy. Used only if use_msa_server is True. Options are 'greedy' and 'complete'.",
)
@click.option(
    "--write_full_pae/--no_write_full_pae",
    default=True,
    help="Whether to write the full predicted aligned error (PAE) matrix as a file.",
)
@click.option(
    "--write_full_pde/--no_write_full_pde",
    default=True,
    help="Whether to write the full predicted docking error (PDE) matrix as a file.",
)
@click.option(
    "--cache",
    type=str,
    default="~/.boltz",
    help="The path to the cache directory, which contains model weights and other resources.",
)
@click.option(
    "--compress_output/--no_compress_output",
    default=False,
    help="Whether to compress the output directory.",
)
def boltz(
    json_path: str,
    output_path: str,
    model: str = "boltz2",
    gpus: int | Iterable[int] | None = None,
    use_msa_server: bool = True,
    msa_server_url: str = "https://api.colabfold.com",
    use_msa_cache: bool = True,
    msa_cache_dir: str = "~/.mabmaker/msa_cache",
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
    compress_output: bool = False,
    # Boltz-2 only
    affinity_mw_correction: bool = False,
    sampling_steps_affinity: int = 200,
    diffusion_samples_affinity: int = 5,
    affinity_checkpoint: str | None = None,
) -> None:
    """
    Structure prediction with Boltz-1.
    """
    run_boltz(
        json_path=os.path.expanduser(json_path),
        output_path=os.path.expanduser(output_path),
        model=model,
        gpus=gpus,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        use_msa_cache=use_msa_cache,
        msa_cache_dir=os.path.expanduser(msa_cache_dir),
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
        compress_output=compress_output,
        # Boltz-2 only
        affinity_mw_correction=affinity_mw_correction,
        sampling_steps_affinity=sampling_steps_affinity,
        diffusion_samples_affinity=diffusion_samples_affinity,
        affinity_checkpoint=affinity_checkpoint,
    )
