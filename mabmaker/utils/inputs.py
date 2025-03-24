# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import json
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable

import abutils
import yaml
from magika import Magika
from natsort import natsorted

from .mixins import BoltzFormattingMixin, ChaiFormattingMixin, ProtenixFormattingMixin


def setup_structure_prediction_run(
    json_path: str,
    output_path: str,
) -> Iterable:
    """
    Setup structure prediction runs from a JSON file.
    """
    # setup output directory structure
    os.makedirs(output_path, exist_ok=True)

    # read input file(s)
    magika = Magika()
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON path not found: {json_path}")
    if os.path.isdir(json_path):
        json_paths = [
            f
            for f in abutils.list_files(json_path)
            if magika.identify_path(Path(f)).output.ct_label == "json"
        ]
    else:
        if not magika.identify_path(Path(json_path)).output.ct_label == "json":
            raise ValueError(
                f"supplied JSON file does not appear to be a JSON file: {json_path}"
            )
        json_paths = [json_path]
    data = []
    for json_path in json_paths:
        with open(json_path, "r") as f:
            data.extend(json.load(f))
    return [StructurePredictionRun(d) for d in data]


# =============================================
#
#                 MODELING RUN
#
# =============================================

# TODO: rewrite to accept either alphafoldserver or alphafold3 input dialects
# differences betwee the two can be found at: https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md


class StructurePredictionRun(
    ChaiFormattingMixin, BoltzFormattingMixin, ProtenixFormattingMixin
):
    """
    A class for parsing and formatting input data for a structure prediction run.

    Parameters
    ----------
    params : dict
        The input parameters. Must be a dictionary with a schema that matches that of the
        `AlphaFold3 input JSON file`_.

    .. warning::
        Only linear (unbranched) glycans are currently supported. Branched glycans will be
        supported in the future.

    .. _AlphaFold3 input JSON file: https://github.com/google-deepmind/alphafold/tree/main/server

    """

    def __init__(self, params: dict):
        self.params = params
        self.name = params.get("name", None)
        self.seeds = list(set(params.get("modelSeeds", ["42"])))
        self.dialect = params.get("dialect", "alphafoldserver")
        self.version = params.get("version", 1)
        self.entities = self.parse_entities()
        self.protein_chains = [
            entity for entity in self.entities if entity.kind == "proteinChain"
        ]
        self.dna_sequences = [
            entity for entity in self.entities if entity.kind == "dnaSequence"
        ]
        self.rna_sequences = [
            entity for entity in self.entities if entity.kind == "rnaSequence"
        ]
        self.ligands = [entity for entity in self.entities if entity.kind == "ligand"]
        self.ions = [entity for entity in self.entities if entity.kind == "ion"]

    @property
    def glycans(self):
        glycans = []
        for chain in self.protein_chains:
            glycans.extend(chain.glycans)
        return glycans

    @property
    def num_protein_chains(self):
        return sum(chain.count for chain in self.protein_chains)

    @property
    def num_glycans(self):
        num = 0
        for chain in self.protein_chains:
            chain_num = len(chain.glycans)
            num += chain_num * chain.count
        return num

    @property
    def num_dna_sequences(self):
        return sum(seq.count for seq in self.dna_sequences)

    @property
    def num_rna_sequences(self):
        return sum(seq.count for seq in self.rna_sequences)

    @property
    def num_ligands(self):
        return sum(ligand.count for ligand in self.ligands)

    @property
    def num_ions(self):
        return sum(ion.count for ion in self.ions)

    def num_entities(
        self, kind: str, include_copies: bool = False, separate_ccds: bool = False
    ):
        """
        Get the number of entities of a given kind.

        Parameters
        ----------
        kind : str
            The kind of entity to count.

        include_copies : bool, optional
            Whether to include the number of copies of each entity. Default is ``False``,
            which counts each entity once regardless of how many copies there are.

        separate_ccds : bool, optional
            Whether to count each CCD as a separate entity. Only relevant for glycans.
            Default is ``False``, which counts multi-CCD glycans as a single entity.

        Returns
        -------
        int
            The number of entities of the given kind.

        """
        # glycans are a special case, since they're attached to proteinChains
        if kind.lower() == "glycan":
            num_entities = 0
            for chain in self.protein_chains:
                if separate_ccds:
                    num_glycans = sum(
                        len(glycan.ccd_list()) for glycan in chain.glycans
                    )
                else:
                    num_glycans = len(chain.glycans)
                if include_copies:
                    num_glycans *= chain.count
                num_entities += num_glycans
        else:
            entities = [e for e in self.entities if e.kind == kind]
            if include_copies:
                num_entities = sum(entity.count for entity in entities)
            else:
                num_entities = len(entities)
        return num_entities

    def parse_entities(self):
        entities = []
        sequences = self.params.get("sequences", [])
        for s in sequences:
            kind = list(s.keys())[0]
            sequence = s[kind]
            if kind == "proteinChain":
                entities.append(ProteinChain(sequence))
            elif kind == "dnaSequence":
                entities.append(NucleicAcidSequence(sequence, kind=kind))
            elif kind == "rnaSequence":
                entities.append(NucleicAcidSequence(sequence, kind=kind))
            elif kind == "ligand":
                entities.append(Ligand(**sequence))
            elif kind == "ion":
                entities.append(Ion(**sequence))
            else:
                raise ValueError(f"unexpected sequence type: {sequence}")
        return entities


# =============================================
#
#                  MODIFICATIONS
#
# =============================================


@dataclass
class Modification:
    """
    A class for parsing and formatting modifications. Used for both protein and nucleic acid
    modifications.

    Parameters
    ----------
    modification_type : str | None, optional
        The type of modification.

    position : int | None, optional
        The position of the modification.
    """

    modification_type: str | None = None
    position: int | None = None


@dataclass
class Glycan:
    """
    A class for parsing and formatting glycans.

    . warning::
        Only linear (unbranched) glycans are currently supported. Branched glycans may be
        supported in the future.

    Parameters
    ----------
    residues : str | None, optional
        The residues of the glycan.

    position : int | None, optional
        The position of the glycan.
    """

    residues: str | None = None
    position: int | None = None
    chain: str | None = None
    protein_chain: str | None = None

    @property
    def chai_formatted(self):
        if self.residues is not None:
            return self.residues.replace("NAG(", "NAG(4-1 ").replace("MAN(", "MAN(6-1 ")

    @property
    def protenix_formatted(self):
        if self.residues is not None:
            return "CCD_" + self.residues.rstrip(")").replace("(", "_")

    def ccd_list(self):
        if self.residues is not None:
            return self.residues.rstrip(")").split("(")


class ModificationMixin:
    def parse_modifications(self):
        modifications = []
        for modification in self.raw.get("modifications", []):
            m = {}
            # protein modifications
            if "ptmType" in modification:
                m["modification_type"] = modification.get("ptmType", None)
                m["position"] = modification.get("ptmPosition", None)
            # nucleic acid modifications
            elif "modificationType" in modification:
                m["modification_type"] = modification.get("modificationType", None)
                m["position"] = modification.get("basePosition", None)
            modifications.append(Modification(**m))
        return modifications


class GlycanMixin:
    def parse_glycans(self):
        glycans = []
        for glycan in self.raw.get("glycans", []):
            glycans.append(Glycan(**glycan))
        return glycans


# =============================================
#
#                   ENTITIES
#
# =============================================


class Entity:
    def __init__(self, sequence: dict):
        self.raw = sequence
        self.sequence = sequence.get("sequence", None)
        self.count = sequence.get("count", 1)
        self.chain = None


class ProteinChain(Entity, ModificationMixin, GlycanMixin):
    def __init__(self, sequence: dict):
        super().__init__(sequence)
        self.kind = "proteinChain"
        self.msa = sequence.get("msa", None)
        self.glycans = self.parse_glycans()
        self.modifications = self.parse_modifications()
        self.use_structure_template = sequence.get("useStructureTemplate", True)
        self.max_template_date = sequence.get("maxTemplateDate", None)


class NucleicAcidSequence(Entity, ModificationMixin):
    def __init__(self, sequence: dict, kind: str):
        super().__init__(sequence)
        self.kind = kind
        self.modifications = self.parse_modifications()


@dataclass
class Ligand:
    """
    A class for parsing and formatting ligands.

    Parameters
    ----------
    ligand : str | None, optional
        The ligand.

    count : int | None, optional
        The count of the ligand.
    """

    kind = "ligand"
    ligand: str | None = None
    count: int | None = None


@dataclass
class Ion:
    """
    A class for parsing and formatting ions.

    Parameters
    ----------
    ion : str | None, optional
        The ion.

    count : int | None, optional
        The count of the ion.
    """

    kind = "ion"
    ion: str | None = None
    count: int | None = None


class StructurePredictionInput:
    def __init__(
        self,
        input_path: str,
        output_path: str,
        seed: int | Iterable[int] = 42,
        use_msa_server: bool = True,
        msa_server_url: str = "https://api.colabfold.com",
        msa_directory: str | None = None,
        num_recycles: int = 3,
        num_diffusion_timesteps: int = 200,
        num_diffusion_samples: int = 5,
        device: str | None = None,
        low_memory: bool = False,
        verbose: bool = False,
        debug: bool = False,
    ):
        """
        Parameters
        ----------
        input_path : str
            Path to a JSON or YAML file containing the structure prediction parameters.
            The data for each run should be in the format of the `AlphaFold3 input JSON file`_::

            ``` json
            {
                "name": "Job name goes here",
                "modelSeeds": [1, 2],  # At least one seed required.
                "sequences": [
                    {"protein": {...}},
                    {"rna": {...}},
                    {"dna": {...}},
                    {"ligand": {...}}
                ],
                "bondedAtomPairs": [...],  # Optional
                "userCCD": "...",  # Optional
                "options": {"opt": "...", "opt2": "..."}  # Optional
            }
            ```

            For inputs containing multiple runs, the JSON file should contain a list of
            dictionaries, each with the same structure. Note that the `name` field is
            used to name the output directory, so it should be unique for each run.

            The `options` field is used to pass additional options to the structure
            prediction tool that should be separately applied to each run, such as
            the number of recycles or diffusion timesteps.

            To use a YAML file, the file should contain a list of dictionaries, the input
            data should follow the same general structure::

            ``` yaml
            - name: "Job name goes here"
              modelSeeds:  # At least one seed required.
                - 1
                - 2
              sequences:
                - protein:
                    ...
                - rna:
                    ...
                - dna:
                    ...
                - ligand:
                    ...
              bondedAtomPairs:  # Optional
                - ...
                - ...
              userCCD:  # Optional
                ...
              options:  # Optional
                opt: ...
                opt2: ...
            ```

        output_path : str
            Path to the output directory.

        seed : int | Iterable[int], optional
            Random seed.

        use_msa_server : bool, optional
            Whether to use the MSA server.

        msa_server_url : str, optional
            URL of the MSA server.

        msa_directory : str | None, optional
            Path to the directory containing the MSA files.

        num_recycles : int, optional
            Number of recycles.

        num_diffusion_timesteps : int, optional
            Number of diffusion timesteps.

        num_diffusion_samples : int, optional
            Number of diffusion samples.

        device : str | None, optional
            Device to use.

        low_memory : bool, optional
            Whether to use low memory mode.

        verbose : bool, optional
            Whether to print verbose output.

        debug : bool, optional
            Whether to print debug output.


        .. _AlphaFold3 input JSON file: https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md

        """
        self.input_path = input_path
        self.output_path = output_path
        self.seed = seed
        self.use_msa_server = use_msa_server
        self.msa_server_url = msa_server_url
        self.msa_directory = msa_directory
        self.num_recycles = num_recycles
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.num_diffusion_samples = num_diffusion_samples
        self.device = device
        self.low_memory = low_memory
        self.verbose = verbose
        self.debug = debug

        # input parsing
        magika = Magika()
        if magika.identify_path(Path(input_path)).output.ct_label == "yaml":
            with open(input_path, "r") as f:
                self.input_data = yaml.load_all(f)
        elif magika.identify_path(Path(input_path)).output.ct_label == "json":
            with open(input_path, "r") as f:
                self.input_data = json.load(f)
        else:
            raise ValueError(f"Input file must be a YAML or JSON file: {input_path}")

    def __iter__(self):
        return iter(self.jobs)

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.jobs[index]
        elif isinstance(index, slice):
            return self.jobs[index]
        elif isinstance(index, str) and index in self.job_names():
            return next(job for job in self.jobs if job["name"] == index)
        else:
            raise ValueError(f"Invalid index type: {type(index)}")

    @cached_property
    def jobs(self):
        return natsorted(self.input_data, key=lambda x: x["name"])

    def job_names(self):
        return [job["name"] for job in self.jobs]


# The actual input should probably mimic that of AlphaFold3,
# which can be found here: https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md

# Briefly, it's a JSON file with the following structure:

# {
#   "name": "Job name goes here",
#   "modelSeeds": [1, 2],  # At least one seed required.
#   "sequences": [
#     {"protein": {...}},
#     {"rna": {...}},
#     {"dna": {...}},
#     {"ligand": {...}}
#   ],
#   "bondedAtomPairs": [...],  # Optional
#   "userCCD": "...",  # Optional
#   "dialect": "alphafold3",  # Required
#   "version": 2  # Required
# }


class StructurePredictionJob:
    def __init__(self, job_data: dict):
        self.job_data = job_data
