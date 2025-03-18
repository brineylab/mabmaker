# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

import json
import os
from collections import deque
from typing import Tuple

import abutils
import yaml

from .chains import get_chain_name_generator

# =============================================
#
#           MODEL-SPECIFIC FORMATTING
#
# =============================================


class BoltzFormattingMixin:
    def build_boltz_input(self, output_path: str | None = None) -> str:
        sequences = []
        ligands = []
        constraints = []

        def ccd_bond_atom_lookup(glycan_ccd: str):
            ccd_dict = {
                "NAG": "O4",
                "MAN": "O6",
            }
            return ccd_dict.get(glycan_ccd, None)

        # get chain names for all entity types (including copies)
        chain_gen = get_chain_name_generator("boltz")
        protein_chain_names = deque(
            [
                next(chain_gen)
                for _ in range(
                    self.num_entities(kind="proteinChain", include_copies=True)
                )
            ]
        )
        dna_chain_names = deque(
            [
                next(chain_gen)
                for _ in range(
                    self.num_entities(kind="dnaSequence", include_copies=True)
                )
            ]
        )
        rna_chain_names = deque(
            [
                next(chain_gen)
                for _ in range(
                    self.num_entities(kind="rnaSequence", include_copies=True)
                )
            ]
        )
        glycan_chain_names = deque(
            [
                next(chain_gen)
                for _ in range(
                    self.num_entities(
                        kind="glycan", include_copies=True, separate_ccds=True
                    )
                )
            ]
        )
        ligand_chain_names = deque(
            [
                next(chain_gen)
                for _ in range(self.num_entities(kind="ligand", include_copies=True))
            ]
        )
        ion_chain_names = deque(
            [
                next(chain_gen)
                for _ in range(self.num_entities(kind="ion", include_copies=True))
            ]
        )

        # protein chains
        for chain in self.protein_chains:
            protein_ids = [protein_chain_names.popleft() for _ in range(chain.count)]
            sequence = {
                "protein": {
                    "id": protein_ids,
                    "sequence": chain.sequence,
                }
            }
            # only add modifications or MSA if they exist
            if chain.modifications:
                sequence["protein"]["modifications"] = [
                    {
                        "position": m.position,
                        "ccd": m.modification_type,
                    }
                    for m in chain.modifications
                ]
            if chain.msa is not None:
                sequence["protein"]["msa"] = chain.msa
            sequences.append(sequence)

            # add glycans (if present)
            for glycan in chain.glycans:
                # boltz requires a separate ligand entry (and bond) for each CCD in the glycan
                glycan_ccd_list = glycan.ccd_list()
                for ccd_idx, glycan_ccd in enumerate(glycan_ccd_list):
                    glycan_ccd_ids = [
                        glycan_chain_names.popleft() for _ in range(chain.count)
                    ]
                    ligands.append(
                        {
                            "ligand": {
                                "id": glycan_ccd_ids,
                                "ccd": glycan_ccd,
                            }
                        }
                    )
                    # add covalent bonds between glycan and protein
                    for protein_copy_id, glycan_ccd_copy_id in zip(
                        protein_ids, glycan_ccd_ids
                    ):
                        if ccd_idx == 0:
                            prev_chain_id = protein_copy_id
                            prev_atom_name = "ND2"
                            prev_atom_position = glycan.position
                        else:
                            prev_chain_id = glycan_ccd_ids[ccd_idx - 1]
                            prev_atom_name = ccd_bond_atom_lookup(
                                glycan_ccd_list[ccd_idx - 1]
                            )
                            prev_atom_position = 1
                        constraints.append(
                            {
                                "bond": {
                                    "atom1": [
                                        prev_chain_id,
                                        prev_atom_position,
                                        prev_atom_name,
                                    ],
                                    "atom2": [glycan_ccd_copy_id, 1, "C1"],
                                }
                            }
                        )
        # DNA sequences
        for seq in self.dna_sequences:
            dna_ids = [dna_chain_names.popleft() for _ in range(seq.count)]
            sequence = {
                "dna": {
                    "id": dna_ids,
                    "sequence": seq.sequence,
                }
            }
            # only add modifications if they exist
            if seq.modifications:
                sequence["dna"]["modifications"] = [
                    {
                        "position": m.position,
                        "ccd": m.modification_type,
                    }
                    for m in seq.modifications
                ]
            sequences.append(sequence)

        # RNA sequences
        for seq in self.rna_sequences:
            rna_ids = [rna_chain_names.popleft() for _ in range(seq.count)]
            sequence = {
                "rna": {
                    "id": rna_ids,
                    "sequence": seq.sequence,
                }
            }
            # only add modifications if they exist
            if seq.modifications:
                sequence["rna"]["modifications"] = [
                    {
                        "position": m.position,
                        "ccd": m.modification_type,
                    }
                    for m in seq.modifications
                ]
            sequences.append(sequence)

        # ligands
        for ligand in self.ligands:
            ligand_ids = [ligand_chain_names.popleft() for _ in range(ligand.count)]
            ligands.append(
                {
                    "ligand": {
                        "id": ligand_ids,
                        "ccd": ligand.ligand,
                    }
                }
            )

        # ions
        for ion in self.ions:
            ion_ids = [ion_chain_names.popleft() for _ in range(ion.count)]
            ligands.append(
                {
                    "ligand": {
                        "id": ion_ids,
                        "ccd": ion.ion,
                    }
                }
            )

        # pull all the data together
        yaml_data = {
            "version": 1,
            "sequences": sequences + ligands,
            "constraints": constraints,
        }

        if output_path is not None:
            if os.path.isdir(output_path):
                output_path = os.path.join(output_path, f"{self.name}.yaml")
            abutils.io.make_dir(os.path.dirname(output_path))
            with open(output_path, "w") as f:
                yaml.dump(yaml_data, f)
            return output_path
        else:
            return yaml.dump(yaml_data)


class ChaiFormattingMixin:
    def build_chai_input(self, output_path: str | None = None) -> Tuple[str, str]:
        fastas = []
        constraints_header = [
            "chainA,res_idxA,chainB,res_idxB,connection_type,confidence,min_distance_angstrom,max_distance_angstrom,comment,restraint_id"
        ]
        constraints = []

        # get chain names for all entity types (including copies)
        chain_gen = get_chain_name_generator("chai")
        protein_chain_names = deque(
            [
                next(chain_gen)
                for _ in range(
                    self.num_entities(kind="proteinChain", include_copies=True)
                )
            ]
        )
        dna_chain_names = deque(
            [
                next(chain_gen)
                for _ in range(
                    self.num_entities(kind="dnaSequence", include_copies=True)
                )
            ]
        )
        rna_chain_names = deque(
            [
                next(chain_gen)
                for _ in range(
                    self.num_entities(kind="rnaSequence", include_copies=True)
                )
            ]
        )
        glycan_chain_names = deque(
            [
                next(chain_gen)
                for _ in range(self.num_entities(kind="glycan", include_copies=True))
            ]
        )
        ligand_chain_names = deque(
            [
                next(chain_gen)
                for _ in range(self.num_entities(kind="ligand", include_copies=True))
            ]
        )
        ion_chain_names = deque(
            [
                next(chain_gen)
                for _ in range(self.num_entities(kind="ion", include_copies=True))
            ]
        )

        # protein chains (FASTA only, protein-glycan bond constraints will be added later)
        for chain_idx, chain in enumerate(self.protein_chains):
            for copy_idx in range(chain.count):
                fasta = f">protein|chain{chain_idx}_copy{copy_idx+1}\n{chain.sequence}"
                fastas.append(fasta)

        # DNA sequences
        for seq_idx, seq in enumerate(self.dna_sequences):
            for copy_idx in range(seq.count):
                fasta = f">dna|sequence{seq_idx}_copy{copy_idx+1}\n{seq.sequence}"
                fastas.append(fasta)

        # RNA sequences
        for seq_idx, seq in enumerate(self.rna_sequences):
            for copy_idx in range(seq.count):
                fasta = f">rna|sequence{seq_idx}_copy{copy_idx+1}\n{seq.sequence}"
                fastas.append(fasta)

        # glycans
        bond_counter = 1
        for chain_idx, chain in enumerate(self.protein_chains):
            for copy_idx in range(chain.count):
                protein_chain_name = protein_chain_names.popleft()
                for glycan_idx, glycan in enumerate(chain.glycans):
                    # glycan fasta
                    fasta = f">glycan|chain{chain_idx}_glycan{glycan_idx}_copy{copy_idx+1}\n{glycan.chai_formatted}"
                    fastas.append(fasta)
                    # protein-glycan bond constraints
                    glycan_chain_name = glycan_chain_names.popleft()
                    constraints.append(
                        f"{protein_chain_name},N{glycan.position}@N,{glycan_chain_name},@C1,covalent,1.0,0.0,0.0,protein-glycan,bond{bond_counter}"
                    )
                    bond_counter += 1

        # ligands
        for ligand_idx, ligand in enumerate(self.ligands):
            for copy_idx in range(ligand.count):
                fasta = f">ligand|chain{ligand_idx}_copy{copy_idx+1}\n{ligand.ligand}"
                fastas.append(fasta)

        # ions
        for ion_idx, ion in enumerate(self.ions):
            for copy_idx in range(ion.count):
                fasta = f">ligand|chain{ion_idx}_copy{copy_idx+1}\n{ion.ion}"
                fastas.append(fasta)

        # write to file
        if output_path is not None:
            abutils.io.make_dir(output_path)
            fasta_path = os.path.join(output_path, f"{self.name}.fasta")
            with open(fasta_path, "w") as f:
                f.write("\n".join(fastas))
            if constraints:
                constraints_path = os.path.join(output_path, f"{self.name}.constraints")
                with open(constraints_path, "w") as f:
                    f.write("\n".join(constraints_header + constraints))
            else:
                constraints_path = None
            return fasta_path, constraints_path
        else:
            return "\n".join(fastas), "\n".join(constraints)


class ProtenixFormattingMixin:
    def build_protenix_input(self, output_path: str | None) -> str:
        """
        Build a Protenix input file. Protenix accepts a JSON file with a format
        that is very similar (but not identical) to the `AlphaFold3 input JSON file`_.

        Parameters
        ----------
        output_path : str | None, optional
            The path to the output directory, into which the Protenix input JSON file will be written.
            If ``None``, the Protenix-formatted input will be returned as a string.

        Returns
        -------
        str
            The path to the Protenix input JSON file.

        .. _AlphaFold3 input JSON file: https://github.com/google-deepmind/alphafold/tree/main/server

        """
        sequences = []
        ligands = []
        ions = []
        covalent_bonds = []

        # get chain names for all entity types
        chain_gen = get_chain_name_generator("protenix")
        protein_chain_names = deque(
            [next(chain_gen) for _ in range(self.num_entities(kind="proteinChain"))]
        )
        _ = deque(  # don't need DNA/RNA chain names, but need to advance the generator
            [
                next(chain_gen)
                for _ in range(
                    self.num_entities(kind="dnaSequence")
                    + self.num_entities(kind="rnaSequence")
                )
            ]
        )
        glycan_chain_names = deque(
            [next(chain_gen) for _ in range(self.num_entities(kind="glycan"))]
        )

        # protein chains
        for chain in self.protein_chains:
            chain_name = protein_chain_names.popleft()
            sequences.append(
                {
                    "proteinChain": {
                        "sequence": chain.sequence,
                        "count": chain.count,
                        "modifications": [
                            {
                                "ptmType": m.modification_type,
                                "ptmPosition": m.position,
                            }
                            for m in chain.modifications
                        ],
                    }
                }
            )
            # add glycans (if present)
            for glycan in chain.glycans:
                glycan_name = glycan_chain_names.popleft()
                ligands.append(
                    {
                        "ligand": {
                            "ligand": glycan.protenix_formatted,
                            "count": chain.count,
                        }
                    }
                )
                # add covalent bonds between glycan and protein
                covalent_bonds.append(
                    {
                        "entity1": chain_name,
                        "position1": glycan.position,
                        "atom1": "ND2",
                        "entity2": glycan_name,
                        "position2": 1,
                        "atom2": "C1",
                    }
                )

        # DNAsequences
        for seq in self.dna_sequences:
            sequences.append(
                {
                    "dnaSequence": {
                        "sequence": seq.sequence,
                        "count": seq.count,
                        "modifications": [
                            {
                                "modificationType": m.modification_type,
                                "basePosition": m.position,
                            }
                            for m in seq.modifications
                        ],
                    }
                }
            )

        # RNA sequences
        for seq in self.rna_sequences:
            sequences.append(
                {
                    "rnaSequence": {
                        "sequence": seq.sequence,
                        "count": seq.count,
                        "modifications": [
                            {
                                "modificationType": m.modification_type,
                                "basePosition": m.position,
                            }
                            for m in seq.modifications
                        ],
                    }
                }
            )

        # ligands
        for ligand in self.ligands:
            ligands.append(
                {
                    "ligand": {
                        "ligand": ligand.ligand,
                        "count": ligand.count,
                    },
                }
            )

        # ions
        for ion in self.ions:
            ions.append(
                {
                    "ion": {
                        "ion": ion.ion,
                        "count": ion.count,
                    },
                }
            )

        # pull all the data together
        data = {
            "name": self.name,
            "sequences": sequences + ligands + ions,
            "covalent_bonds": covalent_bonds,
        }

        # write to file if requested
        if output_path is not None:
            if os.path.isdir(output_path):
                output_path = os.path.join(output_path, f"{self.name}.json")
            abutils.io.make_dir(os.path.dirname(output_path))
            with open(output_path, "w") as f:
                json.dump([data], f, indent=2)
            return output_path
        else:
            return json.dumps(data, indent=2)
