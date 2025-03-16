# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

import json
import os
from collections import deque
from typing import Tuple

import abutils

from .chains import get_chain_name_generator

# =============================================
#
#           MODEL-SPECIFIC FORMATTING
#
# =============================================


class BoltzFormattingMixin:
    def build_boltz_input(self, output_path: str):
        pass


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
                fasta = f">ion|chain{ion_idx}_copy{copy_idx+1}\n{ion.ion}"
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
