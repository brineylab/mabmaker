# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, NeighborSearch, PDBParser


def _get_structure(file_path, quiet: bool = True):
    """
    Parse a PDB or mmCIF file and return the first model object.

    Parameters
    ----------
    file_path : str
        The path to the file to parse.

    quiet : bool, optional, default=True
        Passed directly to the parser (e.g. ``PDBParser(QUIET=quiet)`` or
        ``MMCIFParser(QUIET=quiet)``).

    Returns
    -------
    Bio.PDB.Structure
        The first model object in the file.

    """
    suf = file_path.suffix.lower()
    if suf in {".pdb", ".ent"}:
        parser = PDBParser(QUIET=quiet)
    elif suf in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=quiet)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    return parser.get_structure(file_path.stem, file_path)[0]  # model 0


def atoms_of(chain, heavy_only: bool = True) -> list:
    """
    Return a list of atoms from a Bio.PDB.Chain object.

    Parameters
    ----------
    chain : Bio.PDB.Chain
        The chain to get atoms from.

    heavy_only : bool, optional, default=True
        Whether to only return heavy atoms (i.e. exclude hydrogens).

    Returns
    -------
    list
        A list of atoms.

    """
    if heavy_only:
        return [a for a in chain.get_atoms() if a.element != "H"]
    return list(chain.get_atoms())


def find_antibody_bound_antigen_chain(
    file_path: str,
    antibody_chains: list[str] = ["A", "B"],
    antigen_chains: list[str] | None = None,
    cut_off: float = 5.0,
) -> tuple[str, dict[str, int], dict[str, float]]:
    """
    Identify which antigen chain is bound by the antibody. This is determined by
    counting the number of contacts between the antibody and the antigen, and
    selecting the antigen chain with the most contacts. If there is a tie, the
    antigen chain with the smallest minimum distance to the antibody is selected.

    Parameters
    ----------
    file_path : str
        The path to the file to parse.

    antibody_chains : list[str], optional, default=["A", "B"]
        The chains to consider as antibody.

    antigen_chains : list[str], optional
        The chains to consider as antigen. If not provided, all chains except the
        antibody chains will be considered antigen chains.

    cut_off : float, optional, default=5.0
        The cutoff distance (in Ångstroms) for a pair of residues to be considered
        in contact.

    Returns
    -------
    tuple
        A tuple of the antibody-bound antigen chain name, the number of antibody contacts for each antigen chain,
        and the minimum distances for each antigen chain.

    """
    # read input structure
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input PDB/CIF file does not exist: {file_path}")
    struct = _get_structure(file_path)

    # antigen chains
    if antigen_chains is None:
        antigen_chains = [
            c for c in struct.child_dict.keys() if c not in antibody_chains
        ]

    # gather atoms
    ab_atoms = []
    ag_atoms = {c: [] for c in antigen_chains}
    for ch in struct:
        if ch.id in antibody_chains:
            ab_atoms.extend(atoms_of(ch))
        elif ch.id in antigen_chains:
            ag_atoms[ch.id].extend(atoms_of(ch))

    # neighbour search
    ns = NeighborSearch(ab_atoms)
    contact_counts, min_distances = {}, {}
    for prt, atoms in ag_atoms.items():
        contacts, dmin = set(), np.inf
        for atom in atoms:
            # neighbours within cut_off Å
            for nbr in ns.search(atom.coord, cut_off):
                contacts.add(nbr)
            # absolute minimum distance (quick 10 Å shell)
            near = ns.search(atom.coord, 10.0, level="A")
            if near:
                dmin = min(dmin, min((atom - a for a in near)))
        contact_counts[prt] = len(contacts)
        min_distances[prt] = dmin

    # pick protomer with most contacts (tie‑break by min distance)
    best = max(contact_counts, key=lambda c: (contact_counts[c], -min_distances[c]))
    return best, contact_counts, min_distances
