# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, NeighborSearch, PDBParser
from Bio.PDB.Superimposer import Superimposer

__all__ = [
    "identify_contacts",
    "identify_interface_residues",
    "find_antibody_bound_antigen_chain",
    "rmsd",
    "ssRMSD",
    "fnat",
    "mean_fnat",
    "iRMSD",
    "mean_iRMSD",
]


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
    return parser.get_structure(file_path.stem, file_path)[0]  # model 0


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
            # neighbours within cut_off Å
            for nbr in ns.search(atom.coord, cut_off):
                contacts.add(nbr)
            # absolute minimum distance (quick 10 Å shell)
            near = ns.search(atom.coord, 10.0, level="A")
            if near:
                dmin = min(dmin, min((atom - a for a in near)))
        contact_counts[prt] = len(contacts)
        min_distances[prt] = dmin

    # pick protomer with most contacts (tie‑break by min distance)
    best = max(contact_counts, key=lambda c: (contact_counts[c], -min_distances[c]))
    return best, contact_counts, min_distances


# -----------------------------------------
#               ssRMSD
# -----------------------------------------


def rmsd(
    file_path1: str,
    file_path2: str,
    chains: str | list[str],
    atom_types: str | list[str] = ["CA"],
    quiet: bool = True,
) -> float:
    """
    Calculate the Root Mean Square Deviation (RMSD) between two structures for specific chains.

    Parameters
    ----------
    file_path1 : str
        The path to the first PDB/CIF file.

    file_path2 : str
        The path to the second PDB/CIF file.

    chains : str or list[str]
        The chain ID(s) for which to calculate RMSD.

    atom_types : str or list[str], optional, default=["CA"]
        The atom type(s) to use for RMSD calculation. Default is CA atoms only.

    quiet : bool, optional, default=True
        Suppress verbose output from the PDB parser.

    Returns
    -------
    float
        The RMSD value between the specified chains of the two structures.

    Raises
    ------
    ValueError
        If the specified chains are not found in both structures or if the number of atoms
        doesn't match.
    """

    # convert to Path objects
    if isinstance(file_path1, str):
        file_path1 = Path(file_path1)
    if isinstance(file_path2, str):
        file_path2 = Path(file_path2)

    # convert single chain or atom type to list
    if isinstance(chains, str):
        chains = [chains]
    if isinstance(atom_types, str):
        atom_types = [atom_types]

    # get structures
    struct1 = _get_structure(file_path1, quiet=quiet)
    struct2 = _get_structure(file_path2, quiet=quiet)

    # extract specific atoms from specified chains
    atoms1, atoms2 = [], []
    for chain_id in chains:
        if chain_id not in struct1.child_dict or chain_id not in struct2.child_dict:
            raise ValueError(f"Chain {chain_id} not found in both structures")
        chain1 = struct1[chain_id]
        chain2 = struct2[chain_id]
        # get atoms of specified types
        chain1_atoms = [
            atom
            for atom in chain1.get_atoms()
            if atom.name in atom_types and atom.parent.id[0] == " "
        ]
        chain2_atoms = [
            atom
            for atom in chain2.get_atoms()
            if atom.name in atom_types and atom.parent.id[0] == " "
        ]
        # ensure same number of atoms
        if len(chain1_atoms) != len(chain2_atoms):
            raise ValueError(
                f"Number of atoms in chain {chain_id} doesn't match between structures"
            )
        atoms1.extend(chain1_atoms)
        atoms2.extend(chain2_atoms)

    # verify that we have atoms to align
    if not atoms1 or not atoms2:
        raise ValueError(f"No matching atoms found in specified chains: {chains}")

    # superimpose structures
    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)

    return sup.rms


def ssRMSD(
    files: list[str] | str,
    chains: str | list[str],
    atom_types: str | list[str] = ["CA"],
    quiet: bool = True,
    log_dir: str | Path | None = None,
) -> float:
    """
    Calculate the sum of squared RMSD values for all pairs of PDB/CIF files.

    Parameters
    ----------
    files : list[str] or str
        Either a list of PDB/CIF file paths or a directory path containing PDB/CIF files.

    chains : str or list[str]
        The chain ID(s) for which to calculate RMSD.

    atom_types : str or list[str], optional, default=["CA"]
        The atom type(s) to use for RMSD calculation. Default is CA atoms only.

    quiet : bool, optional, default=True
        Suppress verbose output from the PDB parser.

    log_dir : str or Path or None, optional, default=None
        Directory path to save CSV log file with individual RMSD values.
        If not provided, no log file will be generated.

    Returns
    -------
    float
        The sum of squared RMSD values.

    Raises
    ------
    ValueError
        If less than two files are provided or found in the specified directory.
    FileNotFoundError
        If the specified directory or files do not exist.
    """

    # input is a directory
    if isinstance(files, str):
        dir_path = Path(files)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory does not exist: {dir_path}")
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {dir_path}")

        # Get all PDB and CIF files in the directory
        pdb_files = list(dir_path.glob("*.pdb")) + list(dir_path.glob("*.ent"))
        cif_files = list(dir_path.glob("*.cif")) + list(dir_path.glob("*.mmcif"))
        files = pdb_files + cif_files
    else:
        # convert string paths to Path objects
        files = [Path(f) if isinstance(f, str) else f for f in files]

        # verify all files exist
        for file_path in files:
            if not file_path.exists():
                raise FileNotFoundError(f"File does not exist: {file_path}")

    # need at least 2 files for pairwise comparisons
    if len(files) < 2:
        raise ValueError("At least two PDB/CIF files are required for RMSD calculation")

    # create log directory if provided and it doesn't exist
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    # calculate RMSD for each pair of files
    rmsd_data = []
    for file1, file2 in itertools.combinations(files, 2):
        # pair_key = (str(file1), str(file2))
        try:
            rmsd_val = rmsd(file1, file2, chains, atom_types, quiet)
            rmsd_data.append(
                {"filepath_1": file1, "filepath_2": file2, "rmsd": rmsd_val}
            )
        except Exception as e:
            print(f"Error calculating RMSD for {file1} and {file2}: {e}")
            continue

    # calculate sum of squared RMSD values
    rmsd_values = [r["rmsd"] for r in rmsd_data]
    ss_rmsd = np.sum(np.square(rmsd_values))

    # wave RMSD values to CSV if log directory is provided
    if log_dir is not None and rmsd_data:
        # create DataFrame and save to CSV
        df = pd.DataFrame(rmsd_data)
        # save to CSV
        csv_path = os.path.join(log_dir, "rmsd_values.csv")
        df.to_csv(csv_path, index=False)

    return ss_rmsd


# -----------------------------------------
#               fnat
# -----------------------------------------


def identify_contacts(
    file_path: str | Path,
    antibody_chains: list[str],
    antigen_chains: list[str] | None = None,
    cut_off: float = 5.0,
    quiet: bool = True,
) -> set:
    """
    Identify atom-level contacts between antibody and antigen chains.

    Parameters
    ----------
    file_path : str or Path
        The path to the PDB/CIF file.

    antibody_chains : list[str]
        The chain ID(s) to consider as antibody.

    antigen_chains : list[str] or None, optional, default=None
        The chain ID(s) to consider as antigen. If not provided, the function will
        use find_antibody_bound_antigen_chain to identify the appropriate chain.

    cut_off : float, optional, default=5.0
        The cutoff distance (in Ångstroms) for a pair of atoms to be considered in contact.

    quiet : bool, optional, default=True
        Suppress verbose output from the PDB parser.

    Returns
    -------
    set
        A set of ((ab_residue_id, ab_atom_id), (ag_residue_id, ag_atom_id)) tuples
        representing contacts between antibody and antigen atoms.
    """
    # Convert file_path to Path if it's a string
    if isinstance(file_path, str):
        file_path = Path(file_path)

    # read input structure
    struct = _get_structure(file_path, quiet=quiet)

    # If antigen_chains is None, find the antigen chain with most contacts
    if antigen_chains is None:
        ag_chain, _, _ = find_antibody_bound_antigen_chain(
            file_path, antibody_chains=antibody_chains, cut_off=cut_off
        )
        antigen_chains = [ag_chain]

    # gather antibody atoms
    ab_atoms = []
    for ch in struct:
        if ch.id in antibody_chains:
            ab_atoms.extend(atoms_of(ch))

    # Handle case when no antibody atoms are found (e.g., chain doesn't exist)
    if not ab_atoms:
        return set()

    # gather antigen atoms and identify contacts
    contacts = set()
    for ch in struct:
        if ch.id in antigen_chains:
            ag_atoms = atoms_of(ch)

            # Skip if no antigen atoms are found in this chain
            if not ag_atoms:
                continue

            # Use NeighborSearch to find atoms within cut_off distance
            ns = NeighborSearch(ab_atoms)
            for ag_atom in ag_atoms:
                for ab_atom in ns.search(ag_atom.coord, cut_off, level="A"):
                    # Store a unique identifier for the contact
                    ab_id = (ab_atom.get_parent().id, ab_atom.get_id())
                    ag_id = (ag_atom.get_parent().id, ag_atom.get_id())
                    contacts.add((ab_id, ag_id))

    return contacts


def fnat(
    filepath_1: str | Path,
    filepath_2: str | Path,
    antibody_chains: list[str],
    antigen_chains: list[str] | None = None,
    cut_off: float = 5.0,
    quiet: bool = True,
) -> float:
    """
    Calculate the fraction of native contacts (fnat) between antibody and antigen chains
    for a pair of PDB/CIF files.

    The fraction of native contacts is defined as the number of contacts in the second structure
    that are also present in the first (reference/native) structure, divided by the total
    number of contacts in the first structure.

    Parameters
    ----------
    filepath_1 : str or Path
        Path to the first (reference/native) PDB/CIF file.

    filepath_2 : str or Path
        Path to the second (model) PDB/CIF file.

    antibody_chains : list[str]
        The chain ID(s) to consider as antibody.

    antigen_chains : list[str] or None, optional, default=None
        The chain ID(s) to consider as antigen. If not provided, the function will
        use find_antibody_bound_antigen_chain to identify the appropriate chain.

    cut_off : float, optional, default=5.0
        The cutoff distance (in Ångstroms) for a pair of atoms to be considered in contact.

    quiet : bool, optional, default=True
        Suppress verbose output from the PDB parser.

    Returns
    -------
    float
        The fraction of native contacts between the two structures.
    """
    try:
        # Identify contacts in both structures
        native_contacts = identify_contacts(
            filepath_1, antibody_chains, antigen_chains, cut_off, quiet
        )
        model_contacts = identify_contacts(
            filepath_2, antibody_chains, antigen_chains, cut_off, quiet
        )

        # Calculate the fraction of native contacts
        if len(native_contacts) > 0:
            # Count shared contacts
            shared_contacts = native_contacts.intersection(model_contacts)
            return len(shared_contacts) / len(native_contacts)
        else:
            # No contacts in the native structure
            return 0.0
    except Exception as e:
        print(f"Error calculating fnat for {filepath_1} and {filepath_2}: {e}")
        return 0.0


def mean_fnat(
    files: list[str] | str,
    antibody_chains: str | list[str] = ["A", "B"],
    antigen_chains: str | list[str] | None = None,
    cut_off: float = 5.0,
    quiet: bool = True,
    log_dir: str | Path | None = None,
) -> float:
    """
    Calculate the mean fraction of native contacts (fnat) between antibody and antigen chains
    across all permutations of PDB/CIF files.

    The fraction of native contacts is defined as the number of contacts in the second structure
    that are also present in the first (native) structure, divided by the total number of contacts
    in the first structure.

    Parameters
    ----------
    files : list[str] or str
        Either a list of PDB/CIF file paths or a directory path containing PDB/CIF files.

    antibody_chains : str or list[str], optional, default=["A", "B"]
        The chain ID(s) to consider as antibody.

    antigen_chains : str or list[str] or None, optional, default=None
        The chain ID(s) to consider as antigen. If not provided, the function will
        use find_antibody_bound_antigen_chain to identify the appropriate chain.

    cut_off : float, optional, default=5.0
        The cutoff distance (in Ångstroms) for a pair of atoms to be considered in contact.

    quiet : bool, optional, default=True
        Suppress verbose output from the PDB parser.

    log_dir : str or Path or None, optional, default=None
        Directory path to save CSV log file with individual fnat values.
        If not provided, no log file will be generated.

    Returns
    -------
    float
        The mean fraction of native contacts.

    Raises
    ------
    ValueError
        If less than two files are provided or found in the specified directory.
    FileNotFoundError
        If the specified directory or files do not exist.
    """
    # Convert antibody_chains to list if it's a string
    if isinstance(antibody_chains, str):
        antibody_chains = [antibody_chains]

    # Handle files input (directory or list of files)
    if isinstance(files, str):
        dir_path = Path(files)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory does not exist: {dir_path}")
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {dir_path}")

        # Get all PDB and CIF files in the directory
        pdb_files = list(dir_path.glob("*.pdb")) + list(dir_path.glob("*.ent"))
        cif_files = list(dir_path.glob("*.cif")) + list(dir_path.glob("*.mmcif"))
        files = pdb_files + cif_files
    else:
        # convert string paths to Path objects
        files = [Path(f) if isinstance(f, str) else f for f in files]

        # verify all files exist
        for file_path in files:
            if not file_path.exists():
                raise FileNotFoundError(f"File does not exist: {file_path}")

    # need at least 2 files for pairwise comparisons
    if len(files) < 2:
        raise ValueError("At least two PDB/CIF files are required for fnat calculation")

    # create log directory if provided and it doesn't exist
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    # Convert antigen_chains to list if it's a string
    if isinstance(antigen_chains, str):
        antigen_chains = [antigen_chains]

    # Calculate fnat for all permutations of files
    fnat_data = []
    for file1, file2 in itertools.permutations(files, 2):
        # Calculate fnat for this pair of files
        fnat_val = fnat(file1, file2, antibody_chains, antigen_chains, cut_off, quiet)

        fnat_data.append({"filepath_1": file1, "filepath_2": file2, "fnat": fnat_val})

    # Calculate mean fnat value
    if not fnat_data:
        return 0.0  # Return 0 if no valid fnat calculations

    fnat_values = [r["fnat"] for r in fnat_data]
    mean_fnat_val = np.mean(fnat_values)

    # Save fnat values to CSV if log directory is provided
    if log_dir is not None and fnat_data:
        df = pd.DataFrame(fnat_data)
        csv_path = os.path.join(log_dir, "fnat_values.csv")
        df.to_csv(csv_path, index=False)

    return mean_fnat_val


# -----------------------------------------
#               iRMSD
# -----------------------------------------


def identify_interface_residues(
    file_path: str | Path,
    antibody_chains: list[str],
    antigen_chains: list[str] | None = None,
    interface_cutoff: float = 10.0,
    quiet: bool = True,
) -> tuple[list, list]:
    """
    Identify interface residues between antibody and antigen chains.
    Interface residues are defined as residues with at least one heavy atom
    within interface_cutoff Angstroms of any heavy atom in the partner chain.

    Parameters
    ----------
    file_path : str or Path
        The path to the PDB/CIF file.

    antibody_chains : list[str]
        The chain ID(s) to consider as antibody.

    antigen_chains : list[str] or None, optional, default=None
        The chain ID(s) to consider as antigen. If not provided, the function will
        use find_antibody_bound_antigen_chain to identify the appropriate chain.

    interface_cutoff : float, optional, default=10.0
        The cutoff distance (in Ångstroms) for a residue to be considered part of the interface.
        Default is 10.0Å as used by DockQ.

    quiet : bool, optional, default=True
        Suppress verbose output from the PDB parser.

    Returns
    -------
    tuple
        A tuple containing two lists: antibody interface residues and antigen interface residues.
    """
    # convert file_path to Path if it's a string
    if isinstance(file_path, str):
        file_path = Path(file_path)

    # read input structure
    struct = _get_structure(file_path, quiet=quiet)

    # rf antigen_chains is None, find the antigen chain with most contacts
    if antigen_chains is None:
        ag_chain, _, _ = find_antibody_bound_antigen_chain(
            file_path, antibody_chains=antibody_chains, cut_off=5.0
        )
        antigen_chains = [ag_chain]

    # gather antibody and antigen atoms
    ab_atoms = []
    ag_atoms = []
    for ch in struct:
        if ch.id in antibody_chains:
            ab_atoms.extend(atoms_of(ch))
        elif ch.id in antigen_chains:
            ag_atoms.extend(atoms_of(ch))

    # handle case when no atoms are found
    if not ab_atoms or not ag_atoms:
        return [], []

    # identify antibody interface residues
    ab_interface_residues = set()
    ns_ag = NeighborSearch(ag_atoms)
    for ab_atom in ab_atoms:
        if ns_ag.search(ab_atom.coord, interface_cutoff, level="A"):
            ab_interface_residues.add(ab_atom.get_parent())

    # identify antigen interface residues
    ag_interface_residues = set()
    ns_ab = NeighborSearch(ab_atoms)
    for ag_atom in ag_atoms:
        if ns_ab.search(ag_atom.coord, interface_cutoff, level="A"):
            ag_interface_residues.add(ag_atom.get_parent())

    return list(ab_interface_residues), list(ag_interface_residues)


def iRMSD(
    file_path1: str | Path,
    file_path2: str | Path,
    antibody_chains: str | list[str],
    antigen_chains: list[str] | None = None,
    interface_cutoff: float = 10.0,
    atom_types: str | list[str] = ["CA"],
    quiet: bool = True,
) -> float:
    """
    Calculate the interface Root Mean Square Deviation (iRMSD) between two structures.

    The interface is defined as those residues with any atom within interface_cutoff Angstroms
    of an atom from the other chain. The iRMSD is calculated by superimposing the interface
    residues and calculating the RMSD of the interface residues after superposition.
    This is implemented as DockQ does.

    Parameters
    ----------
    file_path1 : str or Path
        The path to the first (reference/native) PDB/CIF file.

    file_path2 : str or Path
        The path to the second (model) PDB/CIF file.

    antibody_chains : str or list[str]
        The chain ID(s) to consider as antibody. Can be a single chain or multiple chains.

    antigen_chains : list[str] or None, optional, default=None
        The chain ID(s) to consider as antigen. If not provided, the function will
        use find_antibody_bound_antigen_chain to identify the appropriate chain.

    interface_cutoff : float, optional, default=10.0
        The cutoff distance (in Ångstroms) for a residue to be considered part of the interface.
        Default is 10.0Å as used by DockQ.

    atom_types : str or list[str], optional, default=["CA"]
        The atom type(s) to use for iRMSD calculation. Default is CA atoms only.

    quiet : bool, optional, default=True
        Suppress verbose output from the PDB parser.

    Returns
    -------
    float
        The iRMSD value between the interfaces of the two structures.
    """
    # convert paths to Path objects
    if isinstance(file_path1, str):
        file_path1 = Path(file_path1)
    if isinstance(file_path2, str):
        file_path2 = Path(file_path2)

    # convert input parameters to lists if they are strings
    if isinstance(antibody_chains, str):
        antibody_chains = [antibody_chains]
    if isinstance(antigen_chains, str) and antigen_chains is not None:
        antigen_chains = [antigen_chains]
    if isinstance(atom_types, str):
        atom_types = [atom_types]

    # get structures
    struct1 = _get_structure(file_path1, quiet=quiet)
    struct2 = _get_structure(file_path2, quiet=quiet)

    # identify interface residues in the first (native) structure
    ab_interface_res1, ag_interface_res1 = identify_interface_residues(
        file_path1, antibody_chains, antigen_chains, interface_cutoff, quiet
    )

    # combine all interface residues
    interface_residues1 = ab_interface_res1 + ag_interface_res1

    if not interface_residues1:
        raise ValueError("No interface residues found in the native structure")

    # extract interface residue IDs for matching in the model
    interface_res_ids1 = []
    for res in interface_residues1:
        chain_id = res.get_parent().id
        res_id = res.id
        interface_res_ids1.append((chain_id, res_id))

    # collect interface atoms from both structures
    atoms1 = []
    atoms2 = []

    # dictionary to map chains between structures
    chain_map = {}
    for ab_chain in antibody_chains:
        chain_map[ab_chain] = ab_chain
    if antigen_chains:
        for ag_chain in antigen_chains:
            chain_map[ag_chain] = ag_chain

    # collect interface atoms from structure 1
    for chain_id, res_id in interface_res_ids1:
        if chain_id in struct1:
            if res_id in struct1[chain_id]:
                res = struct1[chain_id][res_id]
                for atom in res:
                    if atom.name in atom_types and atom.parent.id[0] == " ":
                        atoms1.append(atom)

    # find corresponding residues in structure 2 using chain mapping
    for chain_id, res_id in interface_res_ids1:
        mapped_chain_id = chain_map.get(chain_id, chain_id)
        if mapped_chain_id in struct2:
            if res_id in struct2[mapped_chain_id]:
                res = struct2[mapped_chain_id][res_id]
                for atom in res:
                    if atom.name in atom_types and atom.parent.id[0] == " ":
                        atoms2.append(atom)

    # ensure same number of interface atoms
    if len(atoms1) != len(atoms2):
        raise ValueError(
            f"Number of interface atoms doesn't match between structures: {len(atoms1)} vs {len(atoms2)}"
        )

    # verify that we have atoms to align
    if not atoms1 or not atoms2:
        raise ValueError("No matching interface atoms found in both structures")

    # superimpose interface residues
    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)

    # return the RMSD of the superimposed interface residues
    return sup.rms


def mean_iRMSD(
    files: list[str] | str,
    antibody_chains: str | list[str],
    antigen_chains: list[str] | None = None,
    interface_cutoff: float = 10.0,
    atom_types: str | list[str] = ["CA"],
    quiet: bool = True,
    log_dir: str | Path | None = None,
) -> float:
    """
    Calculate the mean interface RMSD (iRMSD) for multiple pairs of PDB/CIF files.

    Parameters
    ----------
    files : list[str] or str
        Either a list of PDB/CIF file paths or a directory path containing PDB/CIF files.

    antibody_chains : str or list[str]
        The chain ID(s) to consider as antibody. Can be a single chain or multiple chains.

    antigen_chains : list[str] or None, optional, default=None
        The chain ID(s) to consider as antigen. If not provided, the function will
        use find_antibody_bound_antigen_chain to identify the appropriate chain.

    interface_cutoff : float, optional, default=10.0
        The cutoff distance (in Ångstroms) for a residue to be considered part of the interface.
        Default is 10.0Å as used by DockQ.

    atom_types : str or list[str], optional, default=["CA"]
        The atom type(s) to use for iRMSD calculation. Default is CA atoms only.

    quiet : bool, optional, default=True
        Suppress verbose output from the PDB parser.

    log_dir : str or Path or None, optional, default=None
        Directory path to save CSV log file with individual iRMSD values.
        If not provided, no log file will be generated.

    Returns
    -------
    float
        The mean iRMSD across all file pairs.

    Raises
    ------
    ValueError
        If less than two files are provided or found in the specified directory.
    FileNotFoundError
        If the specified directory or files do not exist.
    """
    # convert antibody_chains to list if it's a string
    if isinstance(antibody_chains, str):
        antibody_chains = [antibody_chains]

    # handle input files (directory or list of files)
    if isinstance(files, str):
        dir_path = Path(files)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory does not exist: {dir_path}")
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {dir_path}")

        # get all PDB and CIF files in the directory
        pdb_files = list(dir_path.glob("*.pdb")) + list(dir_path.glob("*.ent"))
        cif_files = list(dir_path.glob("*.cif")) + list(dir_path.glob("*.mmcif"))
        files = pdb_files + cif_files
    else:
        # convert string paths to Path objects
        files = [Path(f) if isinstance(f, str) else f for f in files]

        # verify all files exist
        for file_path in files:
            if not file_path.exists():
                raise FileNotFoundError(f"File does not exist: {file_path}")

    # need at least 2 files for pairwise comparisons
    if len(files) < 2:
        raise ValueError(
            "At least two PDB/CIF files are required for iRMSD calculation"
        )

    # create log directory if provided and it doesn't exist
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    # convert antigen_chains to list if it's a string
    if isinstance(antigen_chains, str) and antigen_chains is not None:
        antigen_chains = [antigen_chains]

    # calculate iRMSD for all combinations of file pairs
    irmsd_data = []
    for file1, file2 in itertools.combinations(files, 2):
        try:
            irmsd_val = iRMSD(
                file1,
                file2,
                antibody_chains,
                antigen_chains,
                interface_cutoff,
                atom_types,
                quiet,
            )
            irmsd_data.append(
                {"filepath_1": file1, "filepath_2": file2, "irmsd": irmsd_val}
            )
        except Exception as e:
            print(f"Error calculating iRMSD for {file1} and {file2}: {e}")
            continue

    # calculate mean iRMSD value
    if not irmsd_data:
        return 0.0  # return 0 if no valid iRMSD calculations

    irmsd_values = [r["irmsd"] for r in irmsd_data]
    mean_irmsd_val = np.mean(irmsd_values)

    # save iRMSD values to CSV if log directory is provided
    if log_dir is not None and irmsd_data:
        df = pd.DataFrame(irmsd_data)
        csv_path = os.path.join(log_dir, "irmsd_values.csv")
        df.to_csv(csv_path, index=False)

    return mean_irmsd_val
