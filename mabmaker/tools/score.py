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
    "chain_COM_distance",
    "mean_COM_distance",
    "approach_angle",
    "approach_angle_variance",
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


def _get_pdb_and_cif_files(files: str | Path | list[str | Path]) -> list[Path]:
    """
    Get all PDB and CIF files in the directory.

    Parameters
    ----------
    files : str | Path | list[str | Path]
        The path to the directory containing the PDB/CIF files or a list of file paths.

    Returns
    -------
    list[Path]
        A list of PDB and CIF files.
    """
    if isinstance(files, str):  # directory
        dir_path = Path(files)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory does not exist: {dir_path}")
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {dir_path}")
        # get all PDB and CIF files in the directory
        pdb_files = list(dir_path.glob("*.pdb")) + list(dir_path.glob("*.ent"))
        cif_files = list(dir_path.glob("*.cif")) + list(dir_path.glob("*.mmcif"))
        files = pdb_files + cif_files

    else:  # list of files
        files = [Path(f) if isinstance(f, str) else f for f in files]
        for file_path in files:
            if not file_path.exists():
                raise FileNotFoundError(f"File does not exist: {file_path}")


def _get_atoms_of(chain, heavy_only: bool = True) -> list:
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
    # inputs
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input PDB/CIF file does not exist: {file_path}")

    struct = _get_structure(file_path)

    if antigen_chains is None:
        antigen_chains = [
            c for c in struct.child_dict.keys() if c not in antibody_chains
        ]

    # get Ab and Ag atoms
    ab_atoms = []
    ag_atoms = {c: [] for c in antigen_chains}

    for ch in struct:
        if ch.id in antibody_chains:
            ab_atoms.extend(_get_atoms_of(ch))
        elif ch.id in antigen_chains:
            ag_atoms[ch.id].extend(_get_atoms_of(ch))

    if not ab_atoms:
        raise ValueError(f"No antibody atoms found in chains: {antibody_chains}")
    if not any(atoms for atoms in ag_atoms.values()):
        raise ValueError(f"No antigen atoms found in chains: {antigen_chains}")

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
    if not contact_counts:
        raise ValueError("No contacts found between antibody and antigen chains")

    # pick protomer with the most contacts (tie‑break by min distance)
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
    align_chains: str | list[str] | None = None,
    align_antibody_bound_chain: bool = False,
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

    align_chains : str or list[str] or None, optional, default=None
        The chain ID(s) to use for structure alignment. If None, alignment will be based on
        the antibody-bound chain or the chains specified in the chains parameter.

    align_antibody_bound_chain : bool, optional, default=False
        If True and align_chains is None, use the antibody-bound chain for alignment.
        If False and align_chains is None, use the chains specified in the chains parameter
        for both alignment and RMSD calculation.

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
    if isinstance(align_chains, str):
        align_chains = [align_chains]

    # get structures
    struct1 = _get_structure(file_path1, quiet=quiet)
    struct2 = _get_structure(file_path2, quiet=quiet)

    # check RMSD chains first
    for chain_id in chains:
        if chain_id not in struct1.child_dict or chain_id not in struct2.child_dict:
            raise ValueError(f"RMSD chain {chain_id} not found in both structures")

    # determine alignment chains
    if align_chains is None:
        if align_antibody_bound_chain:
            # find antibody-bound chain for alignment
            ag_chain1, _, _ = find_antibody_bound_antigen_chain(file_path1)
            ag_chain2, _, _ = find_antibody_bound_antigen_chain(file_path2)
            align_chains = [
                ag_chain1
            ]  # use only the first structure's chain for alignment
        else:
            # use the same chains for both alignment and RMSD
            align_chains = chains

    # extract atoms for alignment
    align_atoms1, align_atoms2 = [], []
    for chain_id in align_chains:
        if chain_id not in struct1.child_dict or chain_id not in struct2.child_dict:
            raise ValueError(f"Alignment chain {chain_id} not found in both structures")
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
                f"Number of atoms in alignment chain {chain_id} doesn't match between structures"
            )
        align_atoms1.extend(chain1_atoms)
        align_atoms2.extend(chain2_atoms)

    # verify that we have atoms to align
    if not align_atoms1 or not align_atoms2:
        raise ValueError(f"No matching atoms found in alignment chains: {align_chains}")

    # superimpose structures
    sup = Superimposer()
    sup.set_atoms(align_atoms1, align_atoms2)

    # apply rotation/translation to the entire second structure
    sup.apply(struct2.get_atoms())

    # extract atoms for RMSD calculation
    rmsd_atoms1, rmsd_atoms2 = [], []
    for chain_id in chains:
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
                f"Number of atoms in RMSD chain {chain_id} doesn't match between structures"
            )
        rmsd_atoms1.extend(chain1_atoms)
        rmsd_atoms2.extend(chain2_atoms)

    # verify that we have atoms for RMSD calculation
    if not rmsd_atoms1 or not rmsd_atoms2:
        raise ValueError(f"No matching atoms found in RMSD chains: {chains}")

    # calculate RMSD
    return np.sqrt(
        np.mean(
            np.sum(
                (
                    np.array([a.coord for a in rmsd_atoms1])
                    - np.array([a.coord for a in rmsd_atoms2])
                )
                ** 2,
                axis=1,
            )
        )
    )


def ssRMSD(
    file_paths: str | list[str],
    antibody_chains: str | list[str] = ["A", "B"],
    antigen_chains: str | list[str] | None = None,
    atom_types: str | list[str] = ["CA"],
    align_antibody_bound_chain: bool = False,
    log_dir: str | None = None,
) -> float:
    """
    Calculate the sum of squared RMSD values between all pairs of structures.

    Parameters
    ----------
    file_paths : str or list[str]
        Path to a directory containing PDB/CIF files or a list of file paths.

    antibody_chains : str or list[str], optional, default=["A", "B"]
        Chain ID(s) for which to calculate RMSD. Default is ["A", "B"].

    antigen_chains : str or list[str] or None, optional, default=None
        Chain ID(s) for which to calculate RMSD. If not provided, the function will
        use find_antibody_bound_antigen_chain to identify the appropriate chain.

    atom_types : str or list[str], optional, default=["CA"]
        Atom type(s) to use for RMSD calculation.

    log_dir : str or None, optional, default=None
        Directory to save RMSD values in a CSV file.

    Returns
    -------
    float
        Sum of squared RMSD values between all pairs of structures.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    ValueError
        If less than two PDB/CIF files are provided.
    """
    # convert to Path objects and get list of files
    file_paths = _get_pdb_and_cif_files(file_paths)

    if len(file_paths) < 2:
        raise ValueError("At least two PDB/CIF files are required")

    # calculate RMSD between all pairs of files and store results
    rmsd_data = []
    for file1, file2 in itertools.combinations(file_paths, 2):
        try:
            rmsd_val = rmsd(
                file1,
                file2,
                chains=antibody_chains,
                align_chains=antigen_chains,
                atom_types=atom_types,
                align_antibody_bound_chain=align_antibody_bound_chain,
            )
            rmsd_data.append(
                {"filepath_1": str(file1), "filepath_2": str(file2), "rmsd": rmsd_val}
            )
        except Exception as e:
            print(f"Error calculating RMSD for {file1} and {file2}: {str(e)}")
            continue

    # save RMSD values to CSV if log_dir is provided
    if log_dir is not None and rmsd_data:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rmsd_data)
        df.to_csv(log_dir / "rmsd_values.csv", index=False)

    # return sum of squared RMSD values
    return sum(entry["rmsd"] ** 2 for entry in rmsd_data)


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
    # inputs
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if antigen_chains is None:
        ag_chain, _, _ = find_antibody_bound_antigen_chain(
            file_path, antibody_chains=antibody_chains, cut_off=cut_off
        )
        antigen_chains = [ag_chain]

    struct = _get_structure(file_path, quiet=quiet)

    # gather Ab atoms
    ab_atoms = []

    for ch in struct:
        if ch.id in antibody_chains:
            ab_atoms.extend(_get_atoms_of(ch))
    if not ab_atoms:
        return set()

    # gather Ag atoms and identify contacts
    contacts = set()
    for ch in struct:
        if ch.id in antigen_chains:
            ag_atoms = _get_atoms_of(ch)
            if not ag_atoms:
                continue
            # find atoms within cut_off distance
            ns = NeighborSearch(ab_atoms)
            for ag_atom in ag_atoms:
                for ab_atom in ns.search(ag_atom.coord, cut_off, level="A"):
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
    files = _get_pdb_and_cif_files(files)

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
    # inputs
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if antigen_chains is None:
        ag_chain, _, _ = find_antibody_bound_antigen_chain(
            file_path, antibody_chains=antibody_chains, cut_off=5.0
        )
        antigen_chains = [ag_chain]

    struct = _get_structure(file_path, quiet=quiet)

    # gather Ab and Ag atoms
    ab_atoms = []
    ag_atoms = []

    for ch in struct:
        if ch.id in antibody_chains:
            ab_atoms.extend(_get_atoms_of(ch))
        elif ch.id in antigen_chains:
            ag_atoms.extend(_get_atoms_of(ch))

    if not ab_atoms or not ag_atoms:
        return [], []

    # identify Ab interface residues
    ab_interface_residues = set()
    ns_ag = NeighborSearch(ag_atoms)
    for ab_atom in ab_atoms:
        if ns_ag.search(ab_atom.coord, interface_cutoff, level="A"):
            ab_interface_residues.add(ab_atom.get_parent())

    # identify Ag interface residues
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
    # inputs
    if isinstance(antibody_chains, str):
        antibody_chains = [antibody_chains]
    if isinstance(antigen_chains, str) and antigen_chains is not None:
        antigen_chains = [antigen_chains]

    files = _get_pdb_and_cif_files(files)
    if len(files) < 2:
        raise ValueError(
            "At least two PDB/CIF files are required for iRMSD calculation"
        )

    # logging
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

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

    # calculate mean iRMSD
    if not irmsd_data:
        return 0.0  # return 0 if no valid iRMSD calculations

    irmsd_values = [r["irmsd"] for r in irmsd_data]
    mean_irmsd_val = np.mean(irmsd_values)

    # save iRMSD values to CSV if log directory is provided
    if log_dir is not None and irmsd_data:
        df = pd.DataFrame(irmsd_data)
        csv_path = os.path.join(log_dir, "irmsd.csv")
        df.to_csv(csv_path, index=False)

    return mean_irmsd_val


# -----------------------------------------
#             ssCOM distance
# -----------------------------------------


def chain_COM_distance(
    filepath_1: str | Path,
    filepath_2: str | Path,
    antibody_chains: str | list[str] = ["A", "B"],
    antigen_chains: str | list[str] | None = None,
    atom_types: str | list[str] = ["CA"],
    quiet: bool = True,
) -> dict[str, float]:
    """
    Calculate the distance between centers of mass of antibody chains after
    aligning the structures by the antigen chain.

    Parameters
    ----------
    filepath_1 : str or Path
        The path to the first PDB/CIF file.

    filepath_2 : str or Path
        The path to the second PDB/CIF file.

    antibody_chains : str or list[str], optional, default=["A", "B"]
        The chain ID(s) to consider as antibody. Can be a single chain or multiple chains.

    antigen_chains : str or list[str] or None, optional, default=None
        The chain ID(s) to consider as antigen. If not provided, the function will
        use find_antibody_bound_antigen_chain to identify the appropriate chain.

    atom_types : str or list[str], optional, default=["CA"]
        The atom type(s) to use for calculations. Default is CA atoms only.

    quiet : bool, optional, default=True
        Suppress verbose output from the PDB parser.

    Returns
    -------
    dict[str, float]
        Dictionary mapping each antibody chain to the distance (in Ångstroms) between
        the centers of mass of that chain in the two structures.
    """
    # inputs
    if isinstance(filepath_1, str):
        filepath_1 = Path(filepath_1)
    if isinstance(filepath_2, str):
        filepath_2 = Path(filepath_2)

    if isinstance(antibody_chains, str):
        antibody_chains = [antibody_chains]
    if isinstance(antigen_chains, str) and antigen_chains is not None:
        antigen_chains = [antigen_chains]
    if isinstance(atom_types, str):
        atom_types = [atom_types]

    # get structures
    struct1 = _get_structure(filepath_1, quiet=quiet)
    struct2 = _get_structure(filepath_2, quiet=quiet)

    # determine Ag chains if not provided
    if antigen_chains is None:
        antigen_chain1, _, _ = find_antibody_bound_antigen_chain(
            filepath_1, antibody_chains=antibody_chains
        )
        antigen_chain2, _, _ = find_antibody_bound_antigen_chain(
            filepath_2, antibody_chains=antibody_chains
        )
        antigen_chains = [antigen_chain1]

    # extract atoms from Ag chains for alignment
    antigen_atoms1 = []
    antigen_atoms2 = []

    for chain_id in antigen_chains:
        if chain_id in struct1.child_dict and chain_id in struct2.child_dict:
            chain1 = struct1[chain_id]
            chain2 = struct2[chain_id]

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

            # only add atoms if they match in number
            if len(chain1_atoms) == len(chain2_atoms):
                antigen_atoms1.extend(chain1_atoms)
                antigen_atoms2.extend(chain2_atoms)

    # ensure we have atoms to align
    if not antigen_atoms1 or not antigen_atoms2:
        raise ValueError(
            f"No matching antigen atoms found in specified chains: {antigen_chains}"
        )

    # superimpose structures based on Ag atoms
    sup = Superimposer()
    sup.set_atoms(antigen_atoms1, antigen_atoms2)

    # apply rotation/translation to the entire second structure
    sup.apply(struct2.get_atoms())

    # calculate CoM for each Ab chain in both structures
    results = {}
    for chain_id in antibody_chains:
        if chain_id in struct1.child_dict and chain_id in struct2.child_dict:
            chain1_atoms = [
                atom
                for atom in struct1[chain_id].get_atoms()
                if atom.name in atom_types and atom.parent.id[0] == " "
            ]
            chain2_atoms = [
                atom
                for atom in struct2[chain_id].get_atoms()
                if atom.name in atom_types and atom.parent.id[0] == " "
            ]

            # calculate CoM for each chain
            if chain1_atoms and chain2_atoms:
                com1 = np.mean([atom.coord for atom in chain1_atoms], axis=0)
                com2 = np.mean([atom.coord for atom in chain2_atoms], axis=0)

                # calculate Euclidean distance between CoMs
                distance = np.sqrt(np.sum((com1 - com2) ** 2))
                results[chain_id] = distance

    return results


def mean_COM_distance(
    files: list[str] | str,
    antibody_chains: str | list[str] = ["A", "B"],
    antigen_chains: str | list[str] | None = None,
    atom_types: str | list[str] = ["CA"],
    quiet: bool = True,
    log_dir: str | Path | None = None,
) -> dict[str, float]:
    """
    Calculate the sum of squared center of mass distances between antibody chains
    for all pairwise combinations of PDB/CIF files.

    Parameters
    ----------
    files : list[str] or str
        Either a list of PDB/CIF file paths or a directory path containing PDB/CIF files.

    antibody_chains : str or list[str], optional, default=["A", "B"]
        The chain ID(s) to consider as antibody. Can be a single chain or multiple chains.

    antigen_chains : str or list[str] or None, optional, default=None
        The chain ID(s) to consider as antigen. If not provided, the function will
        use find_antibody_bound_antigen_chain to identify the appropriate chain.

    atom_types : str or list[str], optional, default=["CA"]
        The atom type(s) to use for calculations. Default is CA atoms only.

    quiet : bool, optional, default=True
        Suppress verbose output from the PDB parser.

    log_dir : str or Path or None, optional, default=None
        Directory path to save CSV log file with individual COM distance values.
        If not provided, no log file will be generated.

    Returns
    -------
    dict[str, float]
        Dictionary mapping each antibody chain to the sum of squared distances (in Ångstroms)
        between the centers of mass of that chain across all file pairs.
    """
    # inputs
    if isinstance(antibody_chains, str):
        antibody_chains = [antibody_chains]
    if isinstance(antigen_chains, str) and antigen_chains is not None:
        antigen_chains = [antigen_chains]

    files = _get_pdb_and_cif_files(files)
    if len(files) < 2:
        raise ValueError(
            "At least two PDB/CIF files are required for COM distance calculation"
        )

    # logging
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    # calculate CmM distances for all file combinations
    ss_distances = {chain: 0.0 for chain in antibody_chains}
    com_data = []

    for file1, file2 in itertools.combinations(files, 2):
        try:
            distances = chain_COM_distance(
                file1,
                file2,
                antibody_chains,
                antigen_chains,
                atom_types,
                quiet,
            )
            for chain, distance in distances.items():
                ss_distances[chain] += distance**2
            data_entry = {"filepath_1": file1, "filepath_2": file2}
            for chain, distance in distances.items():
                data_entry[f"chain_{chain}_distance"] = distance
            com_data.append(data_entry)

        except Exception as e:
            print(f"Error calculating COM distance for {file1} and {file2}: {e}")
            continue

    # save CoM distance values to CSV if log directory is provided
    if log_dir is not None and com_data:
        df = pd.DataFrame(com_data)
        csv_path = os.path.join(log_dir, "com_distance.csv")
        df.to_csv(csv_path, index=False)

    return ss_distances


# -----------------------------------------
#           Approach Angle
# -----------------------------------------


def approach_angle(
    file_path: str | Path,
    antibody_chains: str | list[str] = ["A", "B"],
    antigen_chains: str | list[str] | None = None,
    interface_cutoff: float = 5.0,
    atom_types: str | list[str] = ["CA"],
    quiet: bool = True,
) -> float:
    """
    Calculate the angle of approach of an antibody bound to an antigen.

    The angle is calculated using three points:
    1. Center of mass of the antigen (using only antibody-bound chain)
    2. Center of mass of the antibody/antigen interface (antigen atoms within interface_cutoff of antibody)
    3. Center of mass of the antibody (all atoms from antibody chains)

    Parameters
    ----------
    file_path : str or Path
        The path to the PDB/CIF file.

    antibody_chains : str or list[str], optional, default=["A", "B"]
        The chain ID(s) to consider as antibody. Can be a single chain or multiple chains.

    antigen_chains : str or list[str] or None, optional, default=None
        The chain ID(s) to consider as antigen. If not provided, the function will
        use find_antibody_bound_antigen_chain to identify the appropriate chain.

    interface_cutoff : float, optional, default=5.0
        The cutoff distance (in Ångstroms) for an antigen atom to be considered part of the interface.

    atom_types : str or list[str], optional, default=["CA"]
        The atom type(s) to use for center of mass calculations. Default is CA atoms only.

    quiet : bool, optional, default=True
        Suppress verbose output from the PDB parser.

    Returns
    -------
    float
        The angle of approach in degrees.
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if isinstance(antibody_chains, str):
        antibody_chains = [antibody_chains]
    if isinstance(atom_types, str):
        atom_types = [atom_types]

    struct = _get_structure(file_path, quiet=quiet)

    # determine antigen chain (if not provided)
    if antigen_chains is None:
        ag_chain, _, _ = find_antibody_bound_antigen_chain(
            file_path, antibody_chains=antibody_chains, cut_off=interface_cutoff
        )
        antigen_chains = [ag_chain]
    elif isinstance(antigen_chains, str):
        antigen_chains = [antigen_chains]

    # Ab and Ag atoms
    ab_atoms = []
    ag_atoms = []

    for ch in struct:
        if ch.id in antibody_chains:
            chain_atoms = [
                atom
                for atom in ch.get_atoms()
                if atom.name in atom_types and atom.parent.id[0] == " "
            ]
            ab_atoms.extend(chain_atoms)
        elif ch.id in antigen_chains:
            chain_atoms = [
                atom
                for atom in ch.get_atoms()
                if atom.name in atom_types and atom.parent.id[0] == " "
            ]
            ag_atoms.extend(chain_atoms)
    if not ab_atoms:
        raise ValueError(f"No antibody atoms found in chains: {antibody_chains}")
    if not ag_atoms:
        raise ValueError(f"No antigen atoms found in chains: {antigen_chains}")

    # calculate Ab and Ag CoM
    ab_com = np.mean([atom.coord for atom in ab_atoms], axis=0)
    ag_com = np.mean([atom.coord for atom in ag_atoms], axis=0)

    # identify interface atoms (Ag atoms within interface_cutoff of any Ab atom)
    ns = NeighborSearch(ab_atoms)
    interface_atoms = []
    for ag_atom in ag_atoms:
        if ns.search(ag_atom.coord, interface_cutoff, level="A"):
            interface_atoms.append(ag_atom)
    if not interface_atoms:
        raise ValueError(f"No interface atoms found within {interface_cutoff}Å cutoff")

    # calculate interface CoM
    interface_com = np.mean([atom.coord for atom in interface_atoms], axis=0)

    # calculate approach angle
    vec1 = ag_com - interface_com  # vector from interface to antigen COM
    unit_vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = ab_com - interface_com  # vector from interface to antibody COM
    unit_vec2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.clip(np.dot(unit_vec1, unit_vec2), -1.0, 1.0)
    angle = np.degrees(np.arccos(dot_product))

    return angle

    # # Calculate angle between vectors
    # norm1 = np.linalg.norm(vec1)
    # norm2 = np.linalg.norm(vec2)

    # # Handle case when vectors have zero magnitude
    # if norm1 < 1e-6 or norm2 < 1e-6:
    #     raise ValueError(
    #         "Unable to calculate approach angle: zero-magnitude vector detected"
    #     )

    # # Calculate dot product and handle numerical precision issues
    # dot_product = np.dot(vec1, vec2)
    # cos_angle = dot_product / (norm1 * norm2)

    # # Ensure cos_angle is within valid range [-1, 1] to avoid NaN results
    # cos_angle = max(min(cos_angle, 1.0), -1.0)

    # # Calculate angle in radians and convert to degrees
    # angle_rad = np.arccos(cos_angle)
    # angle_deg = np.degrees(angle_rad)

    # return angle_deg


def approach_angle_variance(
    files: list[str] | str,
    antibody_chains: str | list[str] = ["A", "B"],
    antigen_chains: str | list[str] | None = None,
    interface_cutoff: float = 5.0,
    atom_types: str | list[str] = ["CA"],
    quiet: bool = True,
    log_dir: str | Path | None = None,
) -> float:
    """
    Calculate the variance in the angle of approach for multiple PDB/CIF files.

    Parameters
    ----------
    files : list[str] or str
        Either a list of PDB/CIF file paths or a directory path containing PDB/CIF files.

    antibody_chains : str or list[str], optional, default=["A", "B"]
        The chain ID(s) to consider as antibody. Can be a single chain or multiple chains.

    antigen_chains : str or list[str] or None, optional, default=None
        The chain ID(s) to consider as antigen. If not provided, the function will
        use find_antibody_bound_antigen_chain to identify the appropriate chain.

    interface_cutoff : float, optional, default=5.0
        The cutoff distance (in Ångstroms) for an antigen atom to be considered part of the interface.

    atom_types : str or list[str], optional, default=["CA"]
        The atom type(s) to use for center of mass calculations. Default is CA atoms only.

    quiet : bool, optional, default=True
        Suppress verbose output from the PDB parser.

    log_dir : str or Path or None, optional, default=None
        Directory path to save CSV log file with individual approach angle values.
        If not provided, no log file will be generated.

    Returns
    -------
    float
        The variance in approach angle across all files.

    Raises
    ------
    ValueError
        If less than two files are provided or found in the specified directory.

    FileNotFoundError
        If the specified directory or files do not exist.
    """
    # inputs
    if isinstance(antibody_chains, str):
        antibody_chains = [antibody_chains]
    if isinstance(antigen_chains, str) and antigen_chains is not None:
        antigen_chains = [antigen_chains]

    files = _get_pdb_and_cif_files(files)
    if len(files) < 2:
        raise ValueError(
            "At least two PDB/CIF files are required for approach angle variance calculation"
        )

    # logging
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    # calculate approach angle for each file
    angle_data = []
    angles = []

    for file_path in files:
        try:
            angle = approach_angle(
                file_path,
                antibody_chains,
                antigen_chains,
                interface_cutoff,
                atom_types,
                quiet,
            )
            angles.append(angle)
            angle_data.append({"filepath": str(file_path), "approach_angle": angle})
        except Exception as e:
            print(f"Error calculating approach angle for {file_path}: {e}")
            continue

    # angle of approach variance
    if not angles:
        variance = 0.0  # Return 0 if no valid angle calculations
    else:
        variance = np.var(angles)

    # save approach angle results and stats to CSV if log directory is provided
    if log_dir is not None and angle_data:
        df = pd.DataFrame(angle_data)
        stats_df = pd.DataFrame([{"metric": "variance", "value": variance}])
        # save results and stats
        csv_path = os.path.join(log_dir, "approach_angles.csv")
        df.to_csv(csv_path, index=False)
        stats_path = os.path.join(log_dir, "approach_angle_stats.csv")
        stats_df.to_csv(stats_path, index=False)

    return variance
