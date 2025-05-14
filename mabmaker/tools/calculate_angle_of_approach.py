import os
import json
import math
import logging
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from Bio.PDB import PDBParser, NeighborSearch, Selection
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
BASE_DIR = Path("./preds/af3/04_nprb_ZM215")
PDB_DIR = BASE_DIR / "pdbs"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = BASE_DIR / "cache"
LOG_DIR = BASE_DIR / "logs"
CACHE_FILE = CACHE_DIR / "angle_cache.json"
CSV_FILE = OUTPUT_DIR / "angles.csv"
DISTANCE_CUTOFF = 8.0  # in angstroms
N_WORKERS = os.cpu_count() or 4

# -------------------------
# Logging Setup
# -------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / "angle_of_approach.log"
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_cache() -> dict:
    if CACHE_FILE.exists():
        with CACHE_FILE.open("r") as f:
            logger.info("Loading existing cache.")
            return json.load(f)
    logger.info("No cache found, starting fresh.")
    return {}


def save_cache(cache: dict):
    with CACHE_FILE.open("w") as f:
        json.dump(cache, f, indent=4)


def write_csv(cache: dict, force_overwrite: bool = False):
    mode = "w" if force_overwrite else "a"
    file_exists = CSV_FILE.exists()

    with CSV_FILE.open(mode, newline="") as csvfile:
        writer = csv.writer(csvfile)
        if force_overwrite or not file_exists:
            writer.writerow(
                ["pdb_filename", "angle_in_degrees"]
            )  # Write header if needed
        for pdb_name, angle in sorted(cache.items()):
            writer.writerow([pdb_name, f"{angle:.2f}"])
    logger.info(
        f"CSV file {CSV_FILE} {'overwritten' if force_overwrite else 'updated'} with {len(cache)} entries."
    )


def compute_centroid(residues: List) -> np.ndarray:
    atoms = [
        atom.get_coord() for res in residues for atom in res if atom.element != "H"
    ]
    if not atoms:
        raise ValueError("No heavy atoms found for centroid calculation.")
    return np.mean(atoms, axis=0)


def compute_angle(vec1: np.ndarray, vec2: np.ndarray) -> float:
    unit_vec1 = vec1 / np.linalg.norm(vec1)
    unit_vec2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.clip(np.dot(unit_vec1, unit_vec2), -1.0, 1.0)
    angle_rad = math.acos(dot_product)
    return math.degrees(angle_rad)


def find_interface_residues(chain_a, chains_bc, cutoff=DISTANCE_CUTOFF) -> List:
    all_atoms = Selection.unfold_entities(chains_bc, "A")
    search = NeighborSearch(all_atoms)

    interface_residues = set()
    for res in chain_a.get_residues():
        if not res.has_id("CA"):  # skip weird residues
            continue
        ca_atom = res["CA"]
        neighbors = search.search(ca_atom.coord, cutoff)
        if neighbors:
            interface_residues.add(res)
    return list(interface_residues)


def process_pdb(pdb_path: Path) -> Optional[Tuple[str, float]]:
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_path.stem, pdb_path)
        model = structure[0]

        chain_a = model["A"]
        chain_b = model["B"]
        # Identify antigen chains (assume all chains after B are antigen chains)
        antigen_chains = [
            chain
            for chain_id, chain in model.child_dict.items()
            if chain_id not in ["A", "B"]
        ]
        if not antigen_chains:
            logger.warning(f"No antigen chains found in {pdb_path.name}. Skipping.")
            return None

        # Find interface residues
        interface_residues = find_interface_residues(
            chain_a, antigen_chains, cutoff=DISTANCE_CUTOFF
        )
        if not interface_residues:
            logger.warning(f"No interface residues found in {pdb_path.name}. Skipping.")
            return None

        # Compute centroids
        heavy_centroid = compute_centroid(list(chain_a.get_residues()))
        light_centroid = compute_centroid(list(chain_b.get_residues()))
        antigen_centroid = compute_centroid(interface_residues)

        # Compute vectors and angle
        antibody_vector = heavy_centroid - light_centroid
        antigen_vector = antigen_centroid - light_centroid

        angle = compute_angle(antibody_vector, antigen_vector)
        logger.info(f"Processed {pdb_path.name}: Angle = {angle:.2f} degrees")
        return pdb_path.name, angle

    except Exception as e:
        logger.error(f"Failed to process {pdb_path.name}: {str(e)}", exc_info=True)
        return None


def main():
    cache = load_cache()
    processed_cache = {}  # Only store new entries to write to CSV

    pdb_files = [f for f in PDB_DIR.glob("*.pdb") if f.name not in cache]

    if not pdb_files:
        logger.info("No new PDB files to process.")
    else:
        logger.info(
            f"Starting processing of {len(pdb_files)} PDB files using {N_WORKERS} workers."
        )

        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            future_to_file = {
                executor.submit(process_pdb, pdb): pdb for pdb in pdb_files
            }

            for future in as_completed(future_to_file):
                result = future.result()
                if result:
                    filename, angle = result
                    cache[filename] = angle
                    processed_cache[filename] = angle

        save_cache(cache)

    # Write only newly processed entries to CSV
    write_csv(processed_cache, force_overwrite=False)
    logger.info("Processing complete. Cache and CSV saved.")


if __name__ == "__main__":
    main()
