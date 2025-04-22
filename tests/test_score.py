import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from Bio.PDB import PDBIO
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

from mabmaker.tools.score import rmsd, ssRMSD


@pytest.fixture
def temp_pdb_dir():
    """Create a temporary directory for PDB test files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def create_simple_pdb_structure():
    """Create a simple PDB structure with one chain and a few CA atoms"""

    def _create_structure(model_id=0, chain_id="A", n_atoms=5, offset_x=0.0):
        # Create a structure, model, and chain
        structure = Structure("test")
        model = Model(model_id)
        structure.add(model)
        chain = Chain(chain_id)
        model.add(chain)

        # Add residues with CA atoms in a line along the x-axis
        for i in range(1, n_atoms + 1):
            residue = Residue((" ", i, " "), "ALA", "")
            chain.add(residue)
            # CA atom with coordinates
            atom = Atom(
                "CA",
                [i + offset_x, 0.0, 0.0],
                0.0,  # B-factor
                1.0,  # occupancy
                " ",  # altloc
                "CA",  # fullname
                i,  # serial number
                "C",  # element
            )
            residue.add(atom)

        return structure

    return _create_structure


@pytest.fixture
def sample_pdb_files(temp_pdb_dir, create_simple_pdb_structure):
    """Create sample PDB files with known structures for testing"""
    file_paths = []

    # Create 4 PDB files with slight variations
    offsets = [0.0, 0.1, 0.2, 0.5]

    for i, offset in enumerate(offsets):
        # Create structure with offset
        structure = create_simple_pdb_structure(offset_x=offset)

        # Save to PDB file
        file_path = os.path.join(temp_pdb_dir, f"test_{i}.pdb")
        io = PDBIO()
        io.set_structure(structure)
        io.save(file_path)
        file_paths.append(file_path)

    return file_paths


@pytest.fixture
def multi_chain_pdb_files(temp_pdb_dir, create_simple_pdb_structure):
    """Create sample PDB files with multiple chains for testing"""
    file_paths = []

    # Create 3 PDB files with multiple chains and variations
    for i in range(3):
        # Create a structure with multiple chains
        structure = Structure(f"test_{i}")
        model = Model(0)
        structure.add(model)

        # Add chains A and B with different offsets
        chain_offsets = {"A": 0.1 * i, "B": 0.2 * i, "C": 0.3 * i}

        for chain_id, offset in chain_offsets.items():
            chain = Chain(chain_id)
            model.add(chain)

            # Add residues with CA atoms
            for j in range(1, 6):
                residue = Residue((" ", j, " "), "ALA", "")
                chain.add(residue)
                atom = Atom("CA", [j + offset, 0.0, 0.0], 0.0, 1.0, " ", "CA", j, "C")
                residue.add(atom)

        # Save to PDB file
        file_path = os.path.join(temp_pdb_dir, f"multi_chain_{i}.pdb")
        io = PDBIO()
        io.set_structure(structure)
        io.save(file_path)
        file_paths.append(file_path)

    return file_paths


class TestRMSD:
    """Tests for the rmsd function."""

    def test_single_chain_rmsd(self, sample_pdb_files):
        """Test RMSD calculation between two structures with a single chain."""
        file_paths = sample_pdb_files

        # Test RMSD between first two files
        calculated_rmsd = rmsd(file_paths[0], file_paths[1], "A")

        # RMSD should be a non-negative number
        assert calculated_rmsd >= 0
        # For our test structures, RMSD should be small since they're nearly identical
        assert calculated_rmsd < 1.0

    def test_single_chain_as_list(self, sample_pdb_files):
        """Test RMSD calculation with a single chain provided as a list."""
        file_paths = sample_pdb_files

        calculated_rmsd = rmsd(file_paths[0], file_paths[1], ["A"])

        # RMSD should be a non-negative number
        assert calculated_rmsd >= 0
        # For our test structures, RMSD should be small
        assert calculated_rmsd < 1.0

    def test_specific_atom_types(self, sample_pdb_files):
        """Test RMSD calculation with specific atom types."""
        file_paths = sample_pdb_files

        # Test with CA atoms explicitly
        calculated_rmsd = rmsd(file_paths[0], file_paths[1], "A", atom_types=["CA"])

        # RMSD should be a non-negative number
        assert calculated_rmsd >= 0
        # For our test structures, RMSD should be small
        assert calculated_rmsd < 1.0

    def test_multi_chain_rmsd(self, multi_chain_pdb_files):
        """Test RMSD calculation with multiple chains."""
        file_paths = multi_chain_pdb_files

        # Calculate RMSD for chain A
        rmsd_a = rmsd(file_paths[0], file_paths[1], "A")
        # Calculate RMSD for chain B
        rmsd_b = rmsd(file_paths[0], file_paths[1], "B")
        # Calculate RMSD for both chains together
        rmsd_ab = rmsd(file_paths[0], file_paths[1], ["A", "B"])

        # RMSD for both chains should be different than for individual chains
        assert rmsd_ab != rmsd_a
        assert rmsd_ab != rmsd_b

    def test_atom_type_as_string(self, sample_pdb_files):
        """Test RMSD calculation with atom type provided as a string."""
        file_paths = sample_pdb_files

        calculated_rmsd = rmsd(file_paths[0], file_paths[1], "A", atom_types="CA")

        # RMSD should be a non-negative number
        assert calculated_rmsd >= 0
        # For our test structures, RMSD should be small
        assert calculated_rmsd < 1.0

    def test_nonexistent_chain(self, sample_pdb_files):
        """Test RMSD calculation with a non-existent chain."""
        file_paths = sample_pdb_files

        with pytest.raises(ValueError, match="Chain Z not found in both structures"):
            rmsd(file_paths[0], file_paths[1], "Z")

    def test_path_str_and_path_object(self, sample_pdb_files):
        """Test RMSD calculation with one path as string and one as Path object."""
        file_paths = sample_pdb_files

        # Use string for first path and Path object for second
        calculated_rmsd = rmsd(file_paths[0], Path(file_paths[1]), "A")

        # RMSD should be a non-negative number
        assert calculated_rmsd >= 0
        # For our test structures, RMSD should be small
        assert calculated_rmsd < 1.0


class TestSSRMSD:
    """Tests for the ssRMSD function."""

    def test_list_of_files(self, sample_pdb_files):
        """Test ssRMSD calculation with a list of files."""
        file_paths = sample_pdb_files

        ss_rmsd = ssRMSD(file_paths, "A")

        # Verify sum of squared RMSD values is positive
        assert ss_rmsd > 0
        # For our test structures, the RMSD values are small
        assert ss_rmsd < len(file_paths) * (len(file_paths) - 1) / 2

    def test_directory_path(self, sample_pdb_files, temp_pdb_dir):
        """Test ssRMSD calculation with a directory path."""
        file_paths = sample_pdb_files

        ss_rmsd = ssRMSD(temp_pdb_dir, "A")

        # Verify sum of squared RMSD is calculated
        assert ss_rmsd > 0

    def test_multiple_chains(self, multi_chain_pdb_files):
        """Test ssRMSD calculation with multiple chains."""
        file_paths = multi_chain_pdb_files

        # Calculate ssRMSD for different chain combinations
        ss_rmsd_a = ssRMSD(file_paths, "A")
        ss_rmsd_b = ssRMSD(file_paths, "B")
        ss_rmsd_ab = ssRMSD(file_paths, ["A", "B"])

        # Results should be different for different chains
        assert ss_rmsd_a != ss_rmsd_b
        assert ss_rmsd_ab != ss_rmsd_a
        assert ss_rmsd_ab != ss_rmsd_b

    def test_with_different_atom_types(self, sample_pdb_files):
        """Test ssRMSD calculation with different atom types."""
        file_paths = sample_pdb_files

        # Should work the same with CA atoms specified
        ss_rmsd = ssRMSD(file_paths, "A", atom_types=["CA"])
        assert ss_rmsd > 0

    def test_error_handling_for_missing_files(self, temp_pdb_dir):
        """Test error handling when files are missing."""
        with pytest.raises(FileNotFoundError, match="Directory does not exist"):
            ssRMSD("/nonexistent/directory", "A")

        with pytest.raises(ValueError, match="At least two PDB/CIF files are required"):
            # Create an empty directory
            empty_dir = os.path.join(temp_pdb_dir, "empty")
            os.makedirs(empty_dir, exist_ok=True)
            ssRMSD(empty_dir, "A")

    def test_single_file_error(self, sample_pdb_files):
        """Test error handling when only one file is provided."""
        file_paths = sample_pdb_files

        with pytest.raises(ValueError, match="At least two PDB/CIF files are required"):
            ssRMSD([file_paths[0]], "A")

    def test_chain_as_string(self, sample_pdb_files):
        """Test ssRMSD calculation with chain provided as a string."""
        file_paths = sample_pdb_files

        ss_rmsd = ssRMSD(file_paths, "A")
        assert ss_rmsd > 0

    def test_with_log_directory(self, sample_pdb_files, temp_pdb_dir):
        """Test ssRMSD calculation with log directory provided."""
        file_paths = sample_pdb_files

        # Create a log directory
        log_dir = os.path.join(temp_pdb_dir, "rmsd_logs")

        # Call ssRMSD with log_dir
        ss_rmsd = ssRMSD(file_paths, "A", log_dir=log_dir)

        # Verify sum of squared RMSD values is positive
        assert ss_rmsd > 0

        # Verify log file was created
        expected_csv_path = os.path.join(log_dir, "rmsd_values.csv")
        assert os.path.exists(expected_csv_path)

        # Verify CSV file content
        df = pd.read_csv(expected_csv_path)

        # Check column names
        assert list(df.columns) == ["filepath_1", "filepath_2", "rmsd"]

        # Check number of entries (should be n choose 2)
        n = len(file_paths)
        expected_pairs = n * (n - 1) // 2
        assert len(df) == expected_pairs

        # Check values are valid
        assert all(df["rmsd"] >= 0)
        assert all(df["rmsd"] < 1.0)  # For our test structures, RMSD should be small

    def test_with_multiple_chains_log(self, multi_chain_pdb_files, temp_pdb_dir):
        """Test ssRMSD calculation with multiple chains and log directory."""
        file_paths = multi_chain_pdb_files
        log_dir = os.path.join(temp_pdb_dir, "multi_chain_logs")

        # Call ssRMSD with multiple chains and log_dir
        ss_rmsd = ssRMSD(file_paths, ["A", "B"], log_dir=log_dir)

        # Verify log file was created with correct name
        expected_csv_path = os.path.join(log_dir, "rmsd_values.csv")
        assert os.path.exists(expected_csv_path)
