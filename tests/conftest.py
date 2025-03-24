import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mabmaker.tools.ligandmpnn import LigandMPNNParameters


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after tests
    for filename in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, filename))
    os.rmdir(temp_dir)


def create_test_file(temp_dir, content):
    """Helper to create a temporary file with the given content"""
    fd, path = tempfile.mkstemp(dir=temp_dir, text=True)
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


@pytest.fixture
def minimal_params():
    """Fixture that returns minimal LigandMPNNParameters."""
    return LigandMPNNParameters(
        pdb_path="/path/to/structure.pdb",
        output_dir="/path/to/output",
        model_type="ligand_mpnn",
        model_checkpoint="/path/to/checkpoint.pt",
        seed=42,
        temperature=0.1,
        fixed_residues=None,
        redesigned_residues=None,
        bias_aa=None,
        bias_aa_per_residue_dict=None,
        omit_aa=None,
        omit_aa_per_residue_dict=None,
        chains_to_design=None,
        parse_these_chains_only=None,
        use_side_chain_context=True,
        use_atom_context=False,
        batch_size=32,
        num_batches=1,
        save_stats=True,
        verbose=True,
    )


@pytest.fixture
def full_params():
    """Fixture that returns LigandMPNNParameters with all fields populated."""
    return LigandMPNNParameters(
        pdb_path="/path/to/structure.pdb",
        output_dir="/path/to/output",
        model_type="ligand_mpnn",
        model_checkpoint="/path/to/checkpoint.pt",
        seed=42,
        temperature=0.1,
        fixed_residues="A1 A2 A3",
        redesigned_residues="B1 B2 B3",
        bias_aa="A:0.5,C:-1.0",
        bias_aa_per_residue_dict={"A1": {"A": 0.5, "C": -1.0}},
        omit_aa="DE",
        omit_aa_per_residue_dict={"A1": "GF"},
        chains_to_design="A,B",
        parse_these_chains_only="A,B,C",
        use_side_chain_context=True,
        use_atom_context=True,
        batch_size=64,
        num_batches=2,
        save_stats=False,
        verbose=False,
    )


@pytest.fixture
def protein_mpnn_params():
    """Fixture that returns LigandMPNNParameters with protein_mpnn model type."""
    return LigandMPNNParameters(
        pdb_path="/path/to/structure.pdb",
        output_dir="/path/to/output",
        model_type="protein_mpnn",
        model_checkpoint="/path/to/checkpoint.pt",
        seed=42,
        temperature=0.1,
        fixed_residues=None,
        redesigned_residues=None,
        bias_aa=None,
        bias_aa_per_residue_dict=None,
        omit_aa=None,
        omit_aa_per_residue_dict=None,
        chains_to_design=None,
        parse_these_chains_only=None,
        use_side_chain_context=True,
        use_atom_context=False,
        batch_size=32,
        num_batches=1,
        save_stats=True,
        verbose=True,
    )


@pytest.fixture
def temp_json_file():
    """Fixture that creates a temporary JSON file."""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp:
        yield tmp.name
        # Clean up after the test
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


@pytest.fixture
def temp_text_file():
    """Fixture that creates a temporary text file."""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp:
        yield tmp.name
        # Clean up after the test
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


@pytest.fixture
def mock_magika():
    """Fixture that mocks the Magika class."""
    with patch("mabmaker.tools.ligandmpnn.Magika") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        mock_output = MagicMock()
        mock_instance.identify_path.return_value = mock_output
        yield mock, mock_output


@pytest.fixture
def sample_pdbs():
    """Fixture that returns a list of sample PDB names."""
    return ["pdb1", "pdb2", "pdb3"]
