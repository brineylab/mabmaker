import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mabmaker.tools.ligandmpnn import (
    LigandMPNNParameters,
    _get_ligandmpnn_cmd,
    _get_model_checkpoint,
    _process_chain_or_residue_data,
    _process_per_residue_data,
    expand_residue_ranges,
)


def create_test_file(temp_dir, content):
    """Helper to create a temporary file with the given content"""
    fd, path = tempfile.mkstemp(dir=temp_dir, text=True)
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


class TestGetLigandMPNNCmd:
    """Tests for the _get_ligandmpnn_cmd function."""

    def test_minimal_parameters(self, minimal_params):
        """Test with minimal required parameters."""
        cmd = _get_ligandmpnn_cmd(minimal_params)

        assert "python" in cmd
        assert f'--pdb_path "{minimal_params.pdb_path}"' in cmd
        assert f"--model_type {minimal_params.model_type}" in cmd
        assert f'--out_folder "{minimal_params.output_dir}"' in cmd
        assert f"--seed {minimal_params.seed}" in cmd
        assert f"--temperature {minimal_params.temperature}" in cmd
        assert f"--batch_size {minimal_params.batch_size}" in cmd
        assert f"--number_of_batches {minimal_params.num_batches}" in cmd
        assert f'--checkpoint_ligand_mpnn "{minimal_params.model_checkpoint}"' in cmd
        assert "--ligand_mpnn_use_side_chain_context 1" in cmd
        assert "--ligand_mpnn_use_atom_context 0" in cmd
        assert "--save_stats 1" in cmd
        assert "--fixed_residues" not in cmd
        assert "--redesigned_residues" not in cmd
        assert "--chains_to_design" not in cmd
        assert "--parse_these_chains_only" not in cmd

    def test_all_parameters(self, full_params):
        """Test with all parameters provided."""
        cmd = _get_ligandmpnn_cmd(full_params)

        assert "python" in cmd
        assert f'--pdb_path "{full_params.pdb_path}"' in cmd
        assert f"--model_type {full_params.model_type}" in cmd
        assert f'--out_folder "{full_params.output_dir}"' in cmd
        assert f"--seed {full_params.seed}" in cmd
        assert f"--temperature {full_params.temperature}" in cmd
        assert f"--batch_size {full_params.batch_size}" in cmd
        assert f"--number_of_batches {full_params.num_batches}" in cmd
        assert f'--checkpoint_ligand_mpnn "{full_params.model_checkpoint}"' in cmd
        assert f'--fixed_residues "{full_params.fixed_residues}"' in cmd
        assert f'--redesigned_residues "{full_params.redesigned_residues}"' in cmd
        assert f'--chains_to_design "{full_params.chains_to_design}"' in cmd
        assert (
            f'--parse_these_chains_only "{full_params.parse_these_chains_only}"' in cmd
        )
        assert "--ligand_mpnn_use_side_chain_context 1" in cmd
        assert "--ligand_mpnn_use_atom_context 1" in cmd
        assert "--save_stats 0" in cmd

    def test_protein_mpnn_model_type(self, protein_mpnn_params):
        """Test with protein_mpnn model type."""
        cmd = _get_ligandmpnn_cmd(protein_mpnn_params)

        assert f"--model_type {protein_mpnn_params.model_type}" in cmd
        assert (
            f'--checkpoint_protein_mpnn "{protein_mpnn_params.model_checkpoint}"' in cmd
        )
        assert "--checkpoint_ligand_mpnn" not in cmd

    def test_boolean_conversion(self):
        """Test boolean conversion in command line arguments."""
        # Test True values
        params = LigandMPNNParameters(
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
            use_atom_context=True,
            batch_size=32,
            num_batches=1,
            save_stats=True,
            verbose=True,
        )
        cmd = _get_ligandmpnn_cmd(params)
        assert "--ligand_mpnn_use_side_chain_context 1" in cmd
        assert "--ligand_mpnn_use_atom_context 1" in cmd
        assert "--save_stats 1" in cmd

        # Test False values
        params = LigandMPNNParameters(
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
            use_side_chain_context=False,
            use_atom_context=False,
            batch_size=32,
            num_batches=1,
            save_stats=False,
            verbose=False,
        )
        cmd = _get_ligandmpnn_cmd(params)
        assert "--ligand_mpnn_use_side_chain_context 0" in cmd
        assert "--ligand_mpnn_use_atom_context 0" in cmd
        assert "--save_stats 0" in cmd


class TestProcessChainOrResidueData:
    """Tests for the _process_chain_or_residue_data function."""

    def test_none_input(self, sample_pdbs):
        """Test with None input."""
        result = _process_chain_or_residue_data(None, sample_pdbs)
        assert result == {}

    def test_string_input(self, sample_pdbs):
        """Test with string input."""
        data = "A,B,C"
        result = _process_chain_or_residue_data(data, sample_pdbs)
        expected = {pdb: data for pdb in sample_pdbs}
        assert result == expected

    def test_list_input(self, sample_pdbs):
        """Test with list input."""
        data = ["A", "B", "C"]
        result = _process_chain_or_residue_data(data, sample_pdbs, sep=" ")
        expected = {pdb: "A B C" for pdb in sample_pdbs}
        assert result == expected

    def test_with_different_separators(self, sample_pdbs):
        """Test with different separators."""
        data = ["A", "B", "C"]

        # Space separator
        result1 = _process_chain_or_residue_data(data, sample_pdbs, sep=" ")
        expected1 = {pdb: "A B C" for pdb in sample_pdbs}
        assert result1 == expected1

        # Comma separator
        result2 = _process_chain_or_residue_data(data, sample_pdbs, sep=",")
        expected2 = {pdb: "A,B,C" for pdb in sample_pdbs}
        assert result2 == expected2

        # Dash separator
        result3 = _process_chain_or_residue_data(data, sample_pdbs, sep="-")
        expected3 = {pdb: "A-B-C" for pdb in sample_pdbs}
        assert result3 == expected3

    def test_json_file_input(self, temp_json_file, sample_pdbs, mock_magika):
        """Test with JSON file input."""
        # Set up mock
        mock, mock_output = mock_magika
        mock_output.ct_label = "json"

        # Create JSON data
        data = {pdb: f"Chain_{pdb}" for pdb in sample_pdbs}
        with open(temp_json_file, "w") as f:
            json.dump(data, f)

        result = _process_chain_or_residue_data(temp_json_file, sample_pdbs)
        assert result == data

    def test_text_file_input(self, temp_text_file, sample_pdbs, mock_magika):
        """Test with text file input."""
        # Set up mock
        mock, mock_output = mock_magika
        mock_output.ct_label = "text"

        # Create text data
        chains = "A B C D"
        with open(temp_text_file, "w") as f:
            f.write(chains)

        result = _process_chain_or_residue_data(temp_text_file, sample_pdbs)
        expected = {pdb: chains for pdb in sample_pdbs}
        assert result == expected

    def test_with_empty_string(self, sample_pdbs):
        """Test with empty string."""
        result = _process_chain_or_residue_data("", sample_pdbs)
        expected = {pdb: "" for pdb in sample_pdbs}
        assert result == expected

    def test_with_empty_list(self, sample_pdbs):
        """Test with empty list."""
        result = _process_chain_or_residue_data([], sample_pdbs)
        expected = {pdb: "" for pdb in sample_pdbs}
        assert result == expected

    def test_with_invalid_input_type(self, sample_pdbs):
        """Test with invalid input type."""
        with pytest.raises(ValueError, match="Invalid chain or residue data"):
            _process_chain_or_residue_data(123, sample_pdbs)


class TestProcessPerResidueData:
    """Tests for the _process_per_residue_data function."""

    def test_none_input(self, sample_pdbs):
        """Test with None input."""
        result = _process_per_residue_data(None, sample_pdbs)
        assert result == {}

    def test_dict_input_with_file_paths(self, sample_pdbs):
        """Test with dictionary input containing file paths."""
        # Create a temporary file to test with file paths
        with tempfile.NamedTemporaryFile(mode="w+") as tmp:
            data = {tmp.name: {"A1": {"A": 0.5}}}

            result = _process_per_residue_data(data, sample_pdbs)
            assert result == data

    def test_dict_input_with_non_file_paths(self, sample_pdbs):
        """Test with dictionary input containing non-file paths."""
        data = {"A1": {"A": 0.5}}

        result = _process_per_residue_data(data, sample_pdbs)
        expected = {pdb: data for pdb in sample_pdbs}
        assert result == expected

    def test_json_file_input(self, temp_json_file, sample_pdbs):
        """Test with JSON file input."""
        data = {"A1": {"A": 0.5}}

        with open(temp_json_file, "w") as f:
            json.dump(data, f)

        result = _process_per_residue_data(temp_json_file, sample_pdbs)
        expected = {pdb: data for pdb in sample_pdbs}
        assert result == expected

    def test_nonexistent_file(self, sample_pdbs):
        """Test with nonexistent file path."""
        with pytest.raises(ValueError, match="does not exist"):
            _process_per_residue_data("/nonexistent/file.json", sample_pdbs)

    def test_with_mixed_dict_keys(self, sample_pdbs):
        """Test with dictionary containing both file and non-file keys."""
        with tempfile.NamedTemporaryFile(mode="w+") as tmp:
            data = {tmp.name: {"A1": {"A": 0.5}}, "residue_key": {"B1": {"B": 0.7}}}

            result = _process_per_residue_data(data, sample_pdbs)
            assert result == data

    def test_with_nested_data(self, sample_pdbs):
        """Test with deeply nested data."""
        data = {
            "A1": {"A": {"scores": [0.5, 0.6], "confidence": 0.9}},
            "B2": {"B": {"scores": [0.3, 0.4], "confidence": 0.8}},
        }

        result = _process_per_residue_data(data, sample_pdbs)
        expected = {pdb: data for pdb in sample_pdbs}
        assert result == expected


class TestExpandResidueRanges:
    """Tests for the expand_residue_ranges function."""

    def test_basic_range_from_file(self, temp_dir):
        """Test basic range expansion from file"""
        file_path = create_test_file(temp_dir, "A1-A5")
        result = expand_residue_ranges(file_path)
        assert result == "A1 A2 A3 A4 A5"

    def test_direct_string_input(self):
        """Test with direct string input instead of file"""
        result = expand_residue_ranges("A1-A5")
        assert result == "A1 A2 A3 A4 A5"

    def test_range_with_chain_on_first_only(self):
        """Test range with chain identifier only on first element"""
        result = expand_residue_ranges("A1-5")
        assert result == "A1 A2 A3 A4 A5"

    def test_mixed_range_formats(self):
        """Test mixed range formats in the same input"""
        result = expand_residue_ranges("A1-A5,B10-15")
        assert result == "A1 A2 A3 A4 A5 B10 B11 B12 B13 B14 B15"

    def test_single_residues(self):
        """Test single residues without ranges"""
        result = expand_residue_ranges("A1,B10,C15")
        assert result == "A1 B10 C15"

    def test_comma_separated_values(self):
        """Test comma-separated inputs"""
        result = expand_residue_ranges("A1-A3,B5,C7-C9")
        assert result == "A1 A2 A3 B5 C7 C8 C9"

    def test_whitespace_separated_values(self):
        """Test whitespace-separated inputs"""
        result = expand_residue_ranges("A1-A3 B5 C7-C9")
        assert result == "A1 A2 A3 B5 C7 C8 C9"

    def test_mixed_separators(self):
        """Test mixed separators (both commas and whitespace)"""
        result = expand_residue_ranges("A1-A3, B5 C7-C9")
        assert result == "A1 A2 A3 B5 C7 C8 C9"

    def test_multiple_whitespaces(self):
        """Test with multiple whitespaces between items"""
        result = expand_residue_ranges("A1-A3   B5  \t  C7-C9")
        assert result == "A1 A2 A3 B5 C7 C8 C9"

    def test_output_to_file(self, temp_dir):
        """Test writing output to a file"""
        input_path = create_test_file(temp_dir, "A1-A5,B10")
        output_path = os.path.join(temp_dir, "output.txt")

        expand_residue_ranges(input_path, output_path)

        with open(output_path, "r") as f:
            content = f.read().strip()

        assert content == "A1 A2 A3 A4 A5 B10"

    def test_empty_input(self, temp_dir):
        """Test with empty input"""
        file_path = create_test_file(temp_dir, "")
        result = expand_residue_ranges(file_path)
        assert result == ""

    def test_invalid_range_format(self):
        """Test with invalid range format"""
        result = expand_residue_ranges("A1--A5")
        assert result == ""  # Should skip invalid range

    def test_invalid_residue_format(self):
        """Test with invalid residue format"""
        result = expand_residue_ranges("A1,123,B5")
        assert result == "A1 B5"  # Should skip invalid residue

    def test_nonexistent_file(self):
        """Test with a non-existent file path"""
        result = expand_residue_ranges("/path/to/nonexistent/file.txt")
        assert result is None  # Should return None for file not found

    def test_reverse_range(self):
        """Test with end number less than start number"""
        result = expand_residue_ranges("A5-A1")
        assert result == ""  # A5-A1 is an invalid range as 5 > 1

    def test_complex_mixed_input(self):
        """Test with a complex mix of range formats and separators"""
        complex_input = "A1-5, B10-B15 C20,D25-30 E40"
        result = expand_residue_ranges(complex_input)
        expected = (
            "A1 A2 A3 A4 A5 B10 B11 B12 B13 B14 B15 C20 D25 D26 D27 D28 D29 D30 E40"
        )
        assert result == expected

    @pytest.mark.parametrize(
        "input_str, expected",
        [
            ("A1-3", "A1 A2 A3"),
            ("B5-B7", "B5 B6 B7"),
            ("C10", "C10"),
            ("D1-5,E7", "D1 D2 D3 D4 D5 E7"),
            ("F1 F2 F3", "F1 F2 F3"),
            ("G1-3 H4-6", "G1 G2 G3 H4 H5 H6"),
        ],
    )
    def test_parametrized_inputs(self, input_str, expected):
        """Test various inputs with parametrization"""
        result = expand_residue_ranges(input_str)
        assert result == expected

    @pytest.mark.parametrize(
        "invalid_input",
        [
            "123-456",  # No chain identifier
            "A-5",  # No start number
            "A1-",  # No end number
            "AB1-5",  # Invalid chain format
            "1A-5A",  # Reversed chain and number
        ],
    )
    def test_invalid_inputs(self, invalid_input):
        """Test various invalid inputs with parametrization"""
        result = expand_residue_ranges(invalid_input)
        assert result == ""  # All should be skipped


@patch("os.path.join")
@patch("os.path.isfile")
class TestGetModelCheckpoint:
    """Tests for the _get_model_checkpoint function."""

    def test_ligand_mpnn_model_type(self, mock_isfile, mock_join):
        """Test with ligand_mpnn model type."""
        mock_isfile.return_value = True
        mock_join.side_effect = lambda *args: "/".join(args)

        result = _get_model_checkpoint("ligand_mpnn", "v_32_010_25")

        assert "ligandmpnn_v_32_010_25.pt" in result
        assert "model_params" in result

        # Verify correct paths are joined
        mock_join.assert_any_call(mock_join.return_value, "model_params")
        mock_join.assert_any_call(mock_join.return_value, "ligandmpnn_v_32_010_25.pt")

    def test_protein_mpnn_model_type(self, mock_isfile, mock_join):
        """Test with protein_mpnn model type."""
        mock_isfile.return_value = True
        mock_join.side_effect = lambda *args: "/".join(args)

        result = _get_model_checkpoint("protein_mpnn", "v_48_020")

        assert "proteinmpnn_v_48_020.pt" in result
        assert "model_params" in result

        # Verify correct paths are joined
        mock_join.assert_any_call(mock_join.return_value, "model_params")
        mock_join.assert_any_call(mock_join.return_value, "proteinmpnn_v_48_020.pt")

    def test_underscore_handling(self, mock_isfile, mock_join):
        """Test handling of underscores in model type."""
        mock_isfile.return_value = True
        mock_join.side_effect = lambda *args: "/".join(args)

        result1 = _get_model_checkpoint("ligand_mpnn", "test")
        result2 = _get_model_checkpoint("ligandmpnn", "test")

        assert result1 == result2
        assert "ligandmpnn_test.pt" in result1

    def test_lowercase_conversion(self, mock_isfile, mock_join):
        """Test conversion to lowercase."""
        mock_isfile.return_value = True
        mock_join.side_effect = lambda *args: "/".join(args)

        result1 = _get_model_checkpoint("LIGAND_MPNN", "TEST")
        result2 = _get_model_checkpoint("ligand_mpnn", "test")

        assert result1 == result2
        assert "ligandmpnn_test.pt" in result1

    def test_special_characters_in_checkpoint(self, mock_isfile, mock_join):
        """Test with special characters in checkpoint name."""
        mock_isfile.return_value = True
        mock_join.side_effect = lambda *args: "/".join(args)

        result = _get_model_checkpoint("ligand_mpnn", "special-chars_v1.2.3")

        assert "ligandmpnn_special-chars_v1.2.3.pt" in result
