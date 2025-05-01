import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from mabmaker.tools.msa import (
    hash_sequence,
    merge_multi_a3m_to_aligned_dataframe,
    msa,
    precompute_boltz_msas,
    process_a3ms_for_chai,
    retrieve_msa_from_cache,
    run_mmseqs2,
    save_msa,
)
from mabmaker.utils.inputs import ProteinChain, StructurePredictionRun


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_sequence():
    """Return a sample protein sequence for testing"""
    return "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA"


@pytest.fixture
def mock_msa_response():
    """Return a mock MSA response for testing"""
    return ">101\nMDVFMKGLSKAKEGVVAAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA\n>UniRef90_P62081\nMDVFMKGLSKAKEGVVAAA-EKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA\n>UniRef90_P63313\nMDVFMKGLSKAKEGVVAAA-EKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA\n"


@pytest.fixture
def mock_a3m_file(temp_dir, mock_msa_response):
    """Create a mock a3m file"""
    a3m_path = os.path.join(temp_dir, "test.a3m")
    with open(a3m_path, "w") as f:
        f.write(mock_msa_response)
    return a3m_path


@pytest.fixture
def mock_run():
    """Create a mock StructurePredictionRun object"""
    # Create a dictionary representing a protein chain
    chain1_dict = {
        "sequence": "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTK",
        "chainId": "A",
    }
    chain2_dict = {
        "sequence": "EQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA",
        "chainId": "B",
    }

    # Create the run dictionary
    run_dict = {
        "name": "test_run",
        "modelSeeds": [42],
        "sequences": [
            {"proteinChain": chain1_dict},
            {"proteinChain": chain2_dict},
        ],
    }

    run = StructurePredictionRun(run_dict)
    return run


@pytest.fixture
def mock_mmseqs2_response():
    """Returns mock responses for mmseqs2 API calls"""

    class MockResponse:
        def __init__(self, json_data, content=None, status_code=200):
            self.json_data = json_data
            self.content = content
            self.status_code = status_code
            self.text = json.dumps(json_data)
            self.raw = MagicMock()

        def json(self):
            return self.json_data

    submit_response = MockResponse({"status": "RUNNING", "id": "mock_id"})
    status_response_running = MockResponse({"status": "RUNNING"})
    status_response_complete = MockResponse({"status": "COMPLETE"})

    # Create a mock tar.gz file for the response
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz") as tmp_file:
        with tarfile.open(fileobj=tmp_file, mode="w:gz") as tar:
            # Add a mock a3m file to the tar
            with tempfile.NamedTemporaryFile(suffix=".a3m", delete=False) as a3m_file:
                a3m_file.write(b">101\nMDVF\n>UniRef90_ABC123\nMDVF\n")
                a3m_file.flush()
                tar.add(a3m_file.name, arcname="uniref.a3m")
                tar.add(a3m_file.name, arcname="bfd.mgnify30.metaeuk30.smag30.a3m")
                os.unlink(a3m_file.name)

    download_response = MockResponse({}, status_code=200)
    download_response.content = open(tmp_file.name, "rb").read()

    os.unlink(tmp_file.name)

    return {
        "submit": submit_response,
        "status_running": status_response_running,
        "status_complete": status_response_complete,
        "download": download_response,
    }


class TestHashSequence:
    """Tests for the hash_sequence function"""

    def test_hash_sequence(self, sample_sequence):
        """Test that hash_sequence returns a valid SHA-256 hash"""
        hash_value = hash_sequence(sample_sequence)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA-256 hashes are 64 hex characters

    def test_hash_sequence_case_insensitive(self, sample_sequence):
        """Test that hash_sequence is case-insensitive"""
        hash_upper = hash_sequence(sample_sequence.upper())
        hash_lower = hash_sequence(sample_sequence.lower())
        assert hash_upper == hash_lower

    def test_hash_sequence_different_inputs(self):
        """Test that different sequences produce different hashes"""
        seq1 = "ACDEFGHIKL"
        seq2 = "ACDEFGHIKM"
        hash1 = hash_sequence(seq1)
        hash2 = hash_sequence(seq2)
        assert hash1 != hash2


class TestSaveMSA:
    """Tests for the save_msa function"""

    # @patch("abutils.io.parse_fasta")
    def test_save_msa(self, temp_dir, mock_msa_response):
        """Test that save_msa properly saves an MSA to a file"""
        # from abutils.core.sequence import Sequence

        # # Mock the response from parse_fasta
        # mock_sequence = MagicMock(spec=Sequence)
        query_sequence = mock_msa_response.split("\n")[1]
        # mock_parse_fasta.return_value = [mock_sequence]

        msa_path = save_msa(mock_msa_response, temp_dir)

        # Verify the file was created
        assert os.path.exists(msa_path)

        # Verify the file contains the expected content
        with open(msa_path, "r") as f:
            content = f.read()
        assert content == mock_msa_response

        # Verify the filename is a hash of the query sequence
        expected_hash = hash_sequence(query_sequence)
        expected_filename = f"{expected_hash}.a3m"
        assert os.path.basename(msa_path) == expected_filename


class TestRetrieveMSAFromCache:
    """Tests for the retrieve_msa_from_cache function"""

    def test_retrieve_msa_not_in_cache(self, temp_dir, sample_sequence):
        """Test retrieving an MSA that is not in the cache"""
        # Use a fresh temp directory as cache
        result = retrieve_msa_from_cache(sample_sequence, temp_dir)
        assert result is None

    def test_retrieve_msa_from_cache(
        self, temp_dir, sample_sequence, mock_msa_response
    ):
        """Test retrieving an MSA that is in the cache"""
        # Create a cached MSA file
        seq_hash = hash_sequence(sample_sequence)
        cache_path = os.path.join(temp_dir, f"{seq_hash}.a3m")
        with open(cache_path, "w") as f:
            f.write(mock_msa_response)

        # Retrieve from cache
        result = retrieve_msa_from_cache(sample_sequence, temp_dir)
        assert result == mock_msa_response


class TestMSA:
    """Tests for the msa function"""

    @patch("mabmaker.tools.msa.run_mmseqs2")
    @patch("mabmaker.tools.msa.retrieve_msa_from_cache")
    @patch("mabmaker.tools.msa.save_msa")
    def test_msa_single_sequence_not_in_cache(
        self,
        mock_save_msa,
        mock_retrieve_cache,
        mock_run_mmseqs2,
        sample_sequence,
        temp_dir,
        mock_msa_response,
    ):
        """Test msa function with a single sequence not in cache"""
        # Setup mocks
        mock_retrieve_cache.return_value = None
        mock_run_mmseqs2.return_value = mock_msa_response
        mock_save_msa.return_value = os.path.join(temp_dir, "result.a3m")

        # Run the function
        result = msa(sample_sequence, temp_dir, use_msa_cache=True)

        # Verify results
        assert result == os.path.join(temp_dir, "result.a3m")
        mock_retrieve_cache.assert_called_once()
        mock_run_mmseqs2.assert_called_once()
        assert mock_save_msa.call_count == 2  # Once for cache, once for output

    @patch("mabmaker.tools.msa.run_mmseqs2")
    @patch("mabmaker.tools.msa.retrieve_msa_from_cache")
    @patch("mabmaker.tools.msa.save_msa")
    def test_msa_single_sequence_in_cache(
        self,
        mock_save_msa,
        mock_retrieve_cache,
        mock_run_mmseqs2,
        sample_sequence,
        temp_dir,
        mock_msa_response,
    ):
        """Test msa function with a single sequence in cache"""
        # Setup mocks
        cache_path = os.path.join(temp_dir, "cached.a3m")
        mock_retrieve_cache.return_value = cache_path

        # Run the function
        with patch(
            "shutil.copy", return_value=os.path.join(temp_dir, "copied.a3m")
        ) as mock_copy:
            result = msa(sample_sequence, temp_dir, use_msa_cache=True)

        # Verify results
        assert result == os.path.join(temp_dir, "copied.a3m")
        mock_retrieve_cache.assert_called_once()
        mock_run_mmseqs2.assert_not_called()
        mock_save_msa.assert_not_called()
        mock_copy.assert_called_once_with(cache_path, temp_dir)

    @patch("mabmaker.tools.msa.run_mmseqs2")
    @patch("mabmaker.tools.msa.retrieve_msa_from_cache")
    @patch("mabmaker.tools.msa.save_msa")
    def test_msa_multiple_sequences(
        self,
        mock_save_msa,
        mock_retrieve_cache,
        mock_run_mmseqs2,
        temp_dir,
        mock_msa_response,
    ):
        """Test msa function with multiple sequences"""
        sequences = ["MDVF", "KGLSK", "AKEGV"]

        # Setup mocks
        mock_retrieve_cache.return_value = None
        mock_run_mmseqs2.return_value = mock_msa_response

        # Create enough side effects for all calls (2 calls per sequence)
        mock_save_msa.side_effect = [
            os.path.join(temp_dir, f"result{i}.a3m") for i in range(1, 7)
        ]

        # Run the function
        result = msa(sequences, temp_dir, use_msa_cache=True)

        # Verify results
        assert isinstance(result, list)
        assert len(result) == len(sequences)
        assert mock_retrieve_cache.call_count == len(sequences)
        assert mock_run_mmseqs2.call_count == len(sequences)
        assert (
            mock_save_msa.call_count == len(sequences) * 2
        )  # Once for cache, once for output

    @patch("mabmaker.tools.msa.run_mmseqs2")
    def test_msa_no_cache(
        self, mock_run_mmseqs2, sample_sequence, temp_dir, mock_msa_response
    ):
        """Test msa function with caching disabled"""
        # Setup mocks
        mock_run_mmseqs2.return_value = mock_msa_response

        with patch(
            "mabmaker.tools.msa.save_msa",
            return_value=os.path.join(temp_dir, "result.a3m"),
        ) as mock_save_msa:
            # Run the function
            result = msa(sample_sequence, temp_dir, use_msa_cache=False)

        # Verify results
        assert result == os.path.join(temp_dir, "result.a3m")
        mock_run_mmseqs2.assert_called_once()
        mock_save_msa.assert_called_once()  # Only called for output, not for cache


class TestPrecomputeBoltzMSAs:
    """Tests for the precompute_boltz_msas function"""

    @patch("mabmaker.tools.msa.msa")
    def test_precompute_boltz_msas(self, mock_msa, mock_run, temp_dir):
        """Test precomputing MSAs for Boltz"""
        # Setup mocks
        mock_msa.return_value = [
            os.path.join(temp_dir, "msa1.a3m"),
            os.path.join(temp_dir, "msa2.a3m"),
        ]

        # Create output directory structure
        output_path = os.path.join(temp_dir, "output")

        # Run the function
        result = precompute_boltz_msas([mock_run], output_path)

        # Verify results
        assert len(result) == 1
        assert result[0].name == mock_run.name
        assert result[0].protein_chains[0].msa == os.path.join(temp_dir, "msa1.a3m")
        assert result[0].protein_chains[1].msa == os.path.join(temp_dir, "msa2.a3m")

        # Verify the MSA directory was created
        expected_dir = os.path.join(
            output_path, mock_run.name, "msas", "precomputed", "a3m"
        )
        assert os.path.exists(expected_dir)

        # Verify msa was called with correct parameters
        mock_msa.assert_called_once()
        call_args = mock_msa.call_args[1]
        assert call_args["sequences"] == [
            chain.sequence for chain in mock_run.protein_chains
        ]
        assert call_args["output_dir"] == expected_dir
