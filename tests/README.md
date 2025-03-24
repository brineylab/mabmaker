# LigandMPNN Test Suite

This test suite provides comprehensive testing for the following functions in the LigandMPNN module:

1. `_get_ligandmpnn_cmd` - Constructs command line arguments for running LigandMPNN
2. `_process_chain_or_residue_data` - Processes chain or residue data into a standardized format
3. `_process_per_residue_data` - Processes per-residue data into a standardized format
4. `_get_model_checkpoint` - Gets the path to a model checkpoint based on model type and checkpoint name
5. `expand_residue_ranges` - Expands residue range notations into individual residue identifiers

## Requirements

- Python 3.8+
- pytest
- unittest.mock (part of the Python standard library)
- tempfile (part of the Python standard library)

## Running the Tests

To run all tests:

```bash
pytest tests/ -v
```

To run tests for a specific function:

```bash
pytest tests/test_ligandmpnn_utils.py::TestGetLigandMPNNCmd -v
pytest tests/test_ligandmpnn_utils.py::TestProcessChainOrResidueData -v
pytest tests/test_ligandmpnn_utils.py::TestProcessPerResidueData -v
pytest tests/test_ligandmpnn_utils.py::TestGetModelCheckpoint -v
pytest tests/test_ligandmpnn_utils.py::TestExpandResidueRanges -v
```

## Test Structure

The test suite is organized into five test classes in `test_ligandmpnn_utils.py`:

1. `TestGetLigandMPNNCmd` - Tests for `_get_ligandmpnn_cmd`
   - Tests for minimal parameters
   - Tests for all parameters
   - Tests for different model types
   - Tests for boolean conversion

2. `TestProcessChainOrResidueData` - Tests for `_process_chain_or_residue_data`
   - Tests for various input types (None, string, list)
   - Tests for file-based inputs (JSON, text)
   - Tests for different separators
   - Tests for edge cases (empty strings, lists)
   - Tests for error handling

3. `TestProcessPerResidueData` - Tests for `_process_per_residue_data`
   - Tests for various input types
   - Tests for file-based inputs
   - Tests for dictionaries with file and non-file keys
   - Tests for nested data structures
   - Tests for error handling

4. `TestGetModelCheckpoint` - Tests for `_get_model_checkpoint`
   - Tests for different model types
   - Tests for handling of model names and characters

5. `TestExpandResidueRanges` - Tests for `expand_residue_ranges`
   - Tests for basic range expansion
   - Tests for different input formats (string, file)
   - Tests for different range notations (A1-A5, A1-5)
   - Tests for mixed formats and separators
   - Tests for edge cases and error handling
   - Parametrized tests for various input patterns

## Fixtures

The test suite uses several fixtures defined in `conftest.py`:

- `minimal_params` - A `LigandMPNNParameters` object with minimal required parameters
- `full_params` - A `LigandMPNNParameters` object with all parameters set
- `protein_mpnn_params` - A `LigandMPNNParameters` object for protein_mpnn model type
- `temp_json_file` - A temporary JSON file that's cleaned up after tests
- `temp_text_file` - A temporary text file that's cleaned up after tests
- `mock_magika` - A mock for the Magika class used for file type detection
- `sample_pdbs` - A list of sample PDB names for testing
- `temp_dir` - A temporary directory for test files that's cleaned up after tests
- `create_test_file` - A utility function to create temporary test files with content

## Extending the Test Suite

To add new tests:

1. Add a new test method to the appropriate test class
2. Use existing fixtures where appropriate
3. Create new fixtures in `conftest.py` if needed
4. Ensure all edge cases and error conditions are covered

## Code Coverage

To run the tests with coverage information:

```bash
pytest tests/test_ligandmpnn_utils.py --cov=mabmaker.tools.ligandmpnn
```

To generate a coverage report:

```bash
pytest tests/test_ligandmpnn_utils.py --cov=mabmaker.tools.ligandmpnn --cov-report=html
```

This will create an HTML report in the `htmlcov` directory. 