import os
import tempfile

import pytest

from mabmaker.tools.ligandmpnn import expand_residue_ranges


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


def test_basic_range_from_file(temp_dir):
    """Test basic range expansion from file"""
    file_path = create_test_file(temp_dir, "A1-A5")
    result = expand_residue_ranges(file_path)
    assert result == "A1 A2 A3 A4 A5"


def test_direct_string_input():
    """Test with direct string input instead of file"""
    result = expand_residue_ranges("A1-A5")
    assert result == "A1 A2 A3 A4 A5"


def test_range_with_chain_on_first_only():
    """Test range with chain identifier only on first element"""
    result = expand_residue_ranges("A1-5")
    assert result == "A1 A2 A3 A4 A5"


def test_mixed_range_formats():
    """Test mixed range formats in the same input"""
    result = expand_residue_ranges("A1-A5,B10-15")
    assert result == "A1 A2 A3 A4 A5 B10 B11 B12 B13 B14 B15"


def test_single_residues():
    """Test single residues without ranges"""
    result = expand_residue_ranges("A1,B10,C15")
    assert result == "A1 B10 C15"


def test_comma_separated_values():
    """Test comma-separated inputs"""
    result = expand_residue_ranges("A1-A3,B5,C7-C9")
    assert result == "A1 A2 A3 B5 C7 C8 C9"


def test_whitespace_separated_values():
    """Test whitespace-separated inputs"""
    result = expand_residue_ranges("A1-A3 B5 C7-C9")
    assert result == "A1 A2 A3 B5 C7 C8 C9"


def test_mixed_separators():
    """Test mixed separators (both commas and whitespace)"""
    result = expand_residue_ranges("A1-A3, B5 C7-C9")
    assert result == "A1 A2 A3 B5 C7 C8 C9"


def test_multiple_whitespaces():
    """Test with multiple whitespaces between items"""
    result = expand_residue_ranges("A1-A3   B5  \t  C7-C9")
    assert result == "A1 A2 A3 B5 C7 C8 C9"


def test_output_to_file(temp_dir):
    """Test writing output to a file"""
    input_path = create_test_file(temp_dir, "A1-A5,B10")
    output_path = os.path.join(temp_dir, "output.txt")

    expand_residue_ranges(input_path, output_path)

    with open(output_path, "r") as f:
        content = f.read().strip()

    assert content == "A1 A2 A3 A4 A5 B10"


def test_empty_input(temp_dir):
    """Test with empty input"""
    file_path = create_test_file(temp_dir, "")
    result = expand_residue_ranges(file_path)
    assert result == ""


def test_invalid_range_format():
    """Test with invalid range format"""
    result = expand_residue_ranges("A1--A5")
    assert result == ""  # Should skip invalid range


def test_invalid_residue_format():
    """Test with invalid residue format"""
    result = expand_residue_ranges("A1,123,B5")
    assert result == "A1 B5"  # Should skip invalid residue


def test_nonexistent_file():
    """Test with a non-existent file path"""
    result = expand_residue_ranges("/path/to/nonexistent/file.txt")
    assert result is None  # Should return None for file not found


def test_reverse_range():
    """Test with end number less than start number"""
    result = expand_residue_ranges("A5-A1")
    assert result == ""  # A5-A1 is an invalid range as 5 > 1


def test_complex_mixed_input():
    """Test with a complex mix of range formats and separators"""
    complex_input = "A1-5, B10-B15 C20,D25-30 E40"
    result = expand_residue_ranges(complex_input)
    expected = "A1 A2 A3 A4 A5 B10 B11 B12 B13 B14 B15 C20 D25 D26 D27 D28 D29 D30 E40"
    assert result == expected


# Additional pytest-specific tests with parametrization
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
def test_parametrized_inputs(input_str, expected):
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
def test_invalid_inputs(invalid_input):
    """Test various invalid inputs with parametrization"""
    result = expand_residue_ranges(invalid_input)
    assert result == ""  # All should be skipped
