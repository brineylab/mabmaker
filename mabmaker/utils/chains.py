# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import itertools
import string

__all__ = ["generate_chain_names", "get_chain_name_generator"]


def get_chain_name_generator(format: str = "chai"):
    if format == "chai":
        return chai_alphabet_generator()
    elif format == "protenix":
        return protenix_alphabet_generator()
    elif format == "boltz":
        return boltz_alphabet_generator()
    else:
        raise ValueError(f"Invalid format: {format}")


def chai_alphabet_generator():
    """
    Generate an alphabet of increasing length using only uppercase and lowercase letters.
    This is the `new naming scheme for chains in Chai`_.

    .. _new naming scheme for chains in Chai: https://github.com/chaidiscovery/chai-lab/issues/290

    Examples
    --------
    >>> alphabet_generator()
    'A'
    >>> next(alphabet_generator())
    'B'
     .
     .
     .
    >>> next(alphabet_generator())
    'a'
    >>> next(alphabet_generator())
    'b'
     .
     .
     .
    >>> next(alphabet_generator())
    'AB'
    >>> next(alphabet_generator())
    'AC'
     .
     .
     .
    >>> next(alphabet_generator())
    'aa'
    >>> next(alphabet_generator())
    'ab'

    Yields
    ------
    str
        The next chain name in the sequence.
    """
    N = 1
    while True:
        for case in ["upper", "lower"]:
            letters = (
                string.ascii_uppercase if case == "upper" else string.ascii_lowercase
            )
            for chars in itertools.product(letters, repeat=N):
                yield "".join(chars)
        N += 1


def boltz_alphabet_generator():
    """
    Generate an alphabet of increasing length using only uppercase and lowercase letters.
    This is identical to the `new naming scheme for chains in Chai`_, which we're repurposaing
    for use with Boltz, which has a `maximum chain length of 5 characters`_.

    .. _new naming scheme for chains in Chai: https://github.com/chaidiscovery/chai-lab/issues/290
    .. _maximum chain length of 5 characters: https://github.com/jwohlwend/boltz/issues/181

    Examples
    --------
    >>> alphabet_generator()
    'A'
    >>> next(alphabet_generator())
    'B'
     .
     .
     .
    >>> next(alphabet_generator())
    'a'
    >>> next(alphabet_generator())
    'b'
     .
     .
     .
    >>> next(alphabet_generator())
    'AB'
    >>> next(alphabet_generator())
    'AC'
     .
     .
     .
    >>> next(alphabet_generator())
    'aa'
    >>> next(alphabet_generator())
    'ab'

    Yields
    ------
    str
        The next chain name in the sequence.
    """
    return chai_alphabet_generator()


def protenix_alphabet_generator():
    """
    Generate an alphabet of increaing integers, as strings (used for chain names in Protenix).

    Examples
    --------
    >>> alphabet_generator()
    '1'
    >>> next(alphabet_generator())
    '2'
     .
     .
     .

    Yields
    ------
    str
        The next chain name in the sequence.
    """
    N = 1
    while True:
        yield str(N)
        N += 1


def generate_chain_names(n: int, format: str = "chai"):
    """
    Generate a list of chain names for Chai-1.

    Parameters
    ----------
    n : int
        The number of chain names to generate.

    Returns
    -------
    list
        A list of chain names.
    """
    generator = get_chain_name_generator(format)
    return [next(generator) for _ in range(n)]
