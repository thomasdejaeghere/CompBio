"""This module provides functionality for generating all possible DNA sequences within a given Hamming distance from an input sequence.

Functions:
    neighbors: Generate all possible sequences within a specified Hamming distance from the input sequence.
"""

import doctest
import sys
from itertools import combinations, product
from typing import Any, Set

import numpy as np
import numpy.typing as npt
from Bio import SeqIO


def neighbors(sequence: str, dist: int) -> Set[str]:
    """Generate all possible sequences that are within a given Hamming distance from the input sequence.
    Each neighbor sequence differs from the input sequence by no more then `dist` positions,
    where each differing position can be any character from the alphabet "ACGT".

    Args:
        sequence: The original DNA sequence.
        dist: The Hamming distance (number of positions to change).

    Returns:
        A set of all neighbor sequences within the specified Hamming distance.

    Examples:
        >>> sorted(neighbors("ACG", 1))
        ['AAG', 'ACA', 'ACC', 'ACG', 'ACT', 'AGG', 'ATG', 'CCG', 'GCG', 'TCG']
        >>> len(neighbors("AAA", 2))
        37
        >>> "TAA" in neighbors("AAA", 1)
        True
        >>> neighbors("A", 1) == {'A', 'C', 'G', 'T'}
        True
    """
    alphabet: str = "ACGT"
    result: Set[str] = set()

    # The following nested for loop generates all possible neighbor sequences within the given Hamming distance.
    # It works by:
    # 1. Selecting all possible combinations of positions in the sequence to change (using combinations).
    # 2. For each combination, generating all possible substitutions at those positions (using product).
    # 3. For each substitution, creating a new sequence with the changes and adding it to the result set.
    for positions in combinations(list(range(len(sequence))), dist):
        for changes in product(alphabet, repeat=dist):
            temp: npt.NDArray[np.str_] = np.array(list(sequence))
            temp[list(positions)] = changes
            result.add(str("".join(temp)))
    return result


if __name__ == "__main__":
    doctest.testmod()
    # Check arguments
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <length>", file=sys.stderr)
        sys.exit(1)

    try:
        lengte: int = int(sys.argv[1])
    except ValueError:
        print("Length argument needs to be an integer", file=sys.stderr)
        sys.exit(1)

    # Get data from stdins
    data: Any = sys.stdin
    record: Any = next(SeqIO.parse(data, "fasta"))
    results: Set[str] = neighbors(str(record.seq), lengte)
    for result in results:
        print(result)
