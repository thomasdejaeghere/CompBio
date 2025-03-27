from itertools import combinations, product
from Bio import SeqIO

import numpy as np


def neighbors(sequence: str, dist: int) -> set[str]:
    alphabet = "ACGT"
    result = set()

    for positions in combinations(list(range(len(sequence))), dist):
        for changes in product(alphabet, repeat=dist):
            temp = np.array(list(sequence))
            temp[list(positions)] = changes
            result.add(str("".join(temp)))
    return result


print(neighbors(*SeqIO.parse('data01.fna', 'fasta'), 1))
