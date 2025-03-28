"""This module implements local sequence alignment using the Smith-Waterman algorithm.

It provides functionality to compute alignment scores, perform traceback to find optimal alignments, and handle multiple sequences for pairwise alignment.
The module is designed to work with sequences in FASTA format and uses the PAM250 substitution matrix for scoring.

As optimizations for this module I used numpy arrays instead of lists to store the motifs and the DNA sequences.
This allows for more efficient operations on the data, as numpy arrays are optimized for numerical operations.
I also initialized parallelization using the numba package. I used Just-in-Time Compilation (jit decorator) for the local_alignment_score_matrix

Functions:
    read_fasta_file(file_path: str | Path) -> npt.NDArray[np.int32]:
        Reads a FASTA file and converts sequences into a NumPy array of integer indices based on a predefined alphabet.

    convert_keys_to_indices(sequence: str) -> npt.NDArray[np.int32]:
        Converts a sequence of characters into their corresponding indices using the alphabet derived from the PAM250 matrix.

    local_alignment_score_matrix(seq1: npt.NDArray[np.int32], seq2: npt.NDArray[np.int32]) -> tuple[npt.NDArray[np.int64], np.int64]:
        Computes the local alignment score matrix for two sequences using the Smith-Waterman algorithm and returns the score matrix and maximum alignment score.

    local_alignment_score(file_path: str) -> int:
        Reads two sequences from a FASTA file, computes their local alignment score matrix, and returns the maximum score.

    local_alignment(file_path: str) -> tuple[str, str]:
        Performs local sequence alignment by computing the score matrix and tracing back to find the optimal alignment. Returns the aligned sequences as strings.

    calculate_scores(sequences: npt.NDArray[np.int32]) -> npt.NDArray[np.int64]:
        Computes a symmetric matrix of pairwise local alignment scores for a set of sequences.

    multiple_local_alignment(infile: str | Path, output: str | Path | None = None, **kwargs) -> list[list[int]] | None:
        Performs pairwise local alignments for multiple sequences in a FASTA file and optionally saves the resulting score matrix to a file.

"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
from Bio.Align.substitution_matrices import Array, load
from Bio.SeqIO import parse

numba_imported: bool = False

try:
    from numba import jit

    numba_imported = True
except ImportError:
    numba_imported = False


def jit_decorator(func):
    if numba_imported:
        return jit(parallel=True)(func)
    return func


pam250: Array = load("PAM250")  # type: ignore
alphabet: list[str] = list(str(pam250.alphabet))

sigma: np.uint8 = np.uint8(5)


def read_fasta_file(file_path: str | Path) -> npt.NDArray[np.int32]:
    """Reads a FASTA file and converts the sequences into a NumPy array of integer indices.

    Args:
        file_path: The path to the FASTA file to be read.

    Returns:
        A NumPy array where each sequence in the FASTA file is
        converted to an array of integer indices. The conversion is performed using
        the `convert_keys_to_indices` function.
    """
    return np.array(
        [
            convert_keys_to_indices(str(record.seq))
            for record in parse(file_path, "fasta")
        ],
        dtype=object,
    )


def convert_keys_to_indices(sequence: str) -> npt.NDArray[np.int32]:
    """Converts a sequence of characters into their corresponding indices based on a predefined alphabet.

    Args:
        sequence: A string representing the sequence of characters to be converted.

    Returns:
        A NumPy array of integers representing the indices of the characters in the sequence.

    Example:
        >>> convert_keys_to_indices("ACGT")
        array([ 0,  4,  7, 16], dtype=int32)
        >>> convert_keys_to_indices("GATTACA")
        array([ 7,  0, 16, 16,  0,  4,  0], dtype=int32)
    """
    indices: npt.NDArray[np.int32] = np.array(
        [alphabet.index(char) for char in sequence], dtype=np.int32
    )
    return indices


@jit_decorator
def local_alignment_score_matrix(
    seq1: npt.NDArray[np.int32], seq2: npt.NDArray[np.int32]
) -> tuple[npt.NDArray[np.int64], np.int64]:
    """Computes the local alignment score matrix for two sequences using the Smith-Waterman algorithm.
    This function calculates the score matrix for local sequence alignment and determines the maximum
    alignment score. The scoring is based on the PAM250 substitution matrix for matches/mismatches
    and a fixed penalty of 5 for insertions and deletions.

    Args:
        seq1: The first sequence represented as a NumPy array of integers.
        seq2: The second sequence represented as a NumPy array of integers.

    Returns:
        A tuple containing:
        - The score matrix as a 2D NumPy array.
        - The maximum alignment score as an integer.

    Example:
        >>> seq1 = np.array([alphabet.index(char) for char in "MEANLY"], dtype=np.int32)
        >>> seq2 = np.array([alphabet.index(char) for char in "PENALTY"], dtype=np.int32)
        >>> score_matrix, max_score = local_alignment_score_matrix(seq1, seq2)
        >>> max_score
        15
        >>> score_matrix
        array([[ 0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  4,  0,  0],
               [ 0,  0,  4,  1,  0,  0,  4,  0],
               [ 0,  1,  0,  4,  3,  0,  1,  1],
               [ 0,  0,  2,  2,  4,  0,  0,  0],
               [ 0,  0,  0,  0,  0, 10,  5,  0],
               [ 0,  0,  0,  0,  0,  5,  7, 15]])
    """
    len_seq1: int = seq1.size
    len_seq2: int = seq2.size
    score_matrix: npt.NDArray[np.int64] = np.zeros(
        (len_seq1 + 1, len_seq2 + 1), dtype=np.int64
    )

    for i in range(1, len_seq1 + 1):
        for j in range(1, len_seq2 + 1):
            match: int = (
                score_matrix[i - 1, j - 1] + pam250[seq1[i - 1], seq2[j - 1]]  # type: ignore
            )
            delete: int = score_matrix[i - 1, j] - sigma
            insert: int = score_matrix[i, j - 1] - sigma
            score_matrix[i, j] = max(match, delete, insert, 0)

    max_score: np.int64 = np.max(score_matrix)
    return score_matrix, max_score


def local_alignment_score(file_path: str) -> int:
    """Computes the maximum local alignment score for two sequences provided in a FASTA file.
    This function reads two sequences from a FASTA file, computes their local alignment
    score matrix, and returns the maximum score.

    Args:
        file_path: The path to the FASTA file containing the sequences.

    Returns:
        The maximum local alignment score.

    Example:
        >>> local_alignment_score('data01.faa')
        15
    """
    sequences: npt.NDArray[np.int32] = read_fasta_file(file_path)
    max_score: np.int64
    _, max_score = local_alignment_score_matrix(sequences[0], sequences[1])
    return int(max_score)


def local_alignment(file_path: str) -> tuple[str, str]:
    """Perform local sequence alignment using the Smith-Waterman algorithm.
    This function reads two sequences from a FASTA file, computes the local alignment
    score matrix, and traces back to find the optimal local alignment.

    Args:
        file_path: The path to the FASTA file containing two sequences.

    Returns:
        A tuple containing the two aligned sequences as strings.

    Example:
        >>> local_alignment('data01.faa')
        ('EANL-Y', 'ENALTY')
    """
    sequences: npt.NDArray[np.int32] = read_fasta_file(file_path)
    seq1: npt.NDArray[np.int32] = sequences[0]
    seq2: npt.NDArray[np.int32] = sequences[1]
    score_matrix: npt.NDArray[np.int64]
    max_score: np.int64
    score_matrix, max_score = local_alignment_score_matrix(seq1, seq2)
    indices: tuple[npt.NDArray[np.intp], ...]
    i: np.intp
    j: np.intp
    indices = np.where(score_matrix == max_score)
    i, j = indices[0][0], indices[1][0]

    align1: npt.NDArray[np.str_] = np.array([], dtype=np.str_)
    align2: npt.NDArray[np.str_] = np.array([], dtype=np.str_)
    while i > 0 and j > 0 and score_matrix[i, j] > 0:
        if (
            score_matrix[i, j]
            == score_matrix[i - 1, j - 1] + pam250[seq1[i - 1], seq2[j - 1]]  # type: ignore
        ):
            align1 = np.insert(align1, 0, alphabet[seq1[i - 1]])
            align2 = np.insert(align2, 0, alphabet[seq2[j - 1]])
            i, j = i - 1, j - 1
        elif score_matrix[i, j] == score_matrix[i - 1, j] - sigma:
            align1 = np.insert(align1, 0, alphabet[seq1[i - 1]])
            align2 = np.insert(align2, 0, "-")
            i -= 1
        else:
            align1 = np.insert(align1, 0, "-")
            align2 = np.insert(align2, 0, alphabet[seq2[j - 1]])
            j -= 1

    return ("".join(align1), "".join(align2))


def calculate_scores(sequences: npt.NDArray[np.int32]) -> npt.NDArray[np.int64]:
    """Calculate pairwise local alignment scores for a set of sequences.
    This function computes a symmetric matrix of local alignment scores
    for all pairs of sequences provided in the input array. The local
    alignment score for each pair is calculated using the
    `local_alignment_score_matrix` function.

    Args:
        sequences: A 1D array of sequences represented as integers.

    Returns:
        A 2D symmetric array where the element at position (i, j)
        represents the local alignment score between sequences[i]
        and sequences[j].

    Example:
    >>> sequences = read_fasta_file('data_10.fna')
    >>> calculate_scores(sequences)
    array([[  0, 409, 270, 307, 181, 430, 222, 265, 309, 281],
           [409,   0, 192, 251, 104, 307, 137, 183, 225, 197],
           [270, 192,   0, 184, 116, 265, 164, 162, 222, 181],
           [307, 251, 184,   0, 123, 242, 123, 180, 184, 166],
           [181, 104, 116, 123,   0, 181, 197, 111, 143, 124],
           [430, 307, 265, 242, 181,   0, 229, 222, 260, 252],
           [222, 137, 164, 123, 197, 229,   0, 140, 117, 166],
           [265, 183, 162, 180, 111, 222, 140,   0, 162, 152],
           [309, 225, 222, 184, 143, 260, 117, 162,   0, 193],
           [281, 197, 181, 166, 124, 252, 166, 152, 193,   0]])
    """
    amount_seq = sequences.size
    scores = np.zeros((amount_seq, amount_seq), dtype=np.int64)
    for i in range(amount_seq):
        for j in range(i + 1, amount_seq):
            _, score = local_alignment_score_matrix(sequences[i], sequences[j])
            scores[i, j] = score
            scores[j, i] = score

    return scores


def multiple_local_alignment(
    infile: str | Path, output: str | Path | None = None, **kwargs
) -> list[list[int]] | None:
    """Perform multiple local alignments on a set of sequences and optionally save the resulting score matrix.

    Args:
        infile: Path to the input FASTA file containing sequences.
        output: Path to save the resulting score matrix. If None, the matrix is returned as a list of lists. Defaults to None.
        **kwargs: Additional keyword arguments for formatting the output file. Supported keys:
            - fmt: Format string for saving the scores (default: "%d").
            - delimiter: Delimiter for separating values in the output file (default: " ").
            - header: Header line for the output file (default: "").

    Returns:
        list[list[int]] | None: If `output` is None, returns the score matrix as a list of lists. Otherwise, saves the matrix to the specified file and returns None.

    Example:
        >>> len(multiple_local_alignment('data_10.fna'))
        10
        >>> from pathlib import Path
        >>> matrix = Path('matrix.txt')
        >>> multiple_local_alignment(Path('data_10.fna'), matrix)
        >>> len(open(matrix).readlines())
        10
        >>> print(matrix.read_text())
        0 409 270 307 181 430 222 265 309 281
        409 0 192 251 104 307 137 183 225 197
        270 192 0 184 116 265 164 162 222 181
        307 251 184 0 123 242 123 180 184 166
        181 104 116 123 0 181 197 111 143 124
        430 307 265 242 181 0 229 222 260 252
        222 137 164 123 197 229 0 140 117 166
        265 183 162 180 111 222 140 0 162 152
        309 225 222 184 143 260 117 162 0 193
        281 197 181 166 124 252 166 152 193 0
        <BLANKLINE>
    """
    sequences: npt.NDArray[np.int32] = read_fasta_file(infile)

    scores: npt.NDArray[np.int64] = calculate_scores(sequences)

    if output:
        np.savetxt(
            output,
            scores,
            fmt=kwargs.get("fmt", "%d"),
            delimiter=kwargs.get("delimiter", " "),
            header=kwargs.get("header", ""),
        )
        return None
    return scores.tolist()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    multiple_local_alignment("data70.fna", "matrix.txt")
