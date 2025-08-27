"""This module implements the Gibbs sampling algorithm to find the best scoring motifs in a collection of DNA sequences.

As optimizations for this module I used numpy arrays instead of lists to store the motifs and the DNA sequences.
This allows for more efficient operations on the data, as numpy arrays are optimized for numerical operations.

Functions:
    gibbs_sampler(k: int, fasta_file: str, N: int, rng: int | None = None, pseudocounts: bool = True, starts: int = 20) -> tuple[str, ...]:
        Perform Gibbs sampling to find the best scoring motifs in a collection of DNA sequences.
    random_kmer(sequence: str, k: int, random_generator: np.random.Generator) -> str:
        Select a random k-mer from a given DNA sequence.
    build_profile(motifs: npt.NDArray[np.str_], pseudocounts: bool) -> np.ndarray:
        Build a profile matrix with or without pseudocounts.
    profile_random_kmer(sequence: str, k: int, profile: np.ndarray, random_generator: np.random.Generator) -> str:
        Select a k-mer from a sequence based on the given profile matrix.
    score(motifs: npt.NDArray[np.str_]) -> int:
        Calculate the score of a set of motifs based on consensus deviation.
"""

import numpy as np
import numpy.typing as npt
from Bio import SeqIO


# I disabled too-many-arguments because the function signature is required by the assignment.
# There are no coherent groups of arguments that can be abstracted.
# pylint: disable=too-many-arguments
def gibbs_sampler(
    k: int,
    fasta_file: str,
    N: int,
    rng: int | np.random.Generator | None = None,
    pseudocounts: bool = True,
    starts: int = 20,
) -> tuple[str, ...]:
    """Perform Gibbs sampling to find the best scoring motifs in a collection of DNA sequences.

    Args:
        k: Length of the k-mers to find.
        fasta_file: Path to the FASTA file containing DNA sequences.
        N: Number of iterations for Gibbs sampling.
        rng: Random seed.
        pseudocounts: Whether to use pseudocounts in the profile matrix.
        starts: Number of random restarts.

    Returns:
        tuple: Best k-mers found in each sequence.

    Example:
        >>> gibbs_sampler(3, "data/data01.fna", 100, rng=42, starts=5)
        ('CGG', 'CAG', 'CAG', 'CAG', 'CAG')
    """
    if isinstance(rng, np.random.Generator):
        random_generator = rng
    else:
        random_generator: np.random.Generator = np.random.default_rng(rng)
    dna_sequences: npt.NDArray[np.str_] = np.array(
        [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]
    )
    t: int = len(dna_sequences)

    best_motifs: npt.NDArray[np.str_] = np.array(
        [random_kmer(seq, k, random_generator) for seq in dna_sequences]
    )
    for _ in range(starts):
        motifs: npt.NDArray[np.str_] = np.array(
            [random_kmer(seq, k, random_generator) for seq in dna_sequences]
        )
        for _ in range(N):
            i: int = int(random_generator.integers(0, t - 1))
            profile: npt.NDArray[np.float64] = build_profile(
                np.concatenate((motifs[:i], motifs[i + 1 :])), k, pseudocounts
            )
            motifs[i] = profile_random_kmer(
                dna_sequences[i], k, profile, random_generator
            )

            if score(motifs, k, t) < score(best_motifs, k, t):
                best_motifs = motifs.copy()
    return tuple(str(motif) for motif in best_motifs)


def random_kmer(sequence: str, k: int, random_generator: np.random.Generator) -> str:
    """Select a random k-mer from a given DNA sequence.

    Args:
        sequence: DNA sequence.
        k: Length of the k-mer
        random_generator: Random number generator.

    Returns:
        Random k-mer from the sequence.

    Example:
        >>> random_kmer("ACGTACGT", 3, np.random.default_rng(42))
        'ACG'
    """
    start: int = int(random_generator.integers(0, len(sequence) - k))
    return sequence[start : start + k]


def build_profile(
    motifs: npt.NDArray[np.str_], k: int, pseudocounts: bool
) -> npt.NDArray[np.float64]:
    """
    Build a profile matrix with or without pseudocounts.

    Args:
        motifs: Array of motifs.
        k: Length of the motifs.
        pseudocounts: Whether to use pseudocounts in the profile matrix.

    Returns:
        Profile matrix.

    Example:
        >>> build_profile(np.array(["ATG", "AAG", "TTG"]), 3, pseudocounts=True)
        array([[0.42857143, 0.28571429, 0.14285714],
               [0.14285714, 0.14285714, 0.14285714],
               [0.14285714, 0.14285714, 0.57142857],
               [0.28571429, 0.42857143, 0.14285714]])
    """
    profile: npt.NDArray[np.float64] = (
        np.ones((4, k)) if pseudocounts else np.zeros((4, k))
    )
    mapping: dict = {"A": 0, "C": 1, "G": 2, "T": 3}

    for j in range(len(motifs[0])):
        for motif in motifs:
            profile[mapping[motif[j]], j] += 1

    profile /= profile.sum(axis=0)
    return profile


def profile_random_kmer(
    sequence: str, k: int, profile: np.ndarray, random_generator: np.random.Generator
) -> str:
    """Select a k-mer from a sequence based on the given profile matrix.

    Args:
        sequence: DNA sequence.
        k: Length of the k-mer.
        profile: Profile matrix.
        random_generator: Random number generator.

    Returns:
        K-mer selected from the sequence based on the profile matrix.

    Example:
        >>> profile = np.array([[0.2, 0.3, 0.5], [0.3, 0.3, 0.2], [0.3, 0.2, 0.2], [0.2, 0.2, 0.1]])
        >>> profile_random_kmer("ATGCGT", 3, profile, np.random.default_rng(42))
        'GCG'
    """

    mapping: npt.NDArray[np.uint8] = np.array([ord("A"), ord("C"), ord("G"), ord("T")])
    seq_arr: npt.NDArray[np.uint8] = np.frombuffer(sequence.encode(), dtype=np.uint8)
    indices: npt.NDArray[np.intp] = np.searchsorted(mapping, seq_arr)
    kmers_indices: npt.NDArray[np.intp] = np.lib.stride_tricks.sliding_window_view(
        indices, k
    )

    probabilities = np.prod(profile[kmers_indices, np.arange(k)], axis=1)
    probabilities /= probabilities.sum()

    selected_index: int = random_generator.choice(len(probabilities), p=probabilities)
    return sequence[selected_index : selected_index + k]


def score(motifs: npt.NDArray[np.str_], k: int, t: int) -> np.uint8:
    """Calculate the score of a set of motifs based on consensus deviation.

    Args:
        motifs: Array of motifs.
        k: Length of the motifs.
        t: Number of motifs.

    Returns:
        Score of the motifs.

    Example:
        >>> int(score(np.array(["ATG", "AAG", "TTG"]), 3, 3))
        2
    """
    counts: npt.NDArray[np.uint8] = np.zeros((4, k), dtype=int)
    mapping: dict = {"A": 0, "C": 1, "G": 2, "T": 3}

    for motif in motifs:
        for j, char in enumerate(motif):
            counts[mapping[char], j] += 1

    consensus: np.uint8 = np.argmax(counts, axis=0)
    score: np.uint8 = t * k - np.sum(counts[consensus, np.arange(k)])
    return score


def main():
    print(gibbs_sampler(4, "data/data01.fna", 100, rng=42, starts=20))


if __name__ == "__main__":
    import doctest

    main()
    doctest.testmod()
