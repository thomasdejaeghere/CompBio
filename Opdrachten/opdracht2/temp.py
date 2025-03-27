import random
import numpy as np
from Bio import SeqIO


def gibbs_sampler(k: int, fasta_file: str, N: int) -> tuple[str, ...]:
    """
    Perform Gibbs sampling to find the best motifs in a collection of DNA sequences.

    Args:
        k (int): Length of the k-mers to find.
        fasta_file (str): Path to the FASTA file containing DNA sequences.
        N (int): Number of iterations for Gibbs sampling.

    Returns:
        list[str]: Best k-mers found in each sequence.
    """
    dna_sequences: list[str] = [
        str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")
    ]
    t = len(dna_sequences)

    motifs = [random_kmer(seq, k) for seq in dna_sequences]
    best_motifs = motifs[:]
    for _ in range(20):
        motifs = [random_kmer(seq, k) for seq in dna_sequences]
        for _ in range(N):
            i = random.randint(0, t - 1)
            profile = build_profile(motifs[:i] + motifs[i + 1 :])
            motifs[i] = profile_random_kmer(dna_sequences[i], k, profile)

            if score(motifs) < score(best_motifs):
                best_motifs = motifs[:]

    return tuple(best_motifs)


def random_kmer(sequence: str, k: int) -> str:
    """Select a random k-mer from a given DNA sequence."""
    start = random.randint(0, len(sequence) - k)
    return sequence[start : start + k]


def build_profile(motifs: list[str]) -> np.ndarray:
    """Build a profile matrix with pseudocounts from a set of motifs."""
    k = len(motifs[0])
    profile = np.ones((4, k))
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}

    for motif in motifs:
        for j, nucleotide in enumerate(motif):
            profile[mapping[nucleotide], j] += 1

    profile /= profile.sum(axis=0)
    return profile


def profile_random_kmer(sequence: str, k: int, profile: np.ndarray) -> str:
    """Select a k-mer from a sequence based on the given profile matrix."""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    probabilities = []

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i : i + k]
        prob = np.prod([profile[mapping[n], j] for j, n in enumerate(kmer)])
        probabilities.append(prob)

    probabilities = np.array(probabilities) / sum(probabilities)
    return sequence[np.random.choice(range(len(sequence) - k + 1), p=probabilities) :][
        :k
    ]


def score(motifs: list[str]) -> int:
    """Calculate the score of a set of motifs based on consensus deviation."""
    mapping = {0: "A", 1: "C", 2: "G", 3: "T"}
    consensus = "".join(
        mapping[idx] for idx in np.argmax(build_profile(motifs), axis=0)
    )
    return sum(sum(1 for a, b in zip(motif, consensus) if a != b) for motif in motifs)


def main() -> None:
    k = 8
    N = 1000
    # for i in range(7):
    fasta_file = "data/data0" + str(1) + ".fna"
    best_motifs = gibbs_sampler(k, fasta_file, N)
    print(best_motifs)


if __name__ == "__main__":
    main()
