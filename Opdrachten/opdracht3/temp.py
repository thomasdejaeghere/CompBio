import numpy as np
from Bio.Align.substitution_matrices import load, Array

# Load the PAM250 matrix
pam250: Array = load("PAM250")  # type: ignore
alphabet = list(str(pam250.alphabet))
pam250_data = np.array(pam250)
print(pam250_data)

pam250_matrix = pam250.data  # Alphabet from PAM250 matrix

def convert_to_indices(sequence: str) -> np.ndarray:
    """Convert sequence of characters to indices based on the PAM250 alphabet."""
    return np.array([alphabet.index(char) for char in sequence], dtype=np.int32)

def get_submatrix(seq1: str, seq2: str, matrix: np.ndarray) -> np.ndarray:
    """Extract a submatrix of PAM250 for the given sequences."""
    # Convert sequences to indices
    seq1_indices = convert_to_indices(seq1)
    seq2_indices = convert_to_indices(seq2)
    
    # Create a 2D submatrix based on the indices
    submatrix = matrix[np.ix_(seq1_indices, seq2_indices)]  # Broadcasting to get all pairwise scores
    
    return submatrix

# Example usage
seq1 = 'MEANLY'
seq2 = 'PENALTY'

submatrix = get_submatrix(seq1, seq2, pam250_matrix)
print(submatrix)
