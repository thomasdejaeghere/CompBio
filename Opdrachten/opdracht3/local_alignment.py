from ctypes import Array
from pathlib import Path
import numpy as np
from Bio.Align import substitution_matrices

def local_alignment_score(file: str) -> np.uint8:
    blosum62: np.ndarray | list[str] = substitution_matrices.load("PAM250")
    score = blosum62["M","A"]
    print(score)
    return np.uint8(0)

def local_alignment(file: str) -> tuple[str, str]:
    return '', ''

def multiple_local_alignment(infile: str | Path, output: str | Path | None = None, **kwargs) -> list[list[int]] | None:
    """
    Perform multiple local alignments on the input file and optionally write the resulting matrix to an output file.
    Args:
        infile (str | Path): Path to the input file containing sequence data.
        output (str | Path | None, optional): Path to the output file where the resulting matrix will be written. 
            Defaults to None.
        **kwargs: Additional keyword arguments for customization.
    Returns:
        list[list[int]] | None: A 2D list representing the alignment matrix if no output file is specified, 
        otherwise None.

    Examples:
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
    pass

if __name__ == "__main__":
    local_alignment_score('test')
    