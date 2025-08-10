"""This module implements a profile Hidden Markov Model (HMM) for multiple sequence alignment.

It provides functionality to build a profile HMM from a multiple sequence alignment, compute match columns, process sequences into transition and emission counts, normalize probabilities,
and align new sequences to the HMM using the Viterbi algorithm.

The module is designed to work with sequences and alignments, allowing for probabilistic modeling of sequence variation and alignment scoring.
It includes support for pseudocounts to handle zero probabilities and uses efficient data structures like NumPy arrays for matrix operations.

The module is not entirely finished and requires further development to fully implement and test the profile HMM functionality. But due to other deadlines, it was not possible for me to finish it.
I will document further on optimizations and improvements in my report.

Functions:
    viterbi(observations: str, emission_mapping: Dict[str, int], hidden_mapping: Dict[str, int], transition_matrix: npt.NDArray[np.float64], emission_matrix: npt.NDArray[np.float64]) -> List[str]:
        Implements the Viterbi algorithm to find the most probable sequence of hidden states for a given observation sequence.

    process_sequence_into_counts(sequence: str, is_match_column: List[bool], transition_counts: Dict[str, Dict[str, float]], emission_counts: Dict[str, Dict[str, float]]) -> None:
        Updates transition and emission counts based on a sequence and match column information.

    add_pseudocounts(counts: Dict[str, Dict[str, float]], pseudocount: float) -> None:
        Adds pseudocounts to transition or emission counts to avoid zero probabilities.

    emission_probability(emission: str, emission_mapping: Dict[str, int], hidden_mapping: Dict[str, int], transition_matrix: npt.NDArray[np.float64], emission_matrix: npt.NDArray[np.float64]) -> float:
        Computes the probability of observing a given emission sequence using the forward algorithm.

    compute_match_columns(alignment: List[str], threshold: float) -> List[bool]:
        Determines which columns in a multiple sequence alignment are match columns based on a gap threshold.

    initialize_counts(states: List[str], alphabet: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        Initializes dictionaries for transition and emission counts with zero values.

    normalize_counts(counts: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        Normalizes counts to probabilities for transition and emission matrices.

    build_state_space(num_match: int) -> List[str]:
        Constructs the state space for a profile HMM based on the number of match states.

    encode_mapping(states: List[str], alphabet: str) -> Tuple[Dict[str, int], Dict[str, int]]:
        Creates mappings for hidden states and observation symbols to indices.

    build_matrices(hidden_map: Dict[str, int], emission_map: Dict[str, int], trans_probs: Dict[str, Dict[str, float]], emit_probs: Dict[str, Dict[str, float]]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        Constructs transition and emission matrices from probability dictionaries.

    profile_HMM_sequence_alignment(text: str, theta: float, pseudocount: float, alphabet: str, alignment: List[str]) -> List[str]:
        Aligns a sequence to a profile HMM built from a multiple sequence alignment and returns the most probable sequence of hidden states.
"""

from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt


def viterbi(
    observations: str,
    emission_mapping: Dict[str, int],
    hidden_mapping: Dict[str, int],
    transition_matrix: npt.NDArray[np.float64],
    emission_matrix: npt.NDArray[np.float64],
) -> List[str]:
    """Implements the Viterbi algorithm to find the most probable sequence of hidden states
    given a sequence of observations, transition probabilities, and emission probabilities.

    Args:
        observations: The observed sequence.
        emission_mapping: Mapping of observation symbols to indices.
        hidden_mapping: Mapping of hidden states to indices.
        transition_matrix: Matrix of transition probabilities between hidden states.
        emission_matrix: Matrix of emission probabilities for each hidden state.

    Returns:
        The most probable sequence of hidden states.

    Example:
        >>> viterbi('yxzxx', {'z': 2, 'x': 0, 'y': 1}, {'B': 1, 'A': 0}, [[0.24439433296892832, 0.7556056670310717], [0.24241542404217956, 0.7575845759578205]], [[0.7180611502606178, 0.19057500286653098, 0.09136384687285123], [0.291514305851776, 0.4407137316635376, 0.26777196248468643]])
        ['B', 'B', 'B', 'B', 'B']
        >>> viterbi('yyyyzyxxxy', {'z': 2, 'x': 0, 'y': 1}, {'B': 1, 'A': 0}, [[0.2915439583949658, 0.7084560416050342], [0.5443249998401023, 0.4556750001598978]], [[0.6496850841876853, 0.13086798248939702, 0.2194469333229175], [0.17166446158695747, 0.681506463779169, 0.14682907463387349]])
        ['B', 'B', 'B', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    """
    num_states: int = len(hidden_mapping)
    num_obs: int = len(observations)

    reverse_hidden: Dict[int, str] = {i: s for s, i in hidden_mapping.items()}
    obs_idx: List[int] = [emission_mapping[c] for c in observations]

    V: List[List[float]] = [[0.0] * num_states for _ in range(num_obs)]
    backpointer: List[List[int]] = [[0] * num_states for _ in range(num_obs)]

    for s in range(num_states):
        V[0][s] = (1.0 / num_states) * emission_matrix[s][obs_idx[0]]

    for t in range(1, num_obs):
        for s in range(num_states):
            max_prob: float = -1.0
            max_state: int = 0
            for s_prev in range(num_states):
                prob: float = (
                    V[t - 1][s_prev]
                    * transition_matrix[s_prev][s]
                    * emission_matrix[s][obs_idx[t]]
                )
                if prob > max_prob:
                    max_prob = prob
                    max_state = s_prev
            V[t][s] = max_prob
            backpointer[t][s] = max_state

    best_path: List[int] = []
    last_state: int = max(range(num_states), key=lambda s: V[-1][s])
    best_path.append(last_state)

    for t in range(num_obs - 1, 0, -1):
        last_state = backpointer[t][last_state]
        best_path.append(last_state)

    best_path.reverse()
    return [reverse_hidden[i] for i in best_path]


def process_sequence_into_counts(
    sequence: str,
    is_match_column: List[bool],
    transition_counts: Dict[str, Dict[str, float]],
    emission_counts: Dict[str, Dict[str, float]],
) -> None:
    """Processes a sequence to update transition and emission counts based on match columns.

    Args:
        sequence: The sequence to process.
        is_match_column: A list indicating whether each column is a match column.
        transition_counts: Dictionary to store transition counts between states.
        emission_counts: Dictionary to store emission counts for each state.

    Example:
        >>> transition_counts = {'S': {'M1': 0.0, 'I0': 0.0, 'E': 0.0}, 'M1': {'M2': 0.0, 'I1': 0.0, 'E': 0.0}}
        >>> emission_counts = {'M1': {'A': 0.0, 'C': 0.0, 'G': 0.0, 'T': 0.0}, 'I0': {'A': 0.0, 'C': 0.0, 'G': 0.0, 'T': 0.0}}
        >>> process_sequence_into_counts('ACGT', [True, False, True, True], transition_counts, emission_counts)
        >>> transition_counts
        {'S': {'M1': 1.0, 'I0': 0.0, 'E': 0.0}, 'M1': {'M2': 0.0, 'I1': 1.0, 'E': 0.0}}
        >>> emission_counts
        {'M1': {'A': 1.0, 'C': 0.0, 'G': 0.0, 'T': 0.0}, 'I0': {'A': 0.0, 'C': 0.0, 'G': 0.0, 'T': 0.0}}
    """
    previous_state: str = "S"
    match_index: int = 0

    for col_index, symbol in enumerate(sequence):
        if is_match_column[col_index]:
            match_index += 1
            current_state: str = (
                f"D{match_index}" if symbol == "-" else f"M{match_index}"
            )
        elif symbol != "-":
            current_state = f"I{match_index}"
        else:
            continue

        if symbol != "-" and current_state in emission_counts:
            emission_counts[current_state][symbol] += 1.0

        if current_state in transition_counts[previous_state]:
            transition_counts[previous_state][current_state] += 1.0

        previous_state = current_state

    if "E" in transition_counts[previous_state]:
        transition_counts[previous_state]["E"] += 1.0


def add_pseudocounts(counts: Dict[str, Dict[str, float]], pseudocount: float) -> None:
    """Adds pseudocounts to transition or emission counts to avoid zero probabilities.

    Args:
        counts: Dictionary of counts to update.
        pseudocount: The pseudocount value to add.
    """
    for src in counts:
        for dst in counts[src]:
            counts[src][dst] += pseudocount


def emission_probability(
    emission: str,
    emission_mapping: Dict[str, int],
    hidden_mapping: Dict[str, int],
    transition_matrix: npt.NDArray[np.float64],
    emission_matrix: npt.NDArray[np.float64],
) -> float:
    """Computes the probability of observing a given emission sequence using the forward algorithm.

    Args:
        emission: The observed emission sequence.
        emission_mapping: Mapping of observation symbols to indices.
        hidden_mapping: Mapping of hidden states to indices.
        transition_matrix: Matrix of transition probabilities between hidden states.
        emission_matrix: Matrix of emission probabilities for each hidden state.

    Returns:
        The probability of the emission sequence.

    Example:
        >>> emission_probability('yyyyzyxxxy', {'y': 1, 'z': 2, 'x': 0}, {'A': 0, 'B': 1}, [[0.2915439583949658, 0.7084560416050342], [0.5443249998401023, 0.4556750001598978]], [[0.6496850841876853, 0.13086798248939702, 0.2194469333229175], [0.17166446158695747, 0.681506463779169, 0.14682907463387349]])
        5.41352210854505e-05
    """
    num_states: int = len(hidden_mapping)
    num_obs: int = len(emission)
    obs_idx: List[int] = [emission_mapping[c] for c in emission]

    alpha: npt.NDArray[np.float64] = np.zeros((num_obs, num_states))

    for i in range(num_states):
        alpha[0][i] = (1.0 / num_states) * emission_matrix[i][obs_idx[0]]

    for t in range(1, num_obs):
        for j in range(num_states):
            alpha[t][j] = sum(
                alpha[t - 1][i]
                * transition_matrix[i][j]
                * emission_matrix[j][obs_idx[t]]
                for i in range(num_states)
            )

    return float(np.sum(alpha[-1]))


def compute_match_columns(alignment: List[str], threshold: float) -> List[bool]:
    """Computes which columns in a multiple sequence alignment are match columns.

    Args:
        alignment: List of aligned sequences.
        threshold: Fraction of gaps allowed for a column to be considered a match column.

    Returns:
        A list indicating whether each column is a match column.
    
    Example:
        >>> compute_match_columns(['ACGT-', 'A-GTT', 'ACG-T', 'ACGTT'], 0.5)
        [True, True, True, True, True]
    """
    num_sequences: int = len(alignment)
    num_columns: int = len(alignment[0])
    match_columns: List[bool] = []

    for j in range(num_columns):
        gap_count: int = sum(1 for i in range(num_sequences) if alignment[i][j] == "-")
        gap_fraction: float = gap_count / num_sequences
        match_columns.append(gap_fraction < threshold)

    return match_columns


def initialize_counts(
    states: List[str], alphabet: str
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Initializes transition and emission count dictionaries with zero values.

    Args:
        states: List of hidden states.
        alphabet: The alphabet of observation symbols.

    Returns:
        A tuple containing transition counts and emission counts dictionaries.

    Example:
        >>> initialize_counts(['S', 'I0', 'M1', 'D1', 'E'], 'ACGT')
        ({'S': {'S': 0.0, 'I0': 0.0, 'M1': 0.0, 'D1': 0.0, 'E': 0.0}, 'I0': {'S': 0.0, 'I0': 0.0, 'M1': 0.0, 'D1': 0.0, 'E': 0.0}, 'M1': {'S': 0.0, 'I0': 0.0, 'M1': 0.0, 'D1': 0.0, 'E': 0.0}, 'D1': {'S': 0.0, 'I0': 0.0, 'M1': 0.0, 'D1': 0.0, 'E': 0.0}, 'E': {'S': 0.0, 'I0': 0.0, 'M1': 0.0, 'D1': 0.0, 'E': 0.0}}, {'I0': {'A': 0.0, 'C': 0.0, 'G': 0.0, 'T': 0.0}, 'M1': {'A': 0.0, 'C': 0.0, 'G': 0.0, 'T': 0.0}})
    """
    transition_counts: Dict[str, Dict[str, float]] = {
        s: {s2: 0.0 for s2 in states} for s in states
    }
    # Ensure all states have a valid entry for transitions to avoid KeyError
    for s in states:
        if s not in transition_counts:
            transition_counts[s] = {s2: 0.0 for s2 in states}
    emission_counts: Dict[str, Dict[str, float]] = {
        s: {a: 0.0 for a in alphabet} for s in states if s[0] in {"M", "I"}
    }
    return transition_counts, emission_counts


def normalize_counts(
    counts: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Normalizes counts to probabilities.

    Args:
        counts: Dictionary of counts to normalize.

    Returns:
        A dictionary of normalized probabilities.

    Example:
        >>> normalize_counts({'S': {'S': 2.0, 'I0': 3.0}, 'I0': {'S': 1.0, 'I0': 4.0}})
        {'S': {'S': 0.4, 'I0': 0.6}, 'I0': {'S': 0.2, 'I0': 0.8}}
    """
    probs: Dict[str, Dict[str, float]] = {}
    for src in counts:
        total: float = sum(counts[src].values())
        if total > 0:
            probs[src] = {dst: count / total for dst, count in counts[src].items()}
        else:
            probs[src] = {dst: 0.0 for dst in counts[src]}
    return probs


def build_state_space(num_match: int) -> List[str]:
    """Builds the state space for a profile HMM.

    Args:
        num_match: Number of match states.

    Returns:
        A list of states in the HMM.

    Example:
        >>> build_state_space(3)
        ['S', 'I0', 'I1', 'M1', 'D1', 'I2', 'M2', 'D2', 'I3', 'M3', 'D3', 'E']
    """
    states: List[str] = ["S"]
    for i in range(num_match + 1):
        states.append(f"I{i}")
        if i > 0:
            states.extend([f"M{i}", f"D{i}"])
    states.append("E")
    return states


def encode_mapping(
    states: List[str], alphabet: str
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Encodes mappings for hidden states and observation symbols.

    Args:
        states: List of hidden states.
        alphabet: The alphabet of observation symbols.

    Returns:
        A tuple containing hidden state mapping and emission mapping.

    Example:
        >>> encode_mapping(['S', 'I0', 'M1', 'D1', 'E'], 'ACGT')
        ({'S': 0, 'I0': 1, 'M1': 2, 'D1': 3, 'E': 4}, {'A': 0, 'C': 1, 'G': 2, 'T': 3})
    """
    hidden_map: Dict[str, int] = {s: i for i, s in enumerate(states)}
    emission_map: Dict[str, int] = {a: i for i, a in enumerate(alphabet)}
    return hidden_map, emission_map


def build_matrices(
    hidden_map: Dict[str, int],
    emission_map: Dict[str, int],
    trans_probs: Dict[str, Dict[str, float]],
    emit_probs: Dict[str, Dict[str, float]],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Builds transition and emission matrices from probabilities.

    Args:
        hidden_map: Mapping of hidden states to indices.
        emission_map: Mapping of observation symbols to indices.
        trans_probs: Transition probabilities between states.
        emit_probs: Emission probabilities for each state.

    Returns:
        A tuple containing the transition matrix and emission matrix.

    Example:
        >>> build_matrices({'S': 0, 'I0': 1, 'M1': 2, 'D1': 3, 'E': 4}, {'A': 0, 'C': 1, 'G': 2, 'T': 3}, {'S': {'I0': 0.5, 'M1': 0.5}, 'I0': {'S': 0.2, 'I0': 0.8}}, {'I0': {'A': 0.1, 'C': 0.2, 'G': 0.7}})
        (array([[0. , 0.5, 0.5, 0. , 0. ],
           [0.2, 0.8, 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. ]]), array([[0. , 0. , 0. , 0. ],
           [0.1, 0.2, 0.7, 0. ],
           [0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. ]]))
    """
    T: npt.NDArray[np.float64] = np.zeros((len(hidden_map), len(hidden_map)))
    E: npt.NDArray[np.float64] = np.zeros((len(hidden_map), len(emission_map)))

    for s1 in trans_probs:
        for s2 in trans_probs[s1]:
            if s1 in hidden_map and s2 in hidden_map:
                T[hidden_map[s1]][hidden_map[s2]] = trans_probs[s1][s2]

    for s in emit_probs:
        for a in emit_probs[s]:
            if s in hidden_map and a in emission_map:
                E[hidden_map[s]][emission_map[a]] = emit_probs[s][a]

    return T, E


def profile_HMM_sequence_alignment(
    text: str, theta: float, pseudocount: float, alphabet: str, alignment: List[str]
) -> List[str]:
    """Aligns a sequence to a profile HMM built from a multiple sequence alignment.

    Args:
        text: The sequence to align.
        theta: Threshold for determining match columns.
        pseudocount: Pseudocount value for smoothing probabilities.
        alphabet: The alphabet of observation symbols.
        alignment: List of aligned sequences.

    Returns:
        The most probable sequence of hidden states for the input sequence.

    Example:
        >>> profile_HMM_sequence_alignment('C', 0.2, 0.01, 'ABC', ['B---B', 'CAACC', 'CC-AB', 'B-BCB', 'CBBBB'])
        ['M1', 'D2']
    """
    match_cols: List[bool] = compute_match_columns(alignment, theta)
    num_match: int = sum(match_cols)
    states: List[str] = build_state_space(num_match)

    transition_counts, emission_counts = initialize_counts(states, alphabet)

    for sequence in alignment:
        process_sequence_into_counts(
            sequence, match_cols, transition_counts, emission_counts
        )

    add_pseudocounts(transition_counts, pseudocount)
    add_pseudocounts(emission_counts, pseudocount)

    trans_probs: Dict[str, Dict[str, float]] = normalize_counts(transition_counts)
    emit_probs: Dict[str, Dict[str, float]] = normalize_counts(emission_counts)

    hidden_map, emission_map = encode_mapping(states, alphabet)
    T, E = build_matrices(hidden_map, emission_map, trans_probs, emit_probs)

    path: List[str] = viterbi(text, emission_map, hidden_map, T, E)
    return path

def main():
    import doctest

    doctest.testmod()


if __name__ == "__main__":
    main()