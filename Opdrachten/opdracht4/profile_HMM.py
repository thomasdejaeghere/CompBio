"""This module implements profile Hidden Markov Model (HMM) sequence alignment for biological sequences.
It provides functionality to build a profile HMM from a multiple sequence alignment, apply pseudocounts
for smoothing probabilities, and perform sequence alignment using an adapted Viterbi algorithm. The module
is designed to work with sequences in FASTA format and arbitrary alphabets (e.g., 'ACGT').

As optimizations for this module:
    - Dict-get() calls are replaced with regular lookups
    - Log-probabilities are precomputed to avoid repeated log calculations.
    - Transition masks reduce the number of iterations in the dynamic programming loop.

Functions:
    parse_state(state: str) -> Tuple[str, int]
    check_transition(s1: str, s2: str, end_idx: int) -> bool
    get_all_states(num_match_states: int) -> List[str]
    get_probabilities(counts: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]
    apply_pseudocounts(transitions_probs, emissions_probs, sigma, states, alphabet)
    profile_HMM_pseudocounts(theta, sigma, alphabet, patterns)
    propagate_silent(V, back, col, t_matrix)
    profile_HMM_sequence_alignment(Text, theta, sigma, alphabet, Alignment)
"""

import argparse
import math
import sys
from typing import Dict, List, Optional, Tuple

from Bio import SeqIO

MIN_FLOAT = -sys.float_info.max

# --- Versie 1 ---


def parse_state(state: str) -> Tuple[str, int]:
    """Parses a state string into its type and index.

    Args:
        state: The state string (e.g., 'M3', 'S', 'E').

    Returns:
        A tuple of (state type, index).

    Example:
        >>> parse_state('M3')
        ('M', 3)
        >>> parse_state('S')
        ('S', 0)
        >>> parse_state('E')
        ('E', -1)
    """
    if state == "S":
        return ("S", 0)
    if state == "E":
        return ("E", -1)
    return state[0], int(state[1:])


# @lru_cache(maxsize=None), Memoization here unnecessarily uses more memory without an execution time speedup
def check_transition(s1: str, s2: str, end_idx: int) -> bool:
    """Checks if a transition from s1 to s2 is valid in the HMM.

    Args:
        s1: The source state.
        s2: The target state.
        end_idx: The last match state index.

    Returns:
        True if the transition is valid, False otherwise.

    Example:
        >>> check_transition('S', 'M1', 3)
        True
        >>> check_transition('M1', 'E', 3)
        False
    """
    t1: str
    i1: int
    t2: str
    i2: int
    t1, i1 = parse_state(s1)
    t2, i2 = parse_state(s2)

    if t2 == "E":
        return s1 in {f"M{end_idx}", f"I{end_idx}", f"D{end_idx}"}
    if t1 == "E" or t2 == "S":
        return False

    rules: Dict[str, set[Tuple[str, int]]] = {
        "S": {("I", 0), ("M", 1), ("D", 1)},
        "M": {("M", i1 + 1), ("D", i1 + 1), ("I", i1)},
        "I": {("M", i1 + 1), ("D", i1 + 1), ("I", i1)},
        "D": {("M", i1 + 1), ("D", i1 + 1), ("I", i1)},
    }
    return (t2, i2) in rules[t1]


def round_matrix(matrix: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Rounds all positive values in a nested dictionary to 10 decimal places.

    Args:
        matrix: The nested dictionary to round.

    Returns:
        A new dictionary with rounded values.

    Example:
        >>> round_matrix({'A': {'B': 0.12345678901, 'C': 0.0}})
        {'A': {'B': 0.123456789}}
    """
    return {
        state1: {state2: round(v, 10) for state2, v in row.items() if v > 0}
        for state1, row in matrix.items()
    }


def get_probabilities(
    counts: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Converts counts to probabilities by normalizing over the sum of values for each key.

    Args:
        counts: A nested dictionary of counts.

    Returns:
        A nested dictionary of probabilities.

    Example:
        >>> get_probabilities({'A': {'B': 2, 'C': 2}})
        {'A': {'B': 0.5, 'C': 0.5}}
    """
    result: Dict[str, Dict[str, float]] = {}
    for key, value in counts.items():
        total: float = sum(value.values())
        result[key] = {
            other_key: (other_value / total if total else 0.0)
            for other_key, other_value in value.items()
        }
    return result


def get_all_states(num_match_states: int) -> List[str]:
    """Generates all HMM states for a given number of match states.

    Args:
        num_match_states: The number of match states.

    Returns:d
        A list of all state names.

    Example:
        >>> get_all_states(2)
        ['S', 'I0', 'M1', 'D1', 'I1', 'M2', 'D2', 'I2', 'E']
    """
    states: List[str] = ["S", "I0"]
    for i in range(1, num_match_states + 1):
        states.extend([f"M{i}", f"D{i}", f"I{i}"])
    states.append("E")
    return states


def apply_pseudocounts(
    transitions_probs: Dict[str, Dict[str, float]],
    emissions_probs: Dict[str, Dict[str, float]],
    sigma: float,
    states: List[str],
    alphabet: str,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Applies pseudocounts to transition and emission probability matrices.

    Args:
        transitions_probs: The transition probability matrix.
        emissions_probs: The emission probability matrix.
        sigma: The pseudocount value.
        states: List of all states.
        alphabet: The alphabet of symbols.

    Returns:
        The transition and emission matrices with pseudocounts applied.

    Example:
        >>> transitions_probs = {'S': {'M1': 0.5, 'I0': 0.5}, 'M1': {'E': 1.0}, 'I0': {}, 'E': {}}
        >>> emissions_probs = {'M1': {'A': 1.0, 'C': 0.0}, 'I0': {'A': 0.0, 'C': 0.0}}
        >>> apply_pseudocounts(transitions_probs, emissions_probs, 1.0, ['S', 'M1', 'I0', 'E'], 'AC')
        ({'S': {'M1': 0.5, 'I0': 0.5}, 'M1': {'E': 1.0}, 'I0': {}, 'E': {}}, {'M1': {'A': 0.6666666666666666, 'C': 0.3333333333333333}, 'I0': {'A': 0.5, 'C': 0.5}})
    """
    transition_matrix: Dict[str, Dict[str, float]] = {}
    emission_matrix: Dict[str, Dict[str, float]] = {}

    for s in states:
        valid_targets: List[str] = list(transitions_probs[s].keys())
        values: List[float] = [transitions_probs[s].get(t, 0.0) for t in valid_targets]
        total: float = sum(values) + sigma * len(valid_targets)
        transition_matrix[s] = {
            t: (transitions_probs[s].get(t, 0.0) + sigma) / total for t in valid_targets
        }

    for key in emissions_probs.keys():
        values = [emissions_probs[key][a] for a in alphabet]
        total = sum(values) + sigma * len(alphabet)
        emission_matrix[key] = {
            a: (emissions_probs[key].get(a, 0.0) + sigma) / total for a in alphabet
        }

    return transition_matrix, emission_matrix


# To improve the maintainability and readability of the code i disabled these warnings
# pylint: disable=too-many-branches, too-many-locals
def profile_HMM_pseudocounts(
    threshold: float, sigma: float, alphabet: str, patterns: List[str]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], int]:
    """Builds a profile HMM with pseudocounts from a multiple sequence alignment.

    Args:
        threshold: Gap threshold for match states.
        sigma: Pseudocount value for smoothing probabilities.
        alphabet: The alphabet of allowed symbols.
        patterns: List of aligned sequences.

    Returns:
        Tuple of (transition matrix, emission matrix, number of match states).

    Example:
        >>> t, e, n = profile_HMM_pseudocounts(0.5, 1.0, 'AC', ['A-', 'CA'])
        >>> sorted(t.keys())
        ['D1', 'E', 'I0', 'I1', 'M1', 'S']
        >>> sorted(e.keys())
        ['I0', 'I1', 'M1']
        >>> n
        1
    """
    n: int = len(patterns)
    pattern_length: int = len(patterns[0])

    match_columns: List[bool] = []
    i: int
    for i in range(pattern_length):
        gap_count: int = 0
        pattern: str
        for pattern in patterns:
            if pattern[i] == "-":
                gap_count += 1
        match_columns.append(gap_count / n < threshold)

    match_idx: List[int] = [i for i, is_match in enumerate(match_columns) if is_match]
    num_match_states: int = len(match_idx)

    states: List[str] = get_all_states(num_match_states)

    transition_counts: Dict[str, Dict[str, float]] = {
        s: {t: 0.0 for t in states if check_transition(s, t, num_match_states)}
        for s in states
    }
    emission_counts: Dict[str, Dict[str, float]] = {
        s: {a: 0.0 for a in alphabet} for s in states if s[0] in ("M", "I")
    }

    for pattern in patterns:
        state_path: List[str] = ["S"]
        emission_path: List[Tuple[str, str]] = []
        match_ptr: int = 1
        for i, symbol in enumerate(pattern):
            if match_columns[i]:
                if symbol == "-":
                    state_path.append(f"D{match_ptr}")
                else:
                    state_path.append(f"M{match_ptr}")
                    emission_path.append((f"M{match_ptr}", symbol))
                match_ptr += 1
            elif symbol != "-":
                state_path.append(f"I{match_ptr - 1}")
                emission_path.append((f"I{match_ptr - 1}", symbol))
        state_path.append("E")

        for s1, s2 in zip(state_path, state_path[1:]):
            transition_counts[s1][s2] += 1.0

        for state, symbol in emission_path:
            emission_counts[state][symbol] += 1.0

    transitions_probs: Dict[str, Dict[str, float]] = get_probabilities(
        transition_counts
    )
    emissions_probs: Dict[str, Dict[str, float]] = get_probabilities(emission_counts)

    transition_matrix, emission_matrix = apply_pseudocounts(
        transitions_probs, emissions_probs, sigma, states, alphabet
    )

    return (
        round_matrix(transition_matrix),
        round_matrix(emission_matrix),
        num_match_states,
    )


def log_prob(p: float) -> float:
    """Returns the log probability, or a very small number if p is zero.

    Args:
        p: The probability value.

    Returns:
        The log of p, or MIN_FLOAT if p is zero.

    Example:
        >>> round(log_prob(0.5), 5)
        -0.69315
        >>> log_prob(0)
        -1.7976931348623157e+308
    """
    return math.log(p) if p > 0 else MIN_FLOAT


def propagate_silent(
    V: Dict[str, List[float]],
    back: Dict[str, List[Optional[str]]],
    col: int,
    t_matrix: Dict[str, Dict[str, float]],
) -> None:
    """Propagates probabilities through silent states in the same column until convergence.

    Args:
        V: The dynamic programming matrix.
        back: The backtracking matrix.
        col: The current column index.
        t_matrix: The transition probability matrix.

    Returns:
        None. Updates V and back in place.

    Example:
        >>> V = {'S': [0, -1e308], 'D1': [-1e308, -1e308]}
        >>> back = {'S': [None, None], 'D1': [None, None]}
        >>> t_matrix = {'S': {'D1': 1.0}, 'D1': {}}
        >>> propagate_silent(V, back, 0, t_matrix)
        >>> V['D1'][0] > -1e308
        True
    """
    changed: bool = True
    while changed:
        changed = False
        for prev in V:
            for s, trans_prob in t_matrix.get(prev, {}).items():
                if s[0] not in ("M", "I") and trans_prob != 0.0:
                    prob: float = V[prev][col] + log_prob(trans_prob)
                    if prob > V[s][col]:
                        V[s][col] = prob
                        back[s][col] = prev
                        changed = True


def backtrack_path(back: Dict[str, List[Optional[str]]], s: str, n: int) -> List[str]:
    """Reconstructs the most probable hidden state path from a backtracking table.

    Args:
        back: A dictionary mapping states to lists of their predecessor states.
        s: The state to start backtracking from (usually the final state).
        Text: The observed sequence.
        n: The length of the observed sequence.

    Returns:
        The reconstructed sequence of hidden states from 'S' to the final state.

    Example:
        >>> back = {"M1": [None, "S"], "S": [None, None]}
        >>> backtrack_path(back, "M1", 1)
        ['M1']
    """
    path: List[str] = []
    col: int = n
    state: str = s

    while state != "S":
        if state[0] in ("M", "I", "D"):
            path.append(state)
        prev: Optional[str] = back[state][col]
        if prev is None:
            break
        if state[0] in ("M", "I"):
            col -= 1
        state = prev

    return list(reversed(path))


# To improve the maintainability and readability of the code i disabled these warnings
# pylint: disable=too-many-branches, too-many-locals
def profile_HMM_sequence_alignment_v1(
    Text: str, theta: float, sigma: float, alphabet: str, Alignment: List[str] | str
) -> List[str]:
    """Aligns a sequence to a profile HMM using an adapted Viterbi algorithm.
    Builds the HMM from a multiple sequence alignment and computes the most probable hidden state path.
    Handles match, insert, delete, and silent states with dynamic programming and backtracking.

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
    if isinstance(Alignment, str):
        Alignment = [str(rec.seq) for rec in SeqIO.parse(Alignment, "fasta")]

    t_matrix: Dict[str, Dict[str, float]]
    e_matrix: Dict[str, Dict[str, float]]
    L: int
    t_matrix, e_matrix, L = profile_HMM_pseudocounts(theta, sigma, alphabet, Alignment)

    states: List[str] = ["S", "I0"]
    for i in range(1, L + 1):
        states.extend([f"M{i}", f"D{i}", f"I{i}"])
    states.append("E")

    n: int = len(Text)
    V: Dict[str, List[float]] = {s: [MIN_FLOAT] * (n + 1) for s in states}
    back: Dict[str, List[Optional[str]]] = {s: [None] * (n + 1) for s in states}
    V["S"][0] = 0.0

    propagate_silent(V, back, 0, t_matrix)

    for i in range(1, n + 1):
        obs: str = Text[i - 1]
        for s in states:
            if s[0] not in ("M", "I"):
                continue
            best_val: float = MIN_FLOAT
            best_prev: Optional[str] = None
            emit_prob: float = e_matrix.get(s, {}).get(obs, 0.0)
            if emit_prob == 0.0:
                continue
            emit_log: float = log_prob(emit_prob)
            for prev in states:
                if s in t_matrix.get(prev, {}):
                    prev_col: int = i - 1 if s[0] in ("M", "I") else i
                    prob: float = (
                        V[prev][prev_col] + log_prob(t_matrix[prev][s]) + emit_log
                    )
                    if prob > best_val:
                        best_val = prob
                        best_prev = prev
            if best_prev is not None:
                V[s][i] = best_val
                back[s][i] = best_prev
        propagate_silent(V, back, i, t_matrix)

    best_score: float = MIN_FLOAT
    best_final: Optional[str] = None
    for s in states:
        if "E" in t_matrix.get(s, {}):
            prob: float = V[s][n] + log_prob(t_matrix[s]["E"])
            if prob > best_score:
                best_score = prob
                best_final = s

    return backtrack_path(back, str(best_final), n)


# --- Versie 2 ---


def apply_pseudocounts_v2(
    transitions_probs: Dict[str, Dict[str, float]],
    emissions_probs: Dict[str, Dict[str, float]],
    sigma: float,
    states: List[str],
    alphabet: str,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Applies pseudocounts to transition and emission probability matrices.
    CHANGELOG:
        - Replaced Dict.get() statements with regular lookups

    Example:
        >>> transitions_probs = {'S': {'M1': 0.5, 'I0': 0.5}, 'M1': {'E': 1.0}, 'I0': {}, 'E': {}}
        >>> emissions_probs = {'M1': {'A': 1.0, 'C': 0.0}, 'I0': {'A': 0.0, 'C': 0.0}}
        >>> apply_pseudocounts_v2(transitions_probs, emissions_probs, 1.0, ['S', 'M1', 'I0', 'E'], 'AC')
        ({'S': {'M1': 0.5, 'I0': 0.5}, 'M1': {'E': 1.0}, 'I0': {}, 'E': {}}, {'M1': {'A': 0.6666666666666666, 'C': 0.3333333333333333}, 'I0': {'A': 0.5, 'C': 0.5}})
    """
    transition_matrix: Dict[str, Dict[str, float]] = {}
    emission_matrix: Dict[str, Dict[str, float]] = {}

    for s in states:
        valid_targets: List[str] = list(transitions_probs[s].keys())
        values: List[float] = [transitions_probs[s][t] for t in valid_targets]
        total: float = sum(values) + sigma * len(valid_targets)
        transition_matrix[s] = {
            t: (transitions_probs[s][t] + sigma) / total for t in valid_targets
        }

    for key in emissions_probs.keys():
        values = [emissions_probs[key][a] for a in alphabet]
        total = sum(values) + sigma * len(alphabet)
        emission_matrix[key] = {
            a: (emissions_probs[key][a] + sigma) / total for a in alphabet
        }

    return transition_matrix, emission_matrix


# To improve the maintainability and readability of the code i disabled these warnings
# pylint: disable=too-many-branches, too-many-locals
def profile_HMM_pseudocounts_v2(
    threshold: float, sigma: float, alphabet: str, patterns: List[str]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], int]:
    """Builds a profile HMM with pseudocounts from a multiple sequence alignment.
    CHANGELOG:
        -Changed function apply_pseudocounts to use second version

    Example:
        >>> t, e, n = profile_HMM_pseudocounts_v2(0.5, 1.0, 'AC', ['A-', 'CA'])
        >>> sorted(t.keys())
        ['D1', 'E', 'I0', 'I1', 'M1', 'S']
        >>> sorted(e.keys())
        ['I0', 'I1', 'M1']
        >>> n
        1
    """
    n: int = len(patterns)
    pattern_length: int = len(patterns[0])

    match_columns: List[bool] = []
    for i in range(pattern_length):
        gap_count: int = 0
        for pattern in patterns:
            if pattern[i] == "-":
                gap_count += 1
        match_columns.append(gap_count / n < threshold)

    match_idx: List[int] = [i for i, is_match in enumerate(match_columns) if is_match]
    num_match_states: int = len(match_idx)

    states: List[str] = get_all_states(num_match_states)

    transition_counts: Dict[str, Dict[str, float]] = {
        s: {t: 0.0 for t in states if check_transition(s, t, num_match_states)}
        for s in states
    }
    emission_counts: Dict[str, Dict[str, float]] = {
        s: {a: 0.0 for a in alphabet} for s in states if s[0] in ("M", "I")
    }

    for pattern in patterns:
        state_path: List[str] = ["S"]
        emission_path: List[Tuple[str, str]] = []
        match_ptr: int = 1
        for i, symbol in enumerate(pattern):
            if match_columns[i]:
                if symbol == "-":
                    state_path.append(f"D{match_ptr}")
                else:
                    state_path.append(f"M{match_ptr}")
                    emission_path.append((f"M{match_ptr}", symbol))
                match_ptr += 1
            elif symbol != "-":
                state_path.append(f"I{match_ptr - 1}")
                emission_path.append((f"I{match_ptr - 1}", symbol))
        state_path.append("E")

        for s1, s2 in zip(state_path, state_path[1:]):
            transition_counts[s1][s2] += 1.0

        for state, symbol in emission_path:
            emission_counts[state][symbol] += 1.0

    transitions_probs: Dict[str, Dict[str, float]] = get_probabilities(
        transition_counts
    )
    emissions_probs: Dict[str, Dict[str, float]] = get_probabilities(emission_counts)

    transition_matrix, emission_matrix = apply_pseudocounts_v2(
        transitions_probs, emissions_probs, sigma, states, alphabet
    )

    return (
        round_matrix(transition_matrix),
        round_matrix(emission_matrix),
        num_match_states,
    )


def propagate_silent_v2(
    V: Dict[str, List[float]],
    back: Dict[str, List[Optional[str]]],
    col: int,
    t_matrix: Dict[str, Dict[str, float]],
) -> None:
    """CHANGELOG:
        - Replaced Dict.get() statements with regular lookups

    Example:
        >>> V = {'S': [0, -1e308], 'D1': [-1e308, -1e308]}
        >>> back = {'S': [None, None], 'D1': [None, None]}
        >>> t_matrix = {'S': {'D1': 1.0}, 'D1': {}}
        >>> propagate_silent_v2(V, back, 0, t_matrix)
        >>> V['D1'][0] > -1e308
        False
    """
    changed: bool = True
    while changed:
        changed = False
        for prev in V:
            for s, trans_prob in t_matrix[prev].items():
                if not s[0] not in ("M", "I") and trans_prob != 0.0:
                    prob: float = V[prev][col] + log_prob(trans_prob)
                    if prob > V[s][col]:
                        V[s][col] = prob
                        back[s][col] = prev
                        changed = True


# To improve the maintainability and readability of the code i disabled these warnings
# pylint: disable=too-many-branches, too-many-locals
def profile_HMM_sequence_alignment_v2(
    Text: str, theta: float, sigma: float, alphabet: str, Alignment: List[str] | str
) -> List[str]:
    """CHANGELOG:
    - Replaced Dict.get() statements with regular lookups

    Example:
        >>> profile_HMM_sequence_alignment_v2('C', 0.2, 0.01, 'ABC', ['B---B', 'CAACC', 'CC-AB', 'B-BCB', 'CBBBB'])
        ['M1', 'I1', 'M2']
    """
    if isinstance(Alignment, str):
        Alignment = [str(rec.seq) for rec in SeqIO.parse(Alignment, "fasta")]

    t_matrix: Dict[str, Dict[str, float]]
    e_matrix: Dict[str, Dict[str, float]]
    L: int
    t_matrix, e_matrix, L = profile_HMM_pseudocounts_v2(
        theta, sigma, alphabet, Alignment
    )

    states: List[str] = ["S", "I0"]
    for i in range(1, L + 1):
        states.extend([f"M{i}", f"D{i}", f"I{i}"])
    states.append("E")

    n: int = len(Text)
    V: Dict[str, List[float]] = {s: [MIN_FLOAT] * (n + 1) for s in states}
    back: Dict[str, List[Optional[str]]] = {s: [None] * (n + 1) for s in states}
    V["S"][0] = 0.0

    propagate_silent_v2(V, back, 0, t_matrix)

    for i in range(1, n + 1):
        obs: str = Text[i - 1]
        for s in states:
            if s[0] not in ("M", "I"):
                continue
            best_val: float = MIN_FLOAT
            best_prev: Optional[str] = None
            emit_prob: float = e_matrix[s][obs]
            if emit_prob == 0.0:
                continue
            emit_log: float = log_prob(emit_prob)
            for prev in states:
                if s in t_matrix[prev]:
                    prev_col: int = i - 1 if s[0] in ("M", "I") else i
                    prob: float = (
                        V[prev][prev_col] + log_prob(t_matrix[prev][s]) + emit_log
                    )
                    if prob > best_val:
                        best_val = prob
                        best_prev = prev
            if best_prev is not None:
                V[s][i] = best_val
                back[s][i] = best_prev
        propagate_silent_v2(V, back, i, t_matrix)

    best_score: float = MIN_FLOAT
    best_final: Optional[str] = None
    for s in states:
        if "E" in t_matrix[s]:
            prob: float = V[s][n] + log_prob(t_matrix[s]["E"])
            if prob > best_score:
                best_score = prob
                best_final = s

    return backtrack_path(back, str(best_final), n)


# --- Versie 3 ---


def round_matrix_v3(matrix: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Rounds all positive values in a nested dictionary to 10 decimal places.
    CHANGELOG:
        - Precompute log_prob once

    Args:
        matrix: The nested dictionary to round.

    Returns:
        A new dictionary with rounded values.

    Example:
        >>> round_matrix({'A': {'B': 0.12345678901, 'C': 0.0}})
        {'A': {'B': 0.123456789}}
    """
    return {
        state1: {state2: log_prob(round(v, 10)) for state2, v in row.items() if v > 0}
        for state1, row in matrix.items()
    }


# To improve the maintainability and readability of the code i disabled these warnings
# pylint: disable=too-many-branches, too-many-locals
def profile_HMM_pseudocounts_v3(
    threshold: float, sigma: float, alphabet: str, patterns: List[str]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], int]:
    """Builds a profile HMM with pseudocounts from a multiple sequence alignment.
    CHANGELOG:
        -Changed function apply_pseudocounts to use second version
        -Changed function round_matrix to use third version

    Args:
        threshold: Gap threshold for match states.
        sigma: Pseudocount value for smoothing probabilities.
        alphabet: The alphabet of allowed symbols.
        patterns: List of aligned sequences.

    Returns:
        Tuple of (transition matrix, emission matrix, number of match states).

    Example:
        >>> t, e, n = profile_HMM_pseudocounts(0.5, 1.0, 'AC', ['A-', 'CA'])
        >>> sorted(t.keys())
        ['D1', 'E', 'I0', 'I1', 'M1', 'S']
        >>> sorted(e.keys())
        ['I0', 'I1', 'M1']
        >>> n
        1
    """
    n: int = len(patterns)
    pattern_length: int = len(patterns[0])

    match_columns: List[bool] = []
    for i in range(pattern_length):
        gap_count: int = 0
        for pattern in patterns:
            if pattern[i] == "-":
                gap_count += 1
        match_columns.append(gap_count / n < threshold)

    match_idx: List[int] = [i for i, is_match in enumerate(match_columns) if is_match]
    num_match_states: int = len(match_idx)

    states: List[str] = get_all_states(num_match_states)

    transition_counts: Dict[str, Dict[str, float]] = {
        s: {t: 0.0 for t in states if check_transition(s, t, num_match_states)}
        for s in states
    }
    emission_counts: Dict[str, Dict[str, float]] = {
        s: {a: 0.0 for a in alphabet} for s in states if s[0] in ("M", "I")
    }

    for pattern in patterns:
        state_path: List[str] = ["S"]
        emission_path: List[Tuple[str, str]] = []
        match_ptr: int = 1
        for i, symbol in enumerate(pattern):
            if match_columns[i]:
                if symbol == "-":
                    state_path.append(f"D{match_ptr}")
                else:
                    state_path.append(f"M{match_ptr}")
                    emission_path.append((f"M{match_ptr}", symbol))
                match_ptr += 1
            elif symbol != "-":
                state_path.append(f"I{match_ptr - 1}")
                emission_path.append((f"I{match_ptr - 1}", symbol))
        state_path.append("E")

        for s1, s2 in zip(state_path, state_path[1:]):
            transition_counts[s1][s2] += 1.0

        for state, symbol in emission_path:
            emission_counts[state][symbol] += 1.0

    transitions_probs: Dict[str, Dict[str, float]] = get_probabilities(
        transition_counts
    )
    emissions_probs: Dict[str, Dict[str, float]] = get_probabilities(emission_counts)

    transition_matrix, emission_matrix = apply_pseudocounts_v2(
        transitions_probs, emissions_probs, sigma, states, alphabet
    )

    return (
        round_matrix_v3(transition_matrix),
        round_matrix_v3(emission_matrix),
        num_match_states,
    )


def propagate_silent_v3(
    V: Dict[str, List[float]],
    back: Dict[str, List[Optional[str]]],
    col: int,
    t_matrix: Dict[str, Dict[str, float]],
) -> None:
    """Propagates probabilities through silent states in the same column until convergence.
    CHANGELOG:
        - log_prob has already been precomputed so here we changed the function call with a regular lookup
    Args:
        V: The dynamic programming matrix.
        back: The backtracking matrix.
        col: The current column index.
        t_matrix: The transition probability matrix.

    Returns:
        None. Updates V and back in place.

    Example:
        >>> V = {'S': [0, -1e308], 'D1': [-1e308, -1e308]}
        >>> back = {'S': [None, None], 'D1': [None, None]}
        >>> t_matrix = {'S': {'D1': 1.0}, 'D1': {}}
        >>> propagate_silent(V, back, 0, t_matrix)
        >>> V['D1'][0] > -1e308
        True
    """
    changed: bool = True
    while changed:
        changed = False
        for prev in V:
            for s, trans_prob in t_matrix[prev].items():
                if s[0] not in ("M", "I"):
                    prob: float = V[prev][col] + trans_prob
                    if prob > V[s][col]:
                        V[s][col] = prob
                        back[s][col] = prev
                        changed = True


# To improve the maintainability and readability of the code i disabled these warnings
# pylint: disable=too-many-branches, too-many-locals
def profile_HMM_sequence_alignment(
    Text: str, theta: float, sigma: float, alphabet: str, Alignment: List[str] | str
) -> List[str]:
    """Aligns a sequence to a profile HMM using an adapted Viterbi algorithm.
    Builds the HMM from a multiple sequence alignment and computes the most probable hidden state path.
    Handles match, insert, delete, and silent states with dynamic programming and backtracking.
    CHANGELOG:
        - log_prob has already been precomputed so here we changed the function call with a regular lookup
        - Precomputed a transition_mask to reduce the inner for loop in the DP loop

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
    if isinstance(Alignment, str):
        Alignment = [str(rec.seq) for rec in SeqIO.parse(Alignment, "fasta")]

    t_matrix: Dict[str, Dict[str, float]]
    e_matrix: Dict[str, Dict[str, float]]
    L: int
    t_matrix, e_matrix, L = profile_HMM_pseudocounts_v3(
        theta, sigma, alphabet, Alignment
    )

    states: List[str] = ["S", "I0"]
    for i in range(1, L + 1):
        states.extend([f"M{i}", f"D{i}", f"I{i}"])
    states.append("E")

    n: int = len(Text)
    V: Dict[str, List[float]] = {s: [MIN_FLOAT] * (n + 1) for s in states}
    back: Dict[str, List[Optional[str]]] = {s: [None] * (n + 1) for s in states}
    V["S"][0] = 0.0
    transition_mask: Dict[str, List[str]] = {
        s: [t for t in states if check_transition(t, s, L)] for s in states
    }

    propagate_silent_v3(V, back, 0, t_matrix)

    for i in range(1, n + 1):
        obs: str = Text[i - 1]
        for s in states:
            if s[0] in ("M", "I"):
                best_val: float = MIN_FLOAT
                best_prev: Optional[str] = None
                emit_prob: float = e_matrix[s][obs]
                for prev in transition_mask[s]:
                    prev_col: int = i - 1 if s[0] in ("M", "I") else i
                    prob: float = V[prev][prev_col] + t_matrix[prev][s] + emit_prob
                    if prob > best_val:
                        best_val = prob
                        best_prev = prev
                if best_prev is not None:
                    V[s][i] = best_val
                    back[s][i] = best_prev
        propagate_silent_v3(V, back, i, t_matrix)

    best_score: float = MIN_FLOAT
    best_final: Optional[str] = None
    for s in states:
        if "E" in t_matrix[s]:
            prob: float = V[s][n] + t_matrix[s]["E"]
            if prob > best_score:
                best_score = prob
                best_final = s

    return backtrack_path(back, str(best_final), n)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    parser = argparse.ArgumentParser(description="Profile HMM sequence alignment")
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        required=True,
        help="Gap threshold for match states (0–1).",
    )
    parser.add_argument(
        "--pseudo",
        "-p",
        type=float,
        required=True,
        help="Pseudocount sigma for smoothing probabilities.",
    )
    parser.add_argument(
        "--alphabet",
        "-a",
        type=str,
        required=True,
        help="Alphabet of allowed symbols (e.g., 'ACGT').",
    )
    parser.add_argument(
        "--alignment",
        "-A",
        type=str,
        required=True,
        help="Multiple sequence alignment in FASTA format.",
    )
    parser.add_argument(
        "--sequence",
        "-s",
        type=str,
        required=True,
        help="Sequence to align against the profile HMM.",
    )

    args = parser.parse_args()

    # Load alignment
    alignment = [str(rec.seq) for rec in SeqIO.parse(args.alignment, "fasta")]

    # Run alignment
    path = profile_HMM_sequence_alignment(
        Text=args.sequence,
        theta=args.threshold,
        sigma=args.pseudo,
        alphabet=args.alphabet,
        Alignment=alignment,
    )

    # Print result
    print("\n".join(path))
