"""
ACUITY Framework — String Similarity Utilities

Pure-Python Levenshtein distance and ratio computation used by
the verification module for fuzzy business name matching.
"""
from __future__ import annotations


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Calculate the Levenshtein similarity ratio between two strings.

    Returns a value in [0, 1] where 1.0 means the strings are identical.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Similarity ratio (1 − normalised_edit_distance).
    """
    if not s1 or not s2:
        return 0.0

    rows = len(s1) + 1
    cols = len(s2) + 1
    distance = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(1, rows):
        distance[i][0] = i
    for k in range(1, cols):
        distance[0][k] = k

    for col in range(1, cols):
        for row in range(1, rows):
            cost = 0 if s1[row - 1] == s2[col - 1] else 1
            distance[row][col] = min(
                distance[row - 1][col] + 1,      # Deletion
                distance[row][col - 1] + 1,      # Insertion
                distance[row - 1][col - 1] + cost  # Substitution
            )

    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - (distance[len(s1)][len(s2)] / max_len)


def levenshtein_details(s1: str, s2: str) -> dict:
    """Calculate the Levenshtein ratio with detailed breakdown.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Dictionary with keys: ``score``, ``edits``, ``max_len``.
    """
    if not s1 or not s2:
        return {"score": 0.0, "edits": 0, "max_len": 0}

    rows = len(s1) + 1
    cols = len(s2) + 1
    distance = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(1, rows):
        distance[i][0] = i
    for k in range(1, cols):
        distance[0][k] = k

    for col in range(1, cols):
        for row in range(1, rows):
            cost = 0 if s1[row - 1] == s2[col - 1] else 1
            distance[row][col] = min(
                distance[row - 1][col] + 1,
                distance[row][col - 1] + 1,
                distance[row - 1][col - 1] + cost
            )

    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return {"score": 1.0, "edits": 0, "max_len": 0}
    edits = distance[len(s1)][len(s2)]
    return {"score": 1.0 - (edits / max_len), "edits": edits, "max_len": max_len}

import re

def _tokenize(s: str) -> list[str]:
    return re.findall(r'\w+', str(s).lower())

def token_sort_ratio(s1: str, s2: str) -> float:
    t1 = _tokenize(s1)
    t2 = _tokenize(s2)
    t1.sort()
    t2.sort()
    return levenshtein_ratio(' '.join(t1), ' '.join(t2))

def token_set_ratio(s1: str, s2: str) -> float:
    t1 = set(_tokenize(s1))
    t2 = set(_tokenize(s2))
    
    intersection = sorted(list(t1.intersection(t2)))
    diff1 = sorted(list(t1.difference(t2)))
    diff2 = sorted(list(t2.difference(t1)))
    
    str_intersection = ' '.join(intersection)
    str1 = ' '.join(intersection + diff1).strip()
    str2 = ' '.join(intersection + diff2).strip()
    
    score1 = levenshtein_ratio(str1, str2)
    score2 = levenshtein_ratio(str_intersection, str1) if str_intersection else 0.0
    score3 = levenshtein_ratio(str_intersection, str2) if str_intersection else 0.0
    
    return max(score1, score2, score3)

def hybrid_fuzzy_match(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
        
    plain_score = levenshtein_ratio(s1.lower(), s2.lower())
    sort_score = token_sort_ratio(s1, s2)
    set_score = token_set_ratio(s1, s2)
    
    # Apply penalty to Token-Set if length disparity is massive (to prevent short acronym false positives)
    len1, len2 = len(s1), len(s2)
    if len1 > 0 and len2 > 0:
        ratio = min(len1, len2) / max(len1, len2)
        if ratio < 0.35:
            # Heavily penalize the Token-Set score
            set_score = set_score * ratio
            
    return max(plain_score, sort_score, set_score)


# ---------------------------------------------------------------------------
# Optimised matching utilities (ported from production bplo_service.py)
# ---------------------------------------------------------------------------

def fast_levenshtein_ratio(s1: str, s2: str, threshold: float) -> float:
    """Early-pruning Levenshtein ratio that aborts when *threshold* is unreachable.

    Uses a two-row dynamic programming approach with a per-row minimum
    check.  If the smallest edit distance achievable in the current row
    already exceeds the maximum number of edits allowed by *threshold*,
    computation stops immediately and ``0.0`` is returned.

    This is significantly faster than the standard implementation for
    batch scenarios where most candidate pairs will be below the
    threshold.

    Args:
        s1: First string (typically the extracted business name).
        s2: Second string (typically a registry entry).
        threshold: Minimum similarity ratio required.  If the pair
            cannot reach this ratio, ``0.0`` is returned early.

    Returns:
        Similarity ratio in ``[0, 1]``, or ``0.0`` if the threshold
        cannot be met.
    """
    if not s1 or not s2:
        return 0.0

    rows = len(s1) + 1
    cols = len(s2) + 1
    max_len = max(len(s1), len(s2))
    max_allowed_edits = int((1.0 - threshold) * max_len)

    prev = list(range(cols))
    curr = [0] * cols

    for row in range(1, rows):
        curr[0] = row
        min_in_row = row
        char_s1 = s1[row - 1]
        for col in range(1, cols):
            cost = 0 if char_s1 == s2[col - 1] else 1
            val = min(
                curr[col - 1] + 1,
                prev[col] + 1,
                prev[col - 1] + cost,
            )
            curr[col] = val
            if val < min_in_row:
                min_in_row = val
        if min_in_row > max_allowed_edits:
            return 0.0
        prev, curr = curr, prev

    return 1.0 - (prev[-1] / max_len)


def pre_tokenize_sort(s: str) -> str:
    """Tokenize, lowercase, sort alphabetically, and rejoin.

    Useful for order-independent comparison of business names
    (e.g. ``"bakeshop juan"`` vs ``"juan bakeshop"``).

    Args:
        s: Input string.

    Returns:
        Lowercased, sorted, space-joined token string.
    """
    tokens = re.findall(r'\w+', str(s).lower())
    tokens.sort()
    return ' '.join(tokens)


def multi_stage_match(s1: str, s2: str, threshold: float) -> float:
    """Multi-stage heuristic matching pipeline for fast verification.

    Applies progressively more expensive checks, bailing out early
    when the candidate pair clearly cannot meet *threshold*:

    1. **Length-ratio screening** — if the theoretical maximum score
       (based purely on string-length difference) is below *threshold*,
       return ``0.0`` immediately.
    2. **SequenceMatcher heuristic** — ``difflib.SequenceMatcher``
       uses a C-optimised heuristic.  If the quick ratio is well below
       the threshold (with a 0.15 grace margin), skip the expensive DP.
    3. **Fast Levenshtein** — full edit-distance computation with
       early row-pruning via :func:`fast_levenshtein_ratio`.

    This mirrors the production verification pipeline in the ACUITY
    system and is designed for batch scenarios with hundreds of
    registry entries.

    Args:
        s1: First string (typically pre-tokenized and sorted).
        s2: Second string (typically pre-tokenized and sorted).
        threshold: Minimum similarity ratio to accept.

    Returns:
        Similarity ratio in ``[0, 1]``, or ``0.0`` if the threshold
        cannot be met.
    """
    import difflib

    if not s1 or not s2:
        return 0.0

    # Stage 1: Length-ratio pre-screening
    max_len = max(len(s1), len(s2))
    if max_len > 0:
        max_possible_score = 1.0 - (abs(len(s1) - len(s2)) / max_len)
        if max_possible_score < threshold:
            return 0.0

    # Stage 2: Fast C-optimised heuristic (SequenceMatcher)
    fast_heuristic = difflib.SequenceMatcher(None, s1, s2).ratio()
    if fast_heuristic < threshold - 0.15:
        return 0.0

    # Stage 3: Full DP with early pruning
    return fast_levenshtein_ratio(s1, s2, threshold)
