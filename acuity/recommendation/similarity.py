"""
ACUITY Framework — Cosine Similarity

Computes relevance scores between a user query vector and the
TF-IDF business profile vectors using pure Python.
"""
from __future__ import annotations

import math


def compute_cosine_scores(
    tfidf_matrix: list[dict[str, float]],
    query_vec: list[dict[str, float]],
) -> list[float]:
    """Return a list of cosine similarity scores for every profile.

    Args:
        tfidf_matrix: List of sparse dictionary vectors representing profiles.
        query_vec: List containing one sparse dictionary vector for the user query.

    Returns:
        List of similarity scores in [0, 1].
    """
    if not query_vec:
        return [0.0] * len(tfidf_matrix)

    query = query_vec[0]
    query_mag = math.sqrt(sum(val ** 2 for val in query.values()))

    if query_mag == 0:
        return [0.0] * len(tfidf_matrix)

    scores = []
    for vector in tfidf_matrix:
        dot_product = 0.0
        for term, weight in query.items():
            if term in vector:
                dot_product += weight * vector[term]

        doc_mag = math.sqrt(sum(val ** 2 for val in vector.values()))

        if doc_mag == 0:
            scores.append(0.0)
        else:
            scores.append(dot_product / (query_mag * doc_mag))

    return scores
