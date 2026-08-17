"""
ACUITY Framework — Combined Ranker

Merges textual relevance scores and proximity distances into a single
ranking, using a configurable weighting scheme.
"""
from __future__ import annotations


def rank_results(
    profiles: list[dict],
    cosine_scores: list[float],
    distances: list[float | None],
    relevance_weight: float = 0.6,
    proximity_weight: float = 0.4,
    top_k: int = 10,
) -> list[dict]:
    """Produce a ranked list of business profiles.

    The final score is a weighted combination of:
        - Textual relevance  (cosine similarity, higher = better)
        - Proximity score    (inverse distance, closer = better)

    Args:
        profiles: List of business profile dicts.
        cosine_scores: List of cosine similarity values per profile.
        distances: List of distances in km (``None`` if unavailable).
        relevance_weight: Weight for textual relevance [0, 1].
        proximity_weight: Weight for proximity [0, 1].
        top_k: How many results to return.

    Returns:
        Sorted list of profile dicts with added scoring fields:
        ``relevance_score``, ``distance_km``, ``proximity_score``, ``final_score``.
    """
    scored: list[dict] = []

    for i, profile in enumerate(profiles):
        relevance = float(cosine_scores[i])

        dist = distances[i]
        if dist is not None:
            # Inverse distance proximity score (1.0 = 0 km away)
            proximity_score = 1.0 / (1.0 + dist)
        else:
            proximity_score = 0.0

        final_score = (relevance_weight * relevance) + (proximity_weight * proximity_score)

        scored.append({
            **profile,
            "relevance_score": round(float(relevance), 4),
            "distance_km": round(float(dist), 2) if dist is not None else None,
            "proximity_score": round(float(proximity_score), 4),
            "final_score": round(float(final_score), 4),
        })

    # Sort descending by final score
    scored.sort(key=lambda x: x["final_score"], reverse=True)

    return list(scored[:top_k])
