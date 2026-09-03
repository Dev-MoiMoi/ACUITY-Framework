"""
ACUITY Framework — Recommendation Interfaces

Abstract base classes defining pluggable extension points for the
recommendation engine. Third-party code can implement these interfaces
to provide custom ranking strategies without modifying ACUITY's source.

Example:
    >>> from acuity.recommendation.interfaces import RankingStrategy
    >>> class MyRanking(RankingStrategy):
    ...     def compute_scores(self, profiles, query):
    ...         return [1.0] * len(profiles)  # everyone is relevant!
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class RankingStrategy(ABC):
    """Abstract interface for text-relevance ranking strategies.

    Implement this class to provide a custom ranking algorithm that can
    be injected into :class:`~acuity.recommendation.engine.RecommendationEngine`
    via its ``ranking_strategy`` constructor parameter.

    The strategy computes a relevance score for each business profile given
    a user query. These scores are then combined with proximity scores
    (Haversine distance — which is a fixed formula and not abstracted)
    to produce the final ranking.

    Example:
        >>> class BM25Ranking(RankingStrategy):
        ...     def compute_scores(self, profiles, query):
        ...         # Implement BM25 scoring
        ...         return scores
    """

    @abstractmethod
    def compute_scores(
        self,
        profiles: list[dict],
        query: str,
    ) -> list[float]:
        """Compute relevance scores for each profile against a query.

        Args:
            profiles: List of business profile dictionaries. Each dict
                may contain keys such as ``"name"``, ``"business_name"``,
                ``"description"``, ``"categories"``, and ``"services"``.
            query: The user's search query text.

        Returns:
            A list of float scores (higher = more relevant), one per
            profile, in the same order as the input list. Scores should
            typically be in the range [0, 1] but this is not strictly
            required.
        """
        ...
