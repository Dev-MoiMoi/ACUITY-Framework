"""
ACUITY Framework — Recommendation Engine

Main orchestrator that combines textual relevance (TF-IDF + cosine)
with geographic proximity (Haversine) to produce ranked results.

Usage:
    >>> from acuity.recommendation import RecommendationEngine
    >>> engine = RecommendationEngine()
    >>> engine.set_profiles([
    ...     {"name": "Juan's Bakery", "description": "Fresh bread and pastries daily"},
    ...     {"name": "JC Auto Repair", "description": "Vulcanizing and oil change"},
    ... ])
    >>> results = engine.recommend(query="bakery", user_lat=14.25, user_lon=121.10)
"""
from __future__ import annotations

import json

from .vectorizer import build_tfidf_matrix, transform_query
from .similarity import compute_cosine_scores
from .proximity import haversine_distance
from .ranker import rank_results
from .interfaces import RankingStrategy
from ..config import AcuityConfig


class RecommendationEngine:
    """End-to-end recommendation engine for ACUITY.

    Combines TF-IDF text similarity with Haversine geographic proximity
    to rank business profiles against user queries.

    Args:
        config: An ``AcuityConfig`` instance. If ``None``, uses defaults.
        relevance_weight: Override the config's relevance weight.
        proximity_weight: Override the config's proximity weight.
        ranking_strategy: An optional
            :class:`~acuity.recommendation.interfaces.RankingStrategy` instance.
            When provided, replaces the built-in TF-IDF + cosine similarity
            computation for text relevance scoring. When ``None`` (the default),
            the existing TF-IDF + cosine pipeline is used.
            Note: Haversine proximity scoring is a fixed, correct geographic
            formula and is not affected by this parameter.
    """

    def __init__(
        self,
        config: AcuityConfig | None = None,
        relevance_weight: float | None = None,
        proximity_weight: float | None = None,
        ranking_strategy: RankingStrategy | None = None,
    ):
        self.config = config or AcuityConfig()
        self.profiles: list[dict] = []
        self.relevance_weight = relevance_weight if relevance_weight is not None else self.config.relevance_weight
        self.proximity_weight = proximity_weight if proximity_weight is not None else self.config.proximity_weight
        self._tfidf_matrix = None
        self._vectorizer = None
        self._ranking_strategy = ranking_strategy

    def set_profiles(self, profiles: list[dict]) -> None:
        """Load business profiles from an in-memory list.

        Accepts profiles with any combination of the following keys:
        ``name``, ``business_name``, ``description``, ``categories``, ``services``.

        Args:
            profiles: List of business profile dictionaries.
        """
        self.profiles = profiles
        # Build TF-IDF matrix from profile text fields
        texts = []
        for p in self.profiles:
            name = p.get("name") or p.get("business_name") or ""
            desc = p.get("description") or ""
            cats = " ".join(p.get("categories") or [])
            srvs = " ".join(p.get("services") or [])

            texts.append(f"{name} {desc} {cats} {srvs}")
        self._vectorizer, self._tfidf_matrix = build_tfidf_matrix(texts)
        print(f"Loaded {len(self.profiles)} profiles. TF-IDF matrix built.")

    def load_profiles(self, path: str) -> None:
        """Load business profiles from a JSON file.

        Args:
            path: Path to a JSON file containing a list of profile dicts.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.set_profiles(data)

    def recommend(
        self,
        query: str,
        user_lat: float | None = None,
        user_lon: float | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """Return top-k ranked business profiles for *query*.

        Args:
            query: User's search query text.
            user_lat: User's latitude (optional, for proximity scoring).
            user_lon: User's longitude (optional, for proximity scoring).
            top_k: Number of results to return. Defaults to ``config.default_top_k``.

        Returns:
            List of profile dicts augmented with ``relevance_score``,
            ``distance_km``, ``proximity_score``, and ``final_score``.
        """
        if top_k is None:
            top_k = self.config.default_top_k

        if not self.profiles:
            return []

        # Textual relevance
        if self._ranking_strategy is not None:
            # Use injected ranking strategy
            cosine_scores = self._ranking_strategy.compute_scores(self.profiles, query)
        elif self._tfidf_matrix is not None and self._vectorizer is not None:
            # Existing TF-IDF + cosine similarity path (unchanged)
            query_vec = transform_query(self._vectorizer, query)
            cosine_scores = compute_cosine_scores(self._tfidf_matrix, query_vec)
        else:
            return []

        # Haversine proximity (if user location provided)
        # NOTE: Haversine distance is a fixed, correct geographic formula
        # with no legitimate variation — it is intentionally not abstracted.
        distances: list[float | None] = []
        if user_lat is not None and user_lon is not None:
            for profile in self.profiles:
                biz_lat = profile.get("latitude")
                biz_lon = profile.get("longitude")
                if biz_lat is not None and biz_lon is not None:
                    distances.append(
                        haversine_distance(user_lat, user_lon, biz_lat, biz_lon)
                    )
                else:
                    distances.append(None)
        else:
            distances = [None] * len(self.profiles)

        # Combine & rank
        results = rank_results(
            profiles=self.profiles,
            cosine_scores=cosine_scores,
            distances=distances,
            relevance_weight=self.relevance_weight,
            proximity_weight=self.proximity_weight,
            top_k=top_k,
        )

        return results
