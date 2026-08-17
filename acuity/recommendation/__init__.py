"""
ACUITY Framework — Recommender Module

Recommendation engine using TF-IDF, cosine similarity, and Haversine
proximity ranking to match user queries with local business profiles.

Quick Start:
    >>> from acuity.recommendation import RecommendationEngine
    >>> engine = RecommendationEngine()
    >>> engine.set_profiles([{"name": "Juan's Bakery", "description": "Fresh bread daily"}])
    >>> results = engine.recommend("bakery")
"""

from .engine import RecommendationEngine

__all__ = ["RecommendationEngine"]
