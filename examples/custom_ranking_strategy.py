"""
ACUITY Framework — Custom Ranking Strategy Example

Demonstrates how to implement a custom ranking strategy by subclassing
:class:`~acuity.recommendation.interfaces.RankingStrategy` and injecting
it into the recommendation engine.

This example uses simple keyword overlap scoring instead of TF-IDF + cosine.

Usage:
    python examples/custom_ranking_strategy.py
"""
from __future__ import annotations

import sys
import io

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from acuity.recommendation.interfaces import RankingStrategy
from acuity.recommendation import RecommendationEngine
from acuity.config import AcuityConfig


class KeywordMatchRanking(RankingStrategy):
    """A simple keyword-overlap ranking strategy for demonstration.

    Scores each profile based on how many query keywords appear in
    the profile's text fields. No TF-IDF, no vectorization — just
    plain keyword counting.

    This class shows how third-party code can replace ACUITY's
    built-in TF-IDF + cosine similarity with any ranking algorithm.
    """

    def compute_scores(
        self,
        profiles: list[dict],
        query: str,
    ) -> list[float]:
        """Score profiles by keyword overlap with the query.

        Args:
            profiles: List of business profile dicts.
            query: User's search query.

        Returns:
            List of float scores in [0, 1], one per profile.
        """
        query_words = set(query.lower().split())
        if not query_words:
            return [0.0] * len(profiles)

        scores = []
        for profile in profiles:
            # Build searchable text from profile fields
            name = profile.get("name") or profile.get("business_name") or ""
            desc = profile.get("description") or ""
            cats = " ".join(profile.get("categories") or [])
            text = f"{name} {desc} {cats}".lower()
            text_words = set(text.split())

            # Score = fraction of query words found in profile text
            matches = len(query_words & text_words)
            score = matches / len(query_words)
            scores.append(score)

        return scores


def main():
    print("=" * 60)
    print("ACUITY — Custom Ranking Strategy Demo")
    print("=" * 60)

    # Create the custom ranking strategy
    custom_ranking = KeywordMatchRanking()

    # Inject it into the recommendation engine
    config = AcuityConfig(relevance_weight=0.8, proximity_weight=0.2)
    engine = RecommendationEngine(config=config, ranking_strategy=custom_ranking)

    # Load some sample profiles
    profiles = [
        {
            "name": "Juan's Bakery",
            "description": "Fresh bread and pastries daily, pandesal ensaymada",
            "latitude": 14.27,
            "longitude": 121.12,
        },
        {
            "name": "JC Auto Repair",
            "description": "Vulcanizing oil change car maintenance",
            "latitude": 14.28,
            "longitude": 121.13,
        },
        {
            "name": "Lina's Laundry",
            "description": "Wash fold dry clean laundry services",
            "latitude": 14.26,
            "longitude": 121.11,
        },
    ]
    engine.set_profiles(profiles)

    # Search!
    query = "bread bakery"
    results = engine.recommend(query, user_lat=14.27, user_lon=121.12, top_k=3)

    print(f"\nQuery: '{query}'")
    print(f"Using: KeywordMatchRanking (custom strategy)\n")

    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['name']}")
        print(f"     Relevance: {result['relevance_score']:.4f}")
        print(f"     Distance:  {result['distance_km']} km")
        print(f"     Final:     {result['final_score']:.4f}")
        print()


if __name__ == "__main__":
    main()
