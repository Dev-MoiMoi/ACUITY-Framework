"""
ACUITY Framework — Basic Recommendation Example

Demonstrates how to use the RecommendationEngine to search and rank
business profiles using TF-IDF text similarity and geographic proximity.

Usage:
    python examples/basic_recommendation.py
"""

from acuity.recommendation import RecommendationEngine
from acuity.config import AcuityConfig


def main():
    # Configure the engine weights
    config = AcuityConfig(
        relevance_weight=0.6,
        proximity_weight=0.4,
        default_top_k=5,
    )

    engine = RecommendationEngine(config=config)

    # Sample business profiles (these would normally come from the extraction pipeline)
    profiles = [
        {
            "name": "Mang Juan's Bakery",
            "description": "Fresh pandesal and ensaymada baked daily. Bread, pastries, cakes for all occasions.",
            "categories": ["bakery", "food"],
            "services": ["pandesal", "ensaymada", "cakes"],
            "latitude": 14.2725,
            "longitude": 121.1250,
        },
        {
            "name": "JC Auto Repair",
            "description": "Complete auto repair services including vulcanizing, oil change, brake repair.",
            "categories": ["auto repair", "vulcanizing"],
            "services": ["vulcanizing", "oil change", "brake repair"],
            "latitude": 14.2680,
            "longitude": 121.1190,
        },
        {
            "name": "Lina's Laundry Services",
            "description": "Professional wash and fold, dry cleaning, and ironing services.",
            "categories": ["laundry"],
            "services": ["wash and fold", "dry clean", "ironing"],
            "latitude": 14.2700,
            "longitude": 121.1210,
        },
        {
            "name": "Ate Rose's Carenderia",
            "description": "Home-cooked Filipino meals, tapsilog, longsilog, and other silog meals.",
            "categories": ["food", "restaurant"],
            "services": ["tapsilog", "longsilog", "Filipino food"],
            "latitude": 14.2715,
            "longitude": 121.1230,
        },
        {
            "name": "RJ's Sari-Sari Store",
            "description": "Convenience store with groceries, snacks, beverages, and load services.",
            "categories": ["sari-sari", "convenience"],
            "services": ["groceries", "snacks", "e-load"],
            "latitude": 14.2690,
            "longitude": 121.1200,
        },
    ]

    # Load profiles into the engine
    engine.set_profiles(profiles)

    # User's location (e.g., their current GPS coordinates)
    user_lat = 14.2710
    user_lon = 121.1220

    print("=" * 60)
    print("ACUITY Framework — Recommendation Engine Demo")
    print("=" * 60)

    # Search queries
    queries = ["bakery bread", "auto repair vulcanizing", "food restaurant", "laundry"]

    for query in queries:
        print(f"\n🔍 Query: \"{query}\"")
        print("-" * 40)

        results = engine.recommend(
            query=query,
            user_lat=user_lat,
            user_lon=user_lon,
            top_k=3,
        )

        for rank, result in enumerate(results, 1):
            print(f"  #{rank} {result['name']}")
            print(f"      Relevance: {result['relevance_score']:.4f}")
            print(f"      Distance:  {result['distance_km']} km")
            print(f"      Final:     {result['final_score']:.4f}")
        print()


if __name__ == "__main__":
    main()
