"""
ACUITY Framework — Extensibility Demo (All Three Extension Points)

End-to-end demonstration of all three pluggable extension points:
  1. Custom NER Backend      (KeywordNERBackend)
  2. Custom Data Source       (CSVDataSource)
  3. Custom Ranking Strategy  (KeywordMatchRanking)

This script imports the package as an external consumer would,
instantiates custom implementations of each interface, and runs
them through the full pipeline — extraction → recommendation →
verification — printing output at each stage.

Usage:
    python examples/demo_extensibility.py
"""
from __future__ import annotations

import csv
import os
import sys
import io
import tempfile

# Ensure UTF-8 output on Windows console (handles ₱ and other Unicode)
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─── Import ACUITY as an external consumer would ─────────────────────────

from acuity.config import AcuityConfig
from acuity.extraction.pipeline import ExtractionPipeline
from acuity.recommendation import RecommendationEngine
from acuity.verification import BPLOVerifier

# Import the abstract interfaces
from acuity.extraction.interfaces import NERBackend
from acuity.scraper.interfaces import DataSource
from acuity.recommendation.interfaces import RankingStrategy


# ═══════════════════════════════════════════════════════════════════════════
# Extension Point 1: Custom NER Backend
# ═══════════════════════════════════════════════════════════════════════════

BUSINESS_TYPE_KEYWORDS = {
    "bakery": "bakery", "bakeshop": "bakery", "pandesal": "bakery",
    "repair": "automotive", "vulcanizing": "automotive", "auto": "automotive",
    "laundry": "laundry", "salon": "beauty", "barber": "beauty",
}

LOCATION_KEYWORDS = [
    "mamatid", "banay-banay", "marinig", "cabuyao", "calamba",
    "biñan", "laguna",
]


class KeywordNERBackend(NERBackend):
    """Keyword-lookup NER: scans text for known business types and locations."""

    def extract_entities(self, text: str) -> dict:
        text_lower = text.lower()

        categories = []
        for keyword, category in BUSINESS_TYPE_KEYWORDS.items():
            if keyword in text_lower and category not in categories:
                categories.append(category)

        locations = []
        for loc in LOCATION_KEYWORDS:
            if loc in text_lower and loc not in locations:
                locations.append(loc.title())

        # Simple name heuristic: title-cased words at the start
        business_names = []
        original_words = text.split()
        if original_words:
            name_parts = []
            for word in original_words:
                if word.lower() in ("po", "ito", "nasa", "open", "located", ",", "-"):
                    break
                if word[0].isupper() or word.startswith("'"):
                    name_parts.append(word)
                elif name_parts:
                    break
            if name_parts:
                business_names.append(" ".join(name_parts))

        return {
            "business_name": business_names,
            "categories": categories,
            "locations": locations,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Extension Point 2: Custom Data Source
# ═══════════════════════════════════════════════════════════════════════════

class CSVDataSource(DataSource):
    """Reads posts from CSV files instead of scraping Facebook."""

    def fetch_posts(self, sources: list[str], max_posts: int = 500) -> list[dict]:
        posts = []
        for csv_path in sources:
            if not os.path.isfile(csv_path):
                continue
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if len(posts) >= max_posts:
                        break
                    text = row.get("text", "").strip()
                    if text:
                        posts.append({
                            "text": text,
                            "poster": row.get("poster", "Unknown"),
                            "scraped_at": row.get("scraped_at", ""),
                        })
        print(f"  [CSVDataSource] Loaded {len(posts)} posts")
        return posts


# ═══════════════════════════════════════════════════════════════════════════
# Extension Point 3: Custom Ranking Strategy
# ═══════════════════════════════════════════════════════════════════════════

class KeywordMatchRanking(RankingStrategy):
    """Keyword-overlap ranking: scores by fraction of query words found."""

    def compute_scores(self, profiles: list[dict], query: str) -> list[float]:
        query_words = set(query.lower().split())
        if not query_words:
            return [0.0] * len(profiles)

        scores = []
        for profile in profiles:
            name = profile.get("name") or profile.get("business_name") or ""
            desc = profile.get("description") or ""
            cats = " ".join(profile.get("categories") or [])
            text_words = set(f"{name} {desc} {cats}".lower().split())
            matches = len(query_words & text_words)
            scores.append(matches / len(query_words))
        return scores


# ═══════════════════════════════════════════════════════════════════════════
# Main Demo
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 66)
    print("  ACUITY Framework — Extensibility Demo")
    print("  All three extension points in action")
    print("=" * 66)

    # ── Step 1: Create sample CSV data ────────────────────────────────
    print("\n▶ STEP 1: Prepare sample data (CSV file)\n")

    csv_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False,
        encoding="utf-8", newline="",
    )
    try:
        writer = csv.DictWriter(csv_file, fieldnames=["text", "poster", "scraped_at"])
        writer.writeheader()
        sample_posts = [
            {
                "text": "Mang Juan's Bakery po ito, nasa Mamatid, Cabuyao. "
                        "Open 8am-5pm daily. Fresh pandesal at ₱5 per piece. "
                        "Contact 0917-123-4567 for bulk orders.",
                "poster": "Juan Santos",
                "scraped_at": "2024-06-15",
            },
            {
                "text": "JC Auto Repair Shop, vulcanizing and oil change. "
                        "Located at Brgy Banay-Banay, Cabuyao City. "
                        "Open 7am-6pm Mon-Sat. Call 0918-987-6543.",
                "poster": "JC Reyes",
                "scraped_at": "2024-06-16",
            },
            {
                "text": "Lina's Laundry Services po, Brgy Marinig, Cabuyao. "
                        "Wash and fold ₱40/kilo. Open Mon-Sat. 0919-555-1234.",
                "poster": "Lina Cruz",
                "scraped_at": "2024-06-17",
            },
            {
                "text": "Selling preloved clothes, DM for prices.",
                "poster": "Random User",
                "scraped_at": "2024-06-18",
            },
        ]
        for post in sample_posts:
            writer.writerow(post)
        csv_file.close()
        print(f"  Created temp CSV with {len(sample_posts)} posts")

        # ── Step 2: Extraction with custom NER + custom data source ──
        print("\n▶ STEP 2: Extract profiles (Custom NER + CSV Data Source)\n")

        config = AcuityConfig(completeness_threshold=1)
        pipeline = ExtractionPipeline(
            config=config,
            ner_backend=KeywordNERBackend(),
            data_source=CSVDataSource(),
        )

        profiles = pipeline.extract_from_source(
            sources=[csv_file.name],
            max_posts=100,
        )

        print(f"\n  Extracted {len(profiles)} business profiles:\n")
        for i, p in enumerate(profiles, 1):
            print(f"  {i}. {p.get('business_name', 'N/A')}")
            print(f"     Categories: {p.get('categories', [])}")
            print(f"     Locations:  {p.get('locations', [])}")
            print(f"     Phones:     {p.get('phones', [])}")
            print(f"     Prices:     {p.get('prices', [])}")

        # ── Step 3: Recommendation with custom ranking strategy ──────
        print("\n▶ STEP 3: Recommend (Custom Ranking Strategy)\n")

        engine = RecommendationEngine(
            config=AcuityConfig(relevance_weight=0.7, proximity_weight=0.3),
            ranking_strategy=KeywordMatchRanking(),
        )

        # Convert extracted profiles into recommendation format
        rec_profiles = []
        for p in profiles:
            rec_profiles.append({
                "name": p.get("business_name", "Unknown"),
                "description": p.get("description", ""),
                "categories": p.get("categories", []),
                "latitude": 14.27 + len(rec_profiles) * 0.01,  # Simulated coords
                "longitude": 121.12 + len(rec_profiles) * 0.01,
            })
        engine.set_profiles(rec_profiles)

        query = "bakery bread"
        results = engine.recommend(query, user_lat=14.27, user_lon=121.12, top_k=5)

        print(f"  Query: '{query}'\n")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['name']}")
            print(f"     Relevance: {r['relevance_score']:.4f}  "
                  f"Distance: {r['distance_km']} km  "
                  f"Final: {r['final_score']:.4f}")

        # ── Step 4: Verification (unchanged — Levenshtein-only) ──────
        print("\n▶ STEP 4: Verify against BPLO registry (unchanged, not abstracted)\n")

        verifier = BPLOVerifier()
        verifier.load_registry_from_list([
            {"name": "Juan's Bakeshop"},
            {"name": "JC Automotive Repair"},
            {"name": "Lina's Laundry Services"},
        ])

        for p in profiles:
            name = p.get("business_name", "")
            result = verifier.verify(name)
            status_icon = {"Verified": "✅", "Pending Verification": "⏳", "Unverified": "❌"}
            icon = status_icon.get(result["status"], "?")
            print(f"  {icon} {name}: {result['status']} (score: {result['score']:.2f})")

        # ── Summary ──────────────────────────────────────────────────
        print("\n" + "=" * 66)
        print("  Demo complete! All three extension points used successfully.")
        print()
        print("  Extension points demonstrated:")
        print("    1. NERBackend         → KeywordNERBackend (keyword lookup)")
        print("    2. DataSource         → CSVDataSource (CSV file reader)")
        print("    3. RankingStrategy    → KeywordMatchRanking (word overlap)")
        print()
        print("  Not abstracted (by design):")
        print("    - Haversine distance  → Fixed geographic formula")
        print("    - Pipeline stage order → Core architecture")
        print("    - Fuzzy matching      → Levenshtein-only (out of scope)")
        print("=" * 66)

    finally:
        os.unlink(csv_file.name)


if __name__ == "__main__":
    main()
