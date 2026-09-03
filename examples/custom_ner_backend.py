"""
ACUITY Framework — Custom NER Backend Example

Demonstrates how to implement a custom NER backend by subclassing
:class:`~acuity.extraction.interfaces.NERBackend` and injecting it
into the extraction pipeline.

This example uses simple keyword matching — no ML models required.

Usage:
    python examples/custom_ner_backend.py
"""
from __future__ import annotations

import sys
import io

# Ensure UTF-8 output on Windows console (handles ₱ and other Unicode)
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from acuity.extraction.interfaces import NERBackend
from acuity.extraction.pipeline import ExtractionPipeline
from acuity.config import AcuityConfig


# --- Known keyword dictionaries (simulating a domain-specific lookup) ---

BUSINESS_TYPE_KEYWORDS = {
    "bakery": "bakery",
    "bakeshop": "bakery",
    "pandesal": "bakery",
    "repair": "automotive",
    "vulcanizing": "automotive",
    "auto": "automotive",
    "laundry": "laundry",
    "salon": "beauty",
    "barber": "beauty",
    "spa": "beauty",
    "pharmacy": "health",
    "clinic": "health",
}

LOCATION_KEYWORDS = [
    "mamatid", "banay-banay", "marinig", "cabuyao", "calamba",
    "biñan", "santa rosa", "san pedro", "laguna",
]


class KeywordNERBackend(NERBackend):
    """A simple keyword-lookup NER backend for demonstration.

    Scans preprocessed text for known business-type and location keywords.
    No ML model is required — this is purely dictionary-based.

    This class shows how third-party code can implement a custom NER
    backend and inject it into ACUITY's pipeline without modifying
    the framework's source code.
    """

    def extract_entities(self, text: str) -> dict:
        """Extract entities using keyword matching.

        Args:
            text: Preprocessed post text.

        Returns:
            Dict with ``business_name``, ``categories``, and ``locations``.
        """
        text_lower = text.lower()
        words = text_lower.split()

        # Detect categories from keywords
        categories = []
        for keyword, category in BUSINESS_TYPE_KEYWORDS.items():
            if keyword in text_lower and category not in categories:
                categories.append(category)

        # Detect locations from keywords
        locations = []
        for loc in LOCATION_KEYWORDS:
            if loc in text_lower and loc not in locations:
                locations.append(loc.title())

        # Simple business name heuristic: look for title-cased phrases
        # at the start of the text (common in community posts)
        business_names = []
        original_words = text.split()
        if original_words:
            name_parts = []
            for word in original_words:
                # Stop collecting at common separators
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


def main():
    print("=" * 60)
    print("ACUITY — Custom NER Backend Demo")
    print("=" * 60)

    # Create the custom backend
    custom_ner = KeywordNERBackend()

    # Inject it into the pipeline — no config.ner_backend string needed!
    config = AcuityConfig(completeness_threshold=1)
    pipeline = ExtractionPipeline(config=config, ner_backend=custom_ner)

    sample_posts = [
        "Mang Juan's Bakery po ito, nasa Mamatid, Cabuyao. Open 8am-5pm daily. "
        "Fresh pandesal and ensaymada! Contact 0917-123-4567. Presyo: ₱5 per piece.",

        "JC Auto Repair Shop, vulcanizing and oil change. "
        "Located at Brgy Banay-Banay, Cabuyao City. "
        "Open 7am-6pm Monday to Saturday. Call 0918-987-6543.",
    ]

    profiles = pipeline.extract_from_texts(sample_posts)

    print(f"\nExtracted {len(profiles)} profiles using KeywordNERBackend:\n")
    for i, profile in enumerate(profiles, 1):
        print(f"--- Profile {i} ---")
        print(f"  Business Name: {profile.get('business_name', 'N/A')}")
        print(f"  Categories:    {profile.get('categories', [])}")
        print(f"  Locations:     {profile.get('locations', [])}")
        print(f"  Phones:        {profile.get('phones', [])}")
        print(f"  Prices:        {profile.get('prices', [])}")
        print()


if __name__ == "__main__":
    main()
