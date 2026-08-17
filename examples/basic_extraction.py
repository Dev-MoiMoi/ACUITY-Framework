"""
ACUITY Framework — Basic Extraction Example

Demonstrates how to use the ExtractionPipeline to extract
business profiles from raw community post text.

Usage:
    python examples/basic_extraction.py
"""

from acuity.extraction import ExtractionPipeline
from acuity.config import AcuityConfig


def main():
    # Create a pipeline without a trained NER model.
    # The pipeline will still work using rule-based extraction alone.
    config = AcuityConfig(
        ner_backend="crf",
        ner_model_path=None,  # Set this to your CRF model path if available
        completeness_threshold=1,  # Lower threshold for this demo
    )

    pipeline = ExtractionPipeline(config=config)

    # Sample community posts (simulating Facebook group posts)
    sample_posts = [
        "Mang Juan's Bakery po ito, nasa Mamatid, Cabuyao. Open 8am-5pm daily. "
        "Fresh pandesal and ensaymada! Contact 0917-123-4567 for bulk orders. "
        "Presyo: ₱5 per piece, ₱50 per dozen.",

        "JC Auto Repair Shop, vulcanizing and oil change. "
        "Located at Brgy Banay-Banay, Cabuyao City. "
        "Open 7am-6pm Monday to Saturday. Call 0918-987-6543.",

        "Selling preloved clothes, DM for prices.",  # Should be filtered out

        "Lina's Laundry Services, Brgy Marinig. "
        "Wash and fold ₱40/kilo, dry clean available. "
        "Open Monday-Saturday 0919-555-1234.",
    ]

    print("=" * 60)
    print("ACUITY Framework — Extraction Pipeline Demo")
    print("=" * 60)

    profiles = pipeline.extract_from_texts(sample_posts)

    print(f"\nExtracted {len(profiles)} business profiles:\n")

    for i, profile in enumerate(profiles, 1):
        print(f"--- Profile {i} ---")
        print(f"  Business Name: {profile.get('business_name', 'N/A')}")
        print(f"  Categories:    {profile.get('categories', [])}")
        print(f"  Locations:     {profile.get('locations', [])}")
        print(f"  Phones:        {profile.get('phones', [])}")
        print(f"  Prices:        {profile.get('prices', [])}")
        print(f"  Hours:         {profile.get('hours', [])}")
        print()


if __name__ == "__main__":
    main()
