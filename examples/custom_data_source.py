"""
ACUITY Framework — Custom Data Source Example

Demonstrates how to implement a custom data source by subclassing
:class:`~acuity.scraper.interfaces.DataSource` and injecting it
into the extraction pipeline.

This example reads posts from a CSV file instead of scraping Facebook.

Usage:
    python examples/custom_data_source.py
"""
from __future__ import annotations

import csv
import os
import sys
import io
import tempfile

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from acuity.scraper.interfaces import DataSource
from acuity.extraction.pipeline import ExtractionPipeline
from acuity.config import AcuityConfig


class CSVDataSource(DataSource):
    """A data source that reads posts from CSV files.

    Expects CSV files with at least a ``text`` column. Optionally
    also reads ``poster`` and ``scraped_at`` columns.

    This class shows how third-party code can replace the Facebook
    scraper with any data source — CSV, database, API, etc.
    """

    def fetch_posts(
        self,
        sources: list[str],
        max_posts: int = 500,
    ) -> list[dict]:
        """Read posts from one or more CSV files.

        Args:
            sources: List of CSV file paths.
            max_posts: Maximum number of posts to return.

        Returns:
            List of post dicts with ``text``, ``poster``, and ``scraped_at``.
        """
        posts = []
        for csv_path in sources:
            if not os.path.isfile(csv_path):
                print(f"Warning: CSV file not found: {csv_path}")
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

            if len(posts) >= max_posts:
                break

        print(f"CSVDataSource: loaded {len(posts)} posts from {len(sources)} file(s)")
        return posts


def main():
    print("=" * 60)
    print("ACUITY — Custom Data Source Demo")
    print("=" * 60)

    # Create a temporary CSV file with sample data
    sample_csv = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False,
        encoding="utf-8", newline="",
    )
    try:
        writer = csv.DictWriter(sample_csv, fieldnames=["text", "poster", "scraped_at"])
        writer.writeheader()
        writer.writerow({
            "text": "Mang Juan's Bakery po, nasa Mamatid. Fresh pandesal! 0917-123-4567. P5 per piece.",
            "poster": "Juan Santos",
            "scraped_at": "2024-06-15",
        })
        writer.writerow({
            "text": "JC Auto Repair, vulcanizing at Brgy Banay-Banay. Open 7am-6pm. 0918-987-6543.",
            "poster": "JC Reyes",
            "scraped_at": "2024-06-16",
        })
        writer.writerow({
            "text": "Selling old clothes, DM me",
            "poster": "Random User",
            "scraped_at": "2024-06-17",
        })
        sample_csv.close()

        # Create the custom data source
        csv_source = CSVDataSource()

        # Inject it into the pipeline
        config = AcuityConfig(completeness_threshold=1)
        pipeline = ExtractionPipeline(config=config, data_source=csv_source)

        # Use extract_from_source to fetch + extract in one call
        profiles = pipeline.extract_from_source(
            sources=[sample_csv.name],
            max_posts=100,
        )

        print(f"\nExtracted {len(profiles)} profiles from CSV data source:\n")
        for i, profile in enumerate(profiles, 1):
            print(f"--- Profile {i} ---")
            print(f"  Business Name: {profile.get('business_name', 'N/A')}")
            print(f"  Categories:    {profile.get('categories', [])}")
            print(f"  Phones:        {profile.get('phones', [])}")
            print(f"  Prices:        {profile.get('prices', [])}")
            print()
    finally:
        os.unlink(sample_csv.name)


if __name__ == "__main__":
    main()
