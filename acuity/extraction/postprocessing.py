"""
ACUITY Framework — Post-processing & Profile Construction

Merges NER outputs and rule-based fields into a unified business profile,
performs validation, and assigns confidence scores.
"""
from __future__ import annotations

import re


def format_business_name(name: str) -> str:
    """Normalise and title-case a raw business name string.

    Args:
        name: Raw business name extracted from a post.

    Returns:
        Cleaned and title-cased name.
    """
    if not name:
        return name
    cleaned = re.sub(r'[#.,!_]', ' ', name)
    cleaned = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned.title()


def build_business_profile(
    raw_text: str,
    entities: dict,
    structured_fields: dict,
    metadata: dict | None = None,
    poster_name: str | None = None,
) -> dict | None:
    """Construct a business profile dict from extraction outputs.

    Args:
        raw_text: The original post text.
        entities: Output from NER (``business_name``, ``categories``, ``locations``).
        structured_fields: Output from rule-based extraction (``phones``, ``prices``, ``hours``).
        metadata: Optional metadata (``source_index``, ``scraped_at``, etc.).
        poster_name: Optional poster name to fall back to if business_name is missing.

    Returns:
        A business profile dict, or ``None`` if insufficient information was extracted.
    """
    b_names = entities.get("business_name", [])
    raw_business_name = " ".join(b_names) if b_names else None
    if not raw_business_name and poster_name:
        raw_business_name = poster_name

    business_name = format_business_name(raw_business_name) if raw_business_name else None

    profile = {
        "business_name": business_name,
        "categories": entities.get("categories", []),
        "locations": entities.get("locations", []),
        "phones": structured_fields.get("phones", []),
        "prices": structured_fields.get("prices", []),
        "hours": structured_fields.get("hours", []),
        "description": raw_text,
        "metadata": metadata or {},
    }

    # Basic validation: require at least some extractable information
    has_info = (
        profile["business_name"]
        or profile["categories"]
        or profile["phones"]
        or profile["prices"]
    )

    return profile if has_info else None
