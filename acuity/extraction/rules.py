"""
ACUITY Framework — Rule-Based Extraction

Complements the NER module by extracting structured fields that follow
predictable patterns in community posts:
  - Phone numbers (PH mobile formats)
  - Addresses / barangay references
  - Operating hours
  - Price mentions
"""
import re


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# Philippine mobile: 09XX-XXX-XXXX or +639XX-XXX-XXXX
PHONE_PATTERN = re.compile(
    r"(?:\+63|0)9\d{2}[\s\-]?\d{3}[\s\-]?\d{4}"
)

# Simple price patterns (₱, PHP, P followed by digits)
PRICE_PATTERN = re.compile(
    r"(?:[₱Pp](?:HP)?)\s?\d[\d,]*(?:\.\d{2})?"
)

# Operating hours heuristic (e.g. "open 8am-5pm", "available 24/7")
HOURS_PATTERN = re.compile(
    r"\b(?:open|available|hours?)\b.*?\d{1,2}\s?(?:am|pm|AM|PM)",
    re.IGNORECASE,
)


def extract_structured_fields(text: str) -> dict:
    """Extract structured information from *text* using regex patterns.

    Args:
        text: Preprocessed post text.

    Returns:
        dict with keys: ``phones``, ``prices``, ``hours``.
        Each value is a list of matched strings.
    """
    return {
        "phones": PHONE_PATTERN.findall(text),
        "prices": PRICE_PATTERN.findall(text),
        "hours": HOURS_PATTERN.findall(text),
    }
