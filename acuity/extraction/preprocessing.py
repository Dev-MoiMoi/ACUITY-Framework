"""
ACUITY Framework — Text Preprocessing

Cleans and normalises raw social media post text for downstream NLP tasks.
Handles Taglish (Tagalog-English code-switching), informal spelling,
and social-media-specific noise.
"""
import re
import unicodedata


def preprocess(text: str) -> str:
    """Full preprocessing pipeline for a single post.

    Args:
        text: Raw post text.

    Returns:
        Cleaned and normalised text.
    """
    text = normalise_unicode(text)
    text = remove_urls(text)
    text = remove_emojis(text)
    text = normalise_whitespace(text)
    return text.strip()


def normalise_unicode(text: str) -> str:
    """Normalise Unicode characters (e.g. fullwidth → ASCII)."""
    return unicodedata.normalize("NFKC", text)


def remove_urls(text: str) -> str:
    """Strip URLs from the text."""
    return re.sub(r"https?://\S+", "", text)


def remove_emojis(text: str) -> str:
    """Remove common emoji / symbol characters."""
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map
        "\U0001f1e0-\U0001f1ff"  # flags
        "\U00002700-\U000027bf"  # dingbats
        "\U0000fe00-\U0000fe0f"  # variation selectors
        "\U0000200d"             # zero width joiner
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def normalise_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into single spaces."""
    return re.sub(r"\s+", " ", text)
