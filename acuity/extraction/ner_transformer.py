"""
ACUITY Framework — Named Entity Recognition (Transformer Backend)

Extracts business-related entities using a fine-tuned HuggingFace
transformer model with BIO tagging:
  - BUSINESS_NAME
  - SERVICE_CATEGORY
  - LOCATION

Requires ``transformers`` and ``torch`` (install with ``pip install acuity-framework[transformers]``).
"""
from __future__ import annotations

from typing import Any


def load_transformer_model(model_path: str):
    """Load a HuggingFace NER pipeline from a local directory.

    Args:
        model_path: Path to the fine-tuned model directory.

    Returns:
        A HuggingFace ``pipeline`` object, or ``None`` on failure.
    """
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore
        print(f"Loading Fine-Tuned NER model from {model_path}...")
        return hf_pipeline("ner", model=model_path, aggregation_strategy="simple")
    except Exception as e:
        print(f"Warning: Could not load NER transformer model: {e}")
        return None


def extract_entities_transformer(text: str, model: Any) -> dict:
    """Extract named entities from *text* using a HuggingFace NER model.

    Args:
        text: Preprocessed post text.
        model: A loaded HuggingFace NER pipeline (from ``load_transformer_model``).

    Returns:
        dict with keys: ``business_name``, ``categories``, ``locations``.
        Each value is a list of extracted strings.
    """
    extracted: dict[str, list[str]] = {
        "business_name": [],
        "categories": [],
        "locations": [],
    }

    if not model:
        return extracted

    # Truncate to ~2000 chars to stay within 512 token limit
    truncated_text = text[:2000]

    hf_results = model(truncated_text)

    for entity in hf_results:
        label = entity.get("entity_group")
        word = entity.get("word", "").strip()
        score = float(entity.get("score", 0))

        # Only keep confident extractions
        if score < 0.30:
            continue

        if label == "BUSINESS_NAME":
            extracted["business_name"].append(word)
        elif label == "LOCATION":
            extracted["locations"].append(word)
        elif label == "SERVICE_CATEGORY":
            extracted["categories"].append(word)

    return extracted
