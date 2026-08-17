"""
ACUITY Framework — Named Entity Recognition (CRF Backend)

Extracts business-related entities from preprocessed post text using
a trained CRF (Conditional Random Field) model with BIO tagging:
  - BUSINESS_NAME
  - SERVICE_CATEGORY
  - LOCATION

The CRF model path must be supplied via ``AcuityConfig.ner_model_path``.
"""
from __future__ import annotations

import pickle
from typing import Any

try:
    import nltk  # type: ignore

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    _NLTK_AVAILABLE = True
except ImportError:
    nltk: Any = None
    _NLTK_AVAILABLE = False


def _extract_features(tokens: list[str], pos_tags: list[tuple[str, str]], i: int) -> dict[str, Any]:
    """Build the feature dictionary for token at index *i*."""
    word = tokens[i]
    postag = pos_tags[i][1]

    features: dict[str, Any] = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word[:3]": word[:3],
        "word[:2]": word[:2],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "postag": postag,
        "postag[:2]": postag[:2],
    }

    if i > 0:
        word1 = tokens[i - 1]
        postag1 = pos_tags[i - 1][1]
        features.update({
            "-1:word.lower()": word1.lower(),
            "-1:word.istitle()": word1.istitle(),
            "-1:word.isupper()": word1.isupper(),
            "-1:postag": postag1,
            "-1:postag[:2]": postag1[:2],
        })
    else:
        features["BOS"] = True

    if i < len(tokens) - 1:
        word1 = tokens[i + 1]
        postag1 = pos_tags[i + 1][1]
        features.update({
            "+1:word.lower()": word1.lower(),
            "+1:word.istitle()": word1.istitle(),
            "+1:word.isupper()": word1.isupper(),
            "+1:postag": postag1,
            "+1:postag[:2]": postag1[:2],
        })
    else:
        features["EOS"] = True

    return features


def _sent2features(tokens: list[str]) -> list[dict[str, Any]]:
    """Convert a sentence (list of tokens) into a list of feature dicts."""
    if not _NLTK_AVAILABLE or nltk is None:
        return []
    pos_tags = nltk.pos_tag(tokens)
    return [_extract_features(tokens, pos_tags, i) for i in range(len(tokens))]


def load_crf_model(model_path: str):
    """Load a pickled CRF model from disk.

    Args:
        model_path: Absolute path to the ``.pkl`` file.

    Returns:
        The loaded CRF model object, or ``None`` on failure.
    """
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load CRF model from {model_path}: {e}")
        return None


def extract_entities_crf(text: str, model: Any) -> dict:
    """Extract named entities from *text* using a CRF model.

    Args:
        text: Preprocessed post text.
        model: A loaded CRF model (e.g., from ``load_crf_model``).

    Returns:
        dict with keys: ``business_name``, ``categories``, ``locations``.
        Each value is a list of extracted strings.
    """
    extracted: dict[str, list[str]] = {
        "business_name": [],
        "categories": [],
        "locations": [],
    }

    if not model or not _NLTK_AVAILABLE:
        return extracted

    tokens = text.split()
    if not tokens:
        return extracted

    features = _sent2features(tokens)
    predictions = model.predict([features])[0]

    # Reconstruct entities from BIO tags
    current_entity_type: str | None = None
    current_entity_tokens: list[str] = []

    def _save_entity() -> None:
        if current_entity_type and current_entity_tokens:
            entity_text = " ".join(current_entity_tokens)
            if current_entity_type == "BUSINESS_NAME":
                extracted["business_name"].append(entity_text)
            elif current_entity_type == "SERVICE_CATEGORY":
                extracted["categories"].append(entity_text)
            elif current_entity_type == "LOCATION":
                extracted["locations"].append(entity_text)

    for token, tag in zip(tokens, predictions):
        if tag.startswith("B-"):
            _save_entity()
            current_entity_type = tag[2:]
            current_entity_tokens = [token]
        elif tag.startswith("I-"):
            if current_entity_type == tag[2:]:
                current_entity_tokens.append(token)
            else:
                _save_entity()
                current_entity_type = tag[2:]
                current_entity_tokens = [token]
        else:
            _save_entity()
            current_entity_type = None
            current_entity_tokens = []

    _save_entity()

    return extracted
