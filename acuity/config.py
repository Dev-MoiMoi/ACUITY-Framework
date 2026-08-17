"""
ACUITY Framework Configuration

Provides a centralized dataclass for all configurable parameters.
No hardcoded paths — users supply their own paths and thresholds.
"""
from dataclasses import dataclass


@dataclass
class AcuityConfig:
    """Configuration for the ACUITY framework.

    All paths default to None, requiring the user to set them
    based on their own project structure.

    Attributes:
        ner_model_path: Path to a trained NER model (CRF .pkl or HuggingFace directory).
        ner_backend: Which NER backend to use: ``"crf"`` or ``"transformer"``.
        completeness_threshold: Minimum number of populated detail fields
            for a profile to be considered complete enough to keep.
        relevance_weight: Weight for textual relevance in recommendation ranking [0, 1].
        proximity_weight: Weight for geographic proximity in recommendation ranking [0, 1].
        default_top_k: Default number of results returned by the recommendation engine.
        fuzzy_match_threshold_verified: Levenshtein ratio threshold (0–1) to mark
            a business as "Verified" against a BPLO registry.
        fuzzy_match_threshold_pending: Levenshtein ratio threshold (0–1) to mark
            a business as "Pending Verification".
        max_flags_threshold: Number of user flags before a profile is auto-deactivated.
    """

    # NLP / Pipeline settings
    ner_model_path: str | None = None
    ner_backend: str = "crf"  # "crf" or "transformer"
    completeness_threshold: int = 2

    # Recommendation settings
    relevance_weight: float = 0.6
    proximity_weight: float = 0.4
    default_top_k: int = 10

    # BPLO Verification settings
    fuzzy_match_threshold_verified: float = 0.8
    fuzzy_match_threshold_pending: float = 0.6

    # Business Profile limits
    max_flags_threshold: int = 3
