"""
ACUITY Framework
================

**A**utomated **C**ommunity **U**nstructured **I**nformation to **T**argeted visibilit**Y**

A machine learning framework for extracting, verifying, and recommending
local micro-enterprise profiles from unstructured community posts.

Modules:
    - acuity.extraction: NLP pipeline (preprocessing → NER → rule-based → profile building)
    - acuity.recommendation: TF-IDF + cosine similarity + Haversine proximity ranking
    - acuity.verification: Business legitimacy verification via fuzzy matching against registries
    - acuity.scraper: Facebook community group post scraper (optional)

Quick Start:
    >>> from acuity.extraction import ExtractionPipeline
    >>> from acuity.recommendation import RecommendationEngine
    >>> from acuity.verification import BPLOVerifier
"""

__version__ = "2.0.0"
__author__ = "ACUITY Research Team"

from .config import AcuityConfig

__all__ = ["AcuityConfig"]
