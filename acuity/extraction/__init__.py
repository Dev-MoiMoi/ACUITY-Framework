"""
ACUITY Framework — Extraction Module

NLP pipeline for extracting structured business information from
unstructured community group posts (e.g., Facebook, forums).

Components:
    - preprocessing: Text cleaning and normalisation
    - ner: Named Entity Recognition (CRF or Transformer backend)
    - rules: Regex-based structured field extraction
    - postprocessing: Profile construction from extraction outputs
    - pipeline: End-to-end orchestrator

Quick Start:
    >>> from acuity.extraction import ExtractionPipeline
    >>> pipeline = ExtractionPipeline()
    >>> profiles = pipeline.extract_from_texts(["Mang Juan's Bakery, Mamatid, open 8am-5pm, 0917-123-4567"])
"""

from .pipeline import ExtractionPipeline

__all__ = ["ExtractionPipeline"]
