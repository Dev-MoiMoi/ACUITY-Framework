"""
ACUITY Framework — Extraction Interfaces

Abstract base classes defining pluggable extension points for the
extraction pipeline. Third-party code can implement these interfaces
to provide custom NER backends without modifying ACUITY's source.

Example:
    >>> from acuity.extraction.interfaces import NERBackend
    >>> class MyNER(NERBackend):
    ...     def extract_entities(self, text):
    ...         return {"business_name": [], "categories": [], "locations": []}
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class NERBackend(ABC):
    """Abstract interface for Named Entity Recognition backends.

    Implement this class to provide a custom NER backend that can be
    injected into :class:`~acuity.extraction.pipeline.ExtractionPipeline`
    via its ``ner_backend`` constructor parameter.

    The backend receives preprocessed text and must return a dictionary
    of extracted entity lists.

    Example:
        >>> class KeywordNER(NERBackend):
        ...     def extract_entities(self, text: str) -> dict:
        ...         names = ["My Shop"] if "shop" in text.lower() else []
        ...         return {"business_name": names, "categories": [], "locations": []}
    """

    @abstractmethod
    def extract_entities(self, text: str) -> dict:
        """Extract named entities from preprocessed text.

        Args:
            text: Preprocessed post text (already cleaned and normalised).

        Returns:
            A dictionary with the following keys, each mapping to a
            list of extracted strings:

            - ``"business_name"``: Extracted business name fragments.
            - ``"categories"``: Extracted service/business categories.
            - ``"locations"``: Extracted location/address fragments.
        """
        ...
