"""
ACUITY Framework — Extraction Pipeline Orchestrator

Coordinates the full extraction flow:
    Raw post text → preprocessing → NER → rule-based extraction → profile construction

This is the main public API for the extraction module. It is fully decoupled
from any web framework, database, or frontend — it takes text in and returns
structured profiles out.

Usage:
    >>> from acuity.extraction import ExtractionPipeline
    >>> pipeline = ExtractionPipeline()
    >>> profiles = pipeline.extract_from_texts([
    ...     "Mang Juan's Bakery, Mamatid, open 8am-5pm, 0917-123-4567",
    ... ])
"""
from __future__ import annotations

from typing import Any

from .preprocessing import preprocess
from .rules import extract_structured_fields
from .postprocessing import build_business_profile
from .interfaces import NERBackend
from ..scraper.interfaces import DataSource
from ..config import AcuityConfig


class ExtractionPipeline:
    """End-to-end extraction pipeline for community post text.

    This class coordinates preprocessing, NER, rule-based extraction,
    and profile construction into a single callable pipeline.

    Args:
        config: An ``AcuityConfig`` instance. If ``None``, uses defaults.
        ner_model: A pre-loaded NER model object. If provided, the pipeline
            will use this model directly instead of loading from ``config.ner_model_path``.
        ner_backend: An optional :class:`~acuity.extraction.interfaces.NERBackend`
            instance. When provided, the pipeline uses this backend for NER
            instead of the built-in CRF/transformer string-branch logic.
            When ``None`` (the default), the existing config-driven backend
            selection is used.
        data_source: An optional :class:`~acuity.scraper.interfaces.DataSource`
            instance. When provided, enables :meth:`extract_from_source` to
            fetch posts from a custom data source before extraction.
            When ``None`` (the default), only :meth:`extract_from_texts`
            is available (existing behavior unchanged).
    """

    def __init__(
        self,
        config: AcuityConfig | None = None,
        ner_model: Any = None,
        ner_backend: NERBackend | None = None,
        data_source: DataSource | None = None,
    ):
        self.config = config or AcuityConfig()
        self._ner_model = ner_model
        self._ner_extract_fn = None
        self._custom_ner_backend = ner_backend
        self._data_source = data_source
        if self._custom_ner_backend is None:
            self._setup_ner()

    def _setup_ner(self) -> None:
        """Initialise the NER backend based on configuration."""
        backend = self.config.ner_backend

        if backend == "crf":
            from .ner_crf import extract_entities_crf, load_crf_model

            if self._ner_model is None and self.config.ner_model_path:
                self._ner_model = load_crf_model(self.config.ner_model_path)

            self._ner_extract_fn = lambda text: extract_entities_crf(text, self._ner_model)

        elif backend == "transformer":
            from .ner_transformer import extract_entities_transformer, load_transformer_model

            if self._ner_model is None and self.config.ner_model_path:
                self._ner_model = load_transformer_model(self.config.ner_model_path)

            self._ner_extract_fn = lambda text: extract_entities_transformer(text, self._ner_model)

        else:
            raise ValueError(f"Unknown NER backend: {backend!r}. Use 'crf' or 'transformer'.")

    def extract_single(
        self,
        text: str,
        metadata: dict | None = None,
        poster_name: str | None = None,
    ) -> dict | None:
        """Run the full extraction pipeline on a single post.

        Args:
            text: Raw post text.
            metadata: Optional metadata dict (e.g., source_index, scraped_at).
            poster_name: Optional poster name to use as fallback business name.

        Returns:
            A business profile dict, or ``None`` if not enough info was extracted.
        """
        # Step 1: Preprocess (clean, normalise)
        cleaned = preprocess(text)

        # Step 2: Named Entity Recognition
        if self._custom_ner_backend is not None:
            entities = self._custom_ner_backend.extract_entities(cleaned)
        elif self._ner_extract_fn:
            entities = self._ner_extract_fn(cleaned)
        else:
            entities = {
                "business_name": [], "categories": [], "locations": []
            }

        # Step 3: Rule-based extraction (contacts, hours, address patterns)
        structured = extract_structured_fields(cleaned)

        # Step 4: Build business profile
        profile = build_business_profile(
            raw_text=text,
            entities=entities,
            structured_fields=structured,
            metadata=metadata,
            poster_name=poster_name,
        )

        return profile

    def extract_from_texts(
        self,
        texts: list[str | dict],
        completeness_threshold: int | None = None,
        deduplicate: bool = True,
    ) -> list[dict]:
        """Run extraction on a batch of texts and return quality-filtered profiles.

        Args:
            texts: A list of raw text strings, or a list of dicts with at least
                a ``"text"`` key (and optionally ``"poster"`` and ``"scraped_at"``).
            completeness_threshold: Minimum number of populated detail fields
                (categories, locations, phones, prices, hours) to keep a profile.
                Defaults to ``config.completeness_threshold``.
            deduplicate: If True, deduplicate profiles by business name.

        Returns:
            List of extracted business profile dicts.
        """
        threshold = completeness_threshold if completeness_threshold is not None else self.config.completeness_threshold

        profiles = []
        for i, item in enumerate(texts):
            # Accept either plain strings or dicts
            if isinstance(item, str):
                raw_text = item
                poster = None
                scraped_at = None
            else:
                raw_text = item.get("text", "")
                poster = item.get("poster")
                scraped_at = item.get("scraped_at")

            profile = self.extract_single(
                text=raw_text,
                metadata={"source_index": i, "scraped_at": scraped_at},
                poster_name=poster,
            )

            if profile:
                # Filter weak profiles by counting populated detail fields
                detail_lists = [
                    profile.get("categories", []),
                    profile.get("locations", []),
                    profile.get("phones", []),
                    profile.get("prices", []),
                    profile.get("hours", []),
                ]
                populated = sum(1 for field in detail_lists if len(field) > 0)

                if populated >= threshold:
                    profiles.append(profile)

        # Deduplicate by business name
        if deduplicate:
            seen_names: set[str] = set()
            unique_profiles = []
            for p in profiles:
                name = p.get("business_name")
                if name and name not in seen_names:
                    unique_profiles.append(p)
                    seen_names.add(name)
            profiles = unique_profiles

        return profiles

    def extract_from_source(
        self,
        sources: list[str],
        max_posts: int = 500,
        completeness_threshold: int | None = None,
        deduplicate: bool = True,
    ) -> list[dict]:
        """Fetch posts from a :class:`~acuity.scraper.interfaces.DataSource` and extract profiles.

        This is a convenience method that combines data fetching and extraction
        into a single call. It requires a ``data_source`` to have been provided
        to the pipeline constructor.

        Args:
            sources: List of source identifiers (meaning depends on the
                :class:`~acuity.scraper.interfaces.DataSource` implementation).
            max_posts: Maximum number of posts to fetch.
            completeness_threshold: Minimum number of populated detail fields
                to keep a profile. Defaults to ``config.completeness_threshold``.
            deduplicate: If True, deduplicate profiles by business name.

        Returns:
            List of extracted business profile dicts.

        Raises:
            RuntimeError: If no ``data_source`` was provided to the constructor.
        """
        if self._data_source is None:
            raise RuntimeError(
                "No data_source was provided to ExtractionPipeline. "
                "Pass a DataSource instance via the data_source parameter, "
                "or use extract_from_texts() directly."
            )

        posts = self._data_source.fetch_posts(sources, max_posts=max_posts)
        return self.extract_from_texts(
            texts=posts,
            completeness_threshold=completeness_threshold,
            deduplicate=deduplicate,
        )
