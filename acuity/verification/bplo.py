"""
ACUITY Framework — BPLO Registry Verification

Verifies extracted business profiles against a government registry
(e.g., BPLO — Business Permits and Licensing Office) using fuzzy
string matching (Levenshtein ratio).

This module is fully self-contained — it uses the framework's own
``levenshtein_ratio`` implementation with no external dependencies.
"""
from __future__ import annotations

import csv
from typing import Any

from ..config import AcuityConfig
from ..utils import (
    levenshtein_ratio,
    hybrid_fuzzy_match,
    fast_levenshtein_ratio,
    multi_stage_match,
    pre_tokenize_sort,
)
from .interfaces import VerificationStrategy


class FastMatchStrategy(VerificationStrategy):
    """Optimised verification strategy using multi-stage heuristic filtering.

    Applies progressively more expensive checks — length-ratio
    pre-screening, ``difflib.SequenceMatcher`` heuristic, then
    early-pruning Levenshtein DP — to quickly eliminate non-matching
    candidates before running the full edit-distance computation.

    This is the same pipeline used in the ACUITY production system
    for batch BPLO registry verification with hundreds of entries.

    Both names are pre-tokenized and sorted before comparison to
    achieve order-independence (e.g. ``"bakeshop juan"`` matches
    ``"juan bakeshop"``).

    Args:
        threshold: Minimum similarity ratio for a match to be
            considered.  Candidates below this are rejected early
            (default: ``0.6``).
    """

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def compute_match_score(
        self,
        candidate_name: str,
        registry_name: str,
    ) -> float:
        """Compute similarity using the multi-stage pipeline.

        Args:
            candidate_name: Lowercased extracted business name.
            registry_name: Lowercased registry entry name.

        Returns:
            Similarity ratio in ``[0, 1]``, or ``0.0`` if the
            threshold cannot be met.
        """
        sorted_candidate = pre_tokenize_sort(candidate_name)
        sorted_registry = pre_tokenize_sort(registry_name)
        return multi_stage_match(
            sorted_candidate, sorted_registry, self.threshold,
        )


class BPLOVerifier:
    """Verifies business names against a loaded registry using fuzzy matching.

    Args:
        config: An ``AcuityConfig`` instance. If ``None``, uses defaults.
        verification_strategy: An optional
            :class:`~acuity.verification.interfaces.VerificationStrategy`
            instance.  When provided, replaces the built-in
            :func:`~acuity.utils.hybrid_fuzzy_match` function for
            computing name similarity scores.  When ``None`` (the
            default), the existing hybrid fuzzy matching is used.

            Use :class:`FastMatchStrategy` for batch scenarios where
            throughput matters more than recall:

            >>> verifier = BPLOVerifier(
            ...     verification_strategy=FastMatchStrategy(threshold=0.6),
            ... )
    """

    def __init__(
        self,
        config: AcuityConfig | None = None,
        verification_strategy: VerificationStrategy | None = None,
    ):
        self.config = config or AcuityConfig()
        self.registry: list[dict[str, str]] = []
        self._verification_strategy = verification_strategy

    def load_registry_from_csv(self, path: str) -> None:
        """Load BPLO registry from a CSV file.

        The CSV is expected to have a header row. At minimum, each row
        should have a ``name`` or ``business_name`` column.

        Args:
            path: Path to the CSV file.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.registry = list(reader)
        except Exception as e:
            print(f"Warning: Could not load BPLO registry from {path}: {e}")

    def load_registry_from_list(self, entries: list[dict[str, str]]) -> None:
        """Load registry from an in-memory list of dictionaries.

        This is useful when the registry is stored in a database rather
        than a CSV file.

        Args:
            entries: List of dictionaries, each with at least a ``name``
                or ``business_name`` key.
        """
        self.registry = entries

    def verify(self, business_name: str) -> dict[str, Any]:
        """Verify a single business name against the loaded registry.

        Args:
            business_name: The business name to verify.

        Returns:
            Dictionary with keys:
                - ``status``: ``"Verified"``, ``"Pending Verification"``, or ``"Unverified"``
                - ``score``: Best similarity ratio (0–1)
                - ``match``: The best matching registry entry, or ``None``
        """
        best_match = None
        best_score = 0.0

        name_lower = business_name.lower().strip()
        if not name_lower or not self.registry:
            return {"status": "Unverified", "score": 0.0, "match": None}

        for entry in self.registry:
            bplo_name = entry.get("name", entry.get("business_name", "")).lower()
            if not bplo_name:
                continue

            if self._verification_strategy is not None:
                score = self._verification_strategy.compute_match_score(
                    name_lower, bplo_name,
                )
            else:
                score = hybrid_fuzzy_match(name_lower, bplo_name)

            if score > best_score:
                best_score = score
                best_match = entry

        threshold_verified = self.config.fuzzy_match_threshold_verified
        threshold_pending = self.config.fuzzy_match_threshold_pending

        if best_score >= threshold_verified:
            status = "Verified"
        elif best_score >= threshold_pending:
            status = "Pending Verification"
        else:
            status = "Unverified"
            best_match = None

        return {
            "status": status,
            "score": round(best_score, 2),
            "match": best_match,
        }

    def verify_batch(self, profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Verify a batch of profiles, updating them in-place with verification status.

        Each profile is updated with:
            - ``status``: Verification status string
            - ``is_verified``: Boolean flag
            - ``verification_score``: Levenshtein similarity score
            - ``matched_registry_name``: Name of the matched registry entry (if any)

        Args:
            profiles: List of business profile dictionaries.

        Returns:
            The same list of profiles, updated in-place.
        """
        for profile in profiles:
            name = profile.get("name", profile.get("business_name", ""))
            result = self.verify(name)

            profile["status"] = result["status"]
            profile["is_verified"] = (result["status"] == "Verified")
            profile["verification_score"] = result["score"]
            if result["match"]:
                profile["matched_registry_name"] = result["match"].get(
                    "name", result["match"].get("business_name")
                )
            else:
                profile["matched_registry_name"] = None

        return profiles
