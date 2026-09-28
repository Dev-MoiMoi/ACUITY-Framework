"""
ACUITY Framework — Verification Interfaces

Abstract base classes defining pluggable extension points for the
verification module.  Third-party code can implement these interfaces
to provide custom business-name matching strategies without modifying
ACUITY's source.

Example:
    >>> from acuity.verification.interfaces import VerificationStrategy
    >>> class PhoneticMatcher(VerificationStrategy):
    ...     def compute_match_score(self, candidate, registry_name):
    ...         # Custom phonetic matching logic
    ...         return 0.85
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class VerificationStrategy(ABC):
    """Abstract interface for business-name matching strategies.

    Implement this class to provide a custom matching algorithm that can
    be injected into :class:`~acuity.verification.bplo.BPLOVerifier`
    via its ``verification_strategy`` constructor parameter.

    The strategy computes a similarity score between a candidate
    business name (extracted from a post) and a registry entry name
    (from the official BPLO database).  These scores are then compared
    against the configurable verification thresholds to assign a
    verification status.

    The default behavior (when no strategy is injected) uses the
    built-in :func:`~acuity.utils.hybrid_fuzzy_match` function, which
    combines Levenshtein ratio, token-sort ratio, and token-set ratio
    with a length-disparity penalty.

    Example:
        >>> class BM25Matcher(VerificationStrategy):
        ...     def compute_match_score(self, candidate, registry_name):
        ...         # BM25 or embedding-based matching
        ...         return score
    """

    @abstractmethod
    def compute_match_score(
        self,
        candidate_name: str,
        registry_name: str,
    ) -> float:
        """Compute a similarity score between two business names.

        Args:
            candidate_name: The extracted business name to verify
                (already lowercased and stripped).
            registry_name: A name from the official registry
                (already lowercased).

        Returns:
            A similarity score in ``[0, 1]`` where ``1.0`` means an
            exact match and ``0.0`` means no similarity at all.
        """
        ...
