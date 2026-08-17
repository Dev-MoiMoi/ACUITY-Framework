"""
ACUITY Framework — Verification Module

Business legitimacy verification via fuzzy matching against
government registries (e.g., BPLO — Business Permits and Licensing Office).

Quick Start:
    >>> from acuity.verification import BPLOVerifier
    >>> verifier = BPLOVerifier()
    >>> verifier.load_registry_from_list([{"name": "Juan's Bakeshop"}])
    >>> result = verifier.verify("Mang Juan's Bakery")
"""

from .bplo import BPLOVerifier

__all__ = ["BPLOVerifier"]
