"""
ACUITY Framework — Verification Module

Business legitimacy verification via fuzzy matching against
government registries (e.g., BPLO — Business Permits and Licensing Office).

Quick Start:
    >>> from acuity.verification import BPLOVerifier
    >>> verifier = BPLOVerifier()
    >>> verifier.load_registry_from_list([{"name": "Juan's Bakeshop"}])
    >>> result = verifier.verify("Mang Juan's Bakery")

For high-throughput batch verification use the optimised strategy:
    >>> from acuity.verification import BPLOVerifier, FastMatchStrategy
    >>> verifier = BPLOVerifier(verification_strategy=FastMatchStrategy(threshold=0.6))
"""

from .bplo import BPLOVerifier, FastMatchStrategy
from .interfaces import VerificationStrategy

__all__ = ["BPLOVerifier", "FastMatchStrategy", "VerificationStrategy"]
