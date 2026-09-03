"""
Tests for the ACUITY Verification module.
"""
import pytest

from acuity.verification import BPLOVerifier
from acuity.utils import levenshtein_ratio, levenshtein_details, token_sort_ratio, token_set_ratio, hybrid_fuzzy_match
from acuity.config import AcuityConfig


# ── Levenshtein Tests ──────────────────────────────────────────────────────

class TestLevenshtein:
    def test_identical_strings(self):
        assert levenshtein_ratio("hello", "hello") == 1.0

    def test_completely_different(self):
        ratio = levenshtein_ratio("abc", "xyz")
        assert ratio < 0.5

    def test_empty_strings(self):
        assert levenshtein_ratio("", "hello") == 0.0
        assert levenshtein_ratio("hello", "") == 0.0
        assert levenshtein_ratio("", "") == 0.0

    def test_similar_strings(self):
        ratio = levenshtein_ratio("bakery", "bakeshop")
        assert 0.3 < ratio < 0.8

    def test_case_sensitivity(self):
        ratio1 = levenshtein_ratio("Bakery", "bakery")
        ratio2 = levenshtein_ratio("bakery", "bakery")
        assert ratio1 < ratio2  # Different case → lower ratio

    def test_details_returns_dict(self):
        details = levenshtein_details("hello", "hallo")
        assert "score" in details
        assert "edits" in details
        assert "max_len" in details
        assert details["edits"] == 1
        assert details["max_len"] == 5


# ── BPLOVerifier Tests ─────────────────────────────────────────────────────

class TestBPLOVerifier:
    def setup_method(self):
        """Set up a verifier with a sample registry for each test."""
        self.config = AcuityConfig(
            fuzzy_match_threshold_verified=0.8,
            fuzzy_match_threshold_pending=0.6,
        )
        self.verifier = BPLOVerifier(config=self.config)
        self.verifier.load_registry_from_list([
            {"name": "Juan's Bakeshop", "address": "Mamatid"},
            {"name": "JC Automotive Repair", "address": "Banay-Banay"},
            {"name": "Lina's Laundry Services", "address": "Marinig"},
        ])

    def test_exact_match(self):
        result = self.verifier.verify("Juan's Bakeshop")
        assert result["status"] == "Verified"
        assert result["score"] == 1.0

    def test_fuzzy_match_verified(self):
        result = self.verifier.verify("Juan's Bakery Shop")
        # Should be close enough to "Juan's Bakeshop"
        assert result["score"] > 0.6

    def test_unverified(self):
        result = self.verifier.verify("Completely Unknown Business XYZ123")
        assert result["status"] == "Unverified"
        assert result["match"] is None

    def test_empty_name(self):
        result = self.verifier.verify("")
        assert result["status"] == "Unverified"
        assert result["score"] == 0.0

    def test_empty_registry(self):
        verifier = BPLOVerifier()
        result = verifier.verify("Any Business")
        assert result["status"] == "Unverified"

    def test_verify_batch(self):
        profiles = [
            {"name": "Juan's Bakeshop", "description": "bread"},
            {"name": "Unknown Store", "description": "stuff"},
        ]
        results = self.verifier.verify_batch(profiles)
        assert results[0]["is_verified"] is True
        assert results[1]["is_verified"] is False
        assert "status" in results[0]
        assert "verification_score" in results[0]

    def test_case_insensitive_matching(self):
        result = self.verifier.verify("JUAN'S BAKESHOP")
        assert result["status"] == "Verified"

    def test_registry_from_list(self):
        verifier = BPLOVerifier()
        verifier.load_registry_from_list([
            {"business_name": "Test Biz"},  # Uses 'business_name' key
        ])
        result = verifier.verify("Test Biz")
        assert result["status"] == "Verified"

# ── Hybrid Match Tests ─────────────────────────────────────────────────────

class TestHybridFuzzyMatch:
    def test_token_sort_ratio(self):
        # Order shouldn't matter
        assert token_sort_ratio("bakeshop juan", "juan bakeshop") == 1.0
        assert token_sort_ratio("juan bakeshop", "bakeshop juan") == 1.0

    def test_token_set_ratio(self):
        # Extra words shouldn't ruin the score completely
        assert token_set_ratio("juan bakeshop in mamatid", "juan bakeshop") == 1.0
        
    def test_hybrid_match_takes_max(self):
        plain = levenshtein_ratio("bakeshop juan", "juan bakeshop") # Will be low
        sort = token_sort_ratio("bakeshop juan", "juan bakeshop") # Will be 1.0
        
        hybrid = hybrid_fuzzy_match("bakeshop juan", "juan bakeshop")
        assert hybrid == 1.0
        assert hybrid > plain

    def test_hybrid_match_penalty(self):
        # "jb" is an acronym for "juan bakeshop". The length ratio is 2 / 13 = 0.15 (which is < 0.35).
        # Token set ratio might normally score it too high if it thinks they share tokens, 
        # but with penalty, it should be lowered to avoid false positives.
        score = hybrid_fuzzy_match("jb", "juan bakeshop in the city")
        assert score < 0.5
