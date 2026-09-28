"""
Tests for the ACUITY Verification module.
"""
import pytest

from acuity.verification import BPLOVerifier, FastMatchStrategy, VerificationStrategy
from acuity.utils import (
    levenshtein_ratio,
    levenshtein_details,
    token_sort_ratio,
    token_set_ratio,
    hybrid_fuzzy_match,
    fast_levenshtein_ratio,
    pre_tokenize_sort,
    multi_stage_match,
)
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


# ── Fast Levenshtein Tests ─────────────────────────────────────────────────

class TestFastLevenshtein:
    def test_identical_strings(self):
        score = fast_levenshtein_ratio("hello", "hello", 0.5)
        assert score == 1.0

    def test_similar_strings_above_threshold(self):
        score = fast_levenshtein_ratio("bakery", "bakeshop", 0.3)
        # Should match the standard Levenshtein ratio
        expected = levenshtein_ratio("bakery", "bakeshop")
        assert abs(score - expected) < 1e-6

    def test_early_exit_below_threshold(self):
        # "abc" vs "xyz" has ratio ~0.0 — should bail early with threshold=0.8
        score = fast_levenshtein_ratio("abc", "xyz", 0.8)
        assert score == 0.0

    def test_empty_strings(self):
        assert fast_levenshtein_ratio("", "hello", 0.5) == 0.0
        assert fast_levenshtein_ratio("hello", "", 0.5) == 0.0
        assert fast_levenshtein_ratio("", "", 0.5) == 0.0

    def test_exact_at_threshold(self):
        # "juans bakeshop" vs "juans bakeshop" → ratio 1.0, any threshold should pass
        score = fast_levenshtein_ratio("juans bakeshop", "juans bakeshop", 0.99)
        assert score == 1.0

    def test_high_threshold_rejects_fuzzy(self):
        # "juans bakery" vs "juans bakeshop" — similar but not identical
        score = fast_levenshtein_ratio("juans bakery", "juans bakeshop", 0.95)
        assert score == 0.0  # Can't reach 0.95

    def test_matches_standard_when_above_threshold(self):
        """When a pair is above threshold, fast and standard should agree."""
        pairs = [
            ("juan bakeshop", "juans bakeshop"),
            ("jc automotive", "jc automotive repair"),
            ("linas laundry", "linas laundry services"),
        ]
        for s1, s2 in pairs:
            standard = levenshtein_ratio(s1, s2)
            fast = fast_levenshtein_ratio(s1, s2, 0.5)
            assert abs(standard - fast) < 1e-6, f"Mismatch for {s1!r} vs {s2!r}"


# ── Pre-Tokenize Sort Tests ───────────────────────────────────────────────

class TestPreTokenizeSort:
    def test_basic_sorting(self):
        assert pre_tokenize_sort("bakeshop juan") == "bakeshop juan"
        assert pre_tokenize_sort("juan bakeshop") == "bakeshop juan"

    def test_case_insensitive(self):
        assert pre_tokenize_sort("Juan's BAKESHOP") == "bakeshop juan s"

    def test_empty_string(self):
        assert pre_tokenize_sort("") == ""

    def test_order_independence(self):
        s1 = pre_tokenize_sort("JC Automotive Repair")
        s2 = pre_tokenize_sort("Repair Automotive JC")
        assert s1 == s2


# ── Multi-Stage Match Tests ────────────────────────────────────────────────

class TestMultiStageMatch:
    def test_identical_strings(self):
        score = multi_stage_match("juans bakeshop", "juans bakeshop", 0.6)
        assert score == 1.0

    def test_similar_above_threshold(self):
        score = multi_stage_match("juans bakeshop", "juan bakeshop", 0.6)
        assert score > 0.8

    def test_length_ratio_early_exit(self):
        # Very different lengths → should exit at stage 1
        score = multi_stage_match("jb", "juan bakeshop in mamatid cabuyao city laguna", 0.6)
        assert score == 0.0

    def test_sequencematcher_early_exit(self):
        # Completely different strings of similar length
        score = multi_stage_match("xxxxxxxxx", "yyyyyyyyy", 0.6)
        assert score == 0.0

    def test_empty_strings(self):
        assert multi_stage_match("", "hello", 0.6) == 0.0
        assert multi_stage_match("hello", "", 0.6) == 0.0

    def test_matches_fast_levenshtein_when_passing(self):
        """When a pair passes all stages, result should match fast_levenshtein_ratio."""
        s1 = "linas laundry"
        s2 = "linas laundry services"
        ms = multi_stage_match(s1, s2, 0.5)
        fl = fast_levenshtein_ratio(s1, s2, 0.5)
        assert abs(ms - fl) < 1e-6

    def test_threshold_zero_passes_everything(self):
        # With threshold 0.0, the heuristic filters should not reject anything,
        # but the actual Levenshtein ratio is still computed — strings with SOME
        # similarity should produce a non-zero score.
        score = multi_stage_match("bakery", "bakeshop", 0.0)
        assert score > 0  # These strings share a prefix → non-zero ratio


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


# ── Pluggable Verification Strategy Tests ──────────────────────────────────

class TestVerificationStrategy:
    """Tests for the pluggable VerificationStrategy extension point."""

    def test_custom_strategy_is_used(self):
        """A custom strategy should replace the default hybrid_fuzzy_match."""

        class AlwaysVerifiedStrategy(VerificationStrategy):
            def compute_match_score(self, candidate_name, registry_name):
                return 1.0  # Everything matches perfectly

        verifier = BPLOVerifier(
            verification_strategy=AlwaysVerifiedStrategy(),
        )
        verifier.load_registry_from_list([{"name": "Any Registry Entry"}])

        result = verifier.verify("Completely Different Name")
        assert result["status"] == "Verified"
        assert result["score"] == 1.0

    def test_custom_strategy_rejects(self):
        """A custom strategy that always returns 0.0 should reject everything."""

        class NeverMatchStrategy(VerificationStrategy):
            def compute_match_score(self, candidate_name, registry_name):
                return 0.0  # Nothing ever matches

        verifier = BPLOVerifier(
            verification_strategy=NeverMatchStrategy(),
        )
        verifier.load_registry_from_list([{"name": "Juan's Bakeshop"}])

        result = verifier.verify("Juan's Bakeshop")
        assert result["status"] == "Unverified"
        assert result["score"] == 0.0

    def test_default_behavior_without_strategy(self):
        """Without a strategy, the default hybrid_fuzzy_match should be used."""
        verifier = BPLOVerifier()
        verifier.load_registry_from_list([{"name": "Juan's Bakeshop"}])

        result = verifier.verify("Juan's Bakeshop")
        assert result["status"] == "Verified"
        assert result["score"] == 1.0

    def test_strategy_receives_lowercased_names(self):
        """The strategy should receive lowercased candidate names."""
        received_args = []

        class SpyStrategy(VerificationStrategy):
            def compute_match_score(self, candidate_name, registry_name):
                received_args.append((candidate_name, registry_name))
                return 0.5

        verifier = BPLOVerifier(
            verification_strategy=SpyStrategy(),
        )
        verifier.load_registry_from_list([{"name": "Juan's Bakeshop"}])
        verifier.verify("JUAN'S BAKESHOP")

        assert len(received_args) == 1
        candidate, registry = received_args[0]
        assert candidate == "juan's bakeshop"
        assert registry == "juan's bakeshop"

    def test_strategy_with_batch_verification(self):
        """Custom strategy should also work with verify_batch."""

        class HighScoreStrategy(VerificationStrategy):
            def compute_match_score(self, candidate_name, registry_name):
                return 0.9  # High score for everything

        verifier = BPLOVerifier(
            verification_strategy=HighScoreStrategy(),
        )
        verifier.load_registry_from_list([{"name": "Registry Entry"}])

        profiles = [
            {"name": "Profile A"},
            {"name": "Profile B"},
        ]
        results = verifier.verify_batch(profiles)
        assert all(p["is_verified"] for p in results)

    def test_custom_strategy_subclass_enforcement(self):
        """VerificationStrategy is an ABC — can't instantiate directly."""
        with pytest.raises(TypeError):
            VerificationStrategy()


# ── FastMatchStrategy Tests ────────────────────────────────────────────────

class TestFastMatchStrategy:
    def test_exact_match(self):
        strategy = FastMatchStrategy(threshold=0.6)
        score = strategy.compute_match_score("juan's bakeshop", "juan's bakeshop")
        assert score == 1.0

    def test_order_independent(self):
        """Token sorting should make word order irrelevant."""
        strategy = FastMatchStrategy(threshold=0.6)
        score = strategy.compute_match_score("bakeshop juan", "juan bakeshop")
        assert score == 1.0

    def test_rejects_dissimilar(self):
        strategy = FastMatchStrategy(threshold=0.6)
        score = strategy.compute_match_score(
            "completely different business",
            "juan's bakeshop",
        )
        assert score == 0.0

    def test_with_verifier(self):
        """FastMatchStrategy should work end-to-end with BPLOVerifier."""
        verifier = BPLOVerifier(
            verification_strategy=FastMatchStrategy(threshold=0.6),
        )
        verifier.load_registry_from_list([
            {"name": "Juan's Bakeshop"},
            {"name": "JC Automotive Repair"},
        ])

        result = verifier.verify("Juan's Bakeshop")
        assert result["status"] == "Verified"

        result = verifier.verify("Unknown XYZ Business")
        assert result["status"] == "Unverified"

    def test_similar_but_not_exact(self):
        """Similar names should score above threshold when sorted forms are close."""
        strategy = FastMatchStrategy(threshold=0.5)
        # After pre_tokenize_sort: "bakeshop juans" vs "bakeshop linas"
        # These sorted forms differ only in the last word
        score = strategy.compute_match_score(
            "juans bakeshop",
            "juans bakeshop mamatid",
        )
        assert score > 0.5

    def test_threshold_controls_sensitivity(self):
        """Higher threshold should reject more pairs."""
        low = FastMatchStrategy(threshold=0.4)
        high = FastMatchStrategy(threshold=0.95)

        # After sorting: "bakeshop juans" vs "bakeshop linas"
        # These have similar lengths but differ in one word
        s1 = "juans bakeshop"
        s2 = "linas bakeshop"

        score_low = low.compute_match_score(s1, s2)
        score_high = high.compute_match_score(s1, s2)

        # Low threshold should produce a score; high threshold should reject
        assert score_low > 0.0
        assert score_high == 0.0
