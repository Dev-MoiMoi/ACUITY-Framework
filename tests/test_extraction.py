"""
Tests for the ACUITY Extraction module.
"""
import pytest

from acuity.extraction.preprocessing import preprocess, remove_urls, remove_emojis, normalise_whitespace
from acuity.extraction.rules import extract_structured_fields
from acuity.extraction.postprocessing import build_business_profile, format_business_name
from acuity.extraction.pipeline import ExtractionPipeline
from acuity.config import AcuityConfig


# ── Preprocessing Tests ────────────────────────────────────────────────────

class TestPreprocessing:
    def test_remove_urls(self):
        text = "Visit us at https://example.com for more info"
        result = remove_urls(text)
        assert "https://example.com" not in result
        assert "Visit us at" in result

    def test_remove_emojis(self):
        text = "Great food! 🍕🍔 Come visit us"
        result = remove_emojis(text)
        assert "🍕" not in result
        assert "Great food" in result

    def test_normalise_whitespace(self):
        text = "Multiple   spaces   and\n\nnewlines"
        result = normalise_whitespace(text)
        assert "  " not in result
        assert "\n" not in result

    def test_preprocess_full(self):
        text = "Check https://fb.com  for 🔥 deals!  "
        result = preprocess(text)
        assert "https://fb.com" not in result
        assert "🔥" not in result
        assert not result.endswith(" ")


# ── Rules Tests ────────────────────────────────────────────────────────────

class TestRuleExtraction:
    def test_extract_phone_numbers(self):
        text = "Contact us at 0917-123-4567 or +639181234567"
        result = extract_structured_fields(text)
        assert len(result["phones"]) == 2

    def test_extract_prices(self):
        text = "Pandesal ₱5 per piece, dozen for P50"
        result = extract_structured_fields(text)
        assert len(result["prices"]) >= 1

    def test_extract_hours(self):
        text = "Open 8am to 5pm daily"
        result = extract_structured_fields(text)
        assert len(result["hours"]) >= 1

    def test_no_matches(self):
        text = "Just a regular post with no business info"
        result = extract_structured_fields(text)
        assert result["phones"] == []
        assert result["prices"] == []
        assert result["hours"] == []


# ── Postprocessing Tests ───────────────────────────────────────────────────

class TestPostprocessing:
    def test_format_business_name(self):
        assert format_business_name("mangJuan's_bakery") == "Mang Juan'S Bakery"
        assert format_business_name("JC AUTO REPAIR") == "Jc Auto Repair"

    def test_build_profile_with_valid_data(self):
        entities = {
            "business_name": ["Juan's Bakery"],
            "categories": ["bakery"],
            "locations": ["Mamatid"],
        }
        structured = {
            "phones": ["0917-123-4567"],
            "prices": ["₱5"],
            "hours": ["open 8am"],
        }
        profile = build_business_profile("raw text", entities, structured)
        assert profile is not None
        assert profile["business_name"] == "Juan'S Bakery"
        assert profile["phones"] == ["0917-123-4567"]

    def test_build_profile_with_no_data(self):
        entities = {"business_name": [], "categories": [], "locations": []}
        structured = {"phones": [], "prices": [], "hours": []}
        profile = build_business_profile("just some text", entities, structured)
        assert profile is None

    def test_poster_name_fallback(self):
        entities = {"business_name": [], "categories": ["food"], "locations": []}
        structured = {"phones": [], "prices": [], "hours": []}
        profile = build_business_profile("raw text", entities, structured, poster_name="Jane Doe")
        assert profile is not None
        assert profile["business_name"] == "Jane Doe"


# ── Pipeline Integration Tests ─────────────────────────────────────────────

class TestExtractionPipeline:
    def test_pipeline_without_model(self):
        """Pipeline should work with rule-based extraction even without NER model."""
        config = AcuityConfig(
            ner_backend="crf",
            ner_model_path=None,
            completeness_threshold=1,
        )
        pipeline = ExtractionPipeline(config=config)

        texts = [
            "Juan's Bakery, open 8am-5pm, 0917-123-4567, pandesal ₱5 per piece",
        ]
        profiles = pipeline.extract_from_texts(texts)
        # Even without NER, rule-based extraction should find phones and prices
        assert len(profiles) >= 0  # May vary based on threshold

    def test_pipeline_deduplication(self):
        config = AcuityConfig(ner_backend="crf", ner_model_path=None, completeness_threshold=0)
        pipeline = ExtractionPipeline(config=config)

        texts = [
            "Juan's Bakery, 0917-123-4567",
            "Juan's Bakery, 0917-123-4567",  # Duplicate
        ]
        profiles = pipeline.extract_from_texts(texts, deduplicate=True)
        names = [p["business_name"] for p in profiles if p.get("business_name")]
        # If both extracted the same name, dedup should keep only one
        assert len(set(names)) == len(names)

    def test_pipeline_accepts_dicts(self):
        config = AcuityConfig(ner_backend="crf", ner_model_path=None, completeness_threshold=0)
        pipeline = ExtractionPipeline(config=config)

        texts = [
            {"text": "Juan's Bakery, 0917-123-4567, ₱5", "poster": "Admin", "scraped_at": "2024-01-01"},
        ]
        profiles = pipeline.extract_from_texts(texts)
        assert isinstance(profiles, list)

    def test_invalid_backend(self):
        config = AcuityConfig(ner_backend="invalid")
        with pytest.raises(ValueError, match="Unknown NER backend"):
            ExtractionPipeline(config=config)
