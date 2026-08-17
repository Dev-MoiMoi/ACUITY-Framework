"""
Tests for the ACUITY Recommendation module.
"""
import pytest

from acuity.recommendation import RecommendationEngine
from acuity.recommendation.vectorizer import CustomTfidfVectorizer, build_tfidf_matrix, transform_query
from acuity.recommendation.similarity import compute_cosine_scores
from acuity.recommendation.proximity import haversine_distance
from acuity.recommendation.ranker import rank_results
from acuity.config import AcuityConfig


# ── Vectorizer Tests ───────────────────────────────────────────────────────

class TestTfidfVectorizer:
    def test_fit_transform(self):
        docs = ["bakery bread pastry", "auto repair vulcanizing", "laundry wash fold"]
        vectorizer, vectors = build_tfidf_matrix(docs)
        assert len(vectors) == 3
        assert all(isinstance(v, dict) for v in vectors)
        assert len(vectorizer.vocabulary) > 0

    def test_transform_query(self):
        docs = ["bakery bread pastry", "auto repair vulcanizing"]
        vectorizer, _ = build_tfidf_matrix(docs)
        query_vec = transform_query(vectorizer, "bakery")
        assert len(query_vec) == 1
        assert isinstance(query_vec[0], dict)

    def test_stop_words_removed(self):
        vectorizer = CustomTfidfVectorizer()
        tokens = vectorizer._tokenize_and_ngrams("the is a bakery and shop")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "bakery" in tokens

    def test_bigrams_generated(self):
        vectorizer = CustomTfidfVectorizer(ngram_range=(1, 2))
        tokens = vectorizer._tokenize_and_ngrams("auto repair shop")
        assert "auto repair" in tokens
        assert "repair shop" in tokens


# ── Similarity Tests ───────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors(self):
        vectors = [{"term1": 1.0, "term2": 2.0}]
        query = [{"term1": 1.0, "term2": 2.0}]
        scores = compute_cosine_scores(vectors, query)
        assert abs(scores[0] - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        vectors = [{"term1": 1.0}]
        query = [{"term2": 1.0}]
        scores = compute_cosine_scores(vectors, query)
        assert scores[0] == 0.0

    def test_empty_query(self):
        vectors = [{"term1": 1.0}]
        scores = compute_cosine_scores(vectors, [])
        assert scores[0] == 0.0


# ── Proximity Tests ────────────────────────────────────────────────────────

class TestHaversine:
    def test_same_point(self):
        dist = haversine_distance(14.27, 121.12, 14.27, 121.12)
        assert dist == 0.0

    def test_known_distance(self):
        # Manila to Cebu is approximately 565 km
        dist = haversine_distance(14.5995, 120.9842, 10.3157, 123.8854)
        assert 500 < dist < 650

    def test_short_distance(self):
        # Two points ~1km apart
        dist = haversine_distance(14.2700, 121.1200, 14.2790, 121.1200)
        assert 0.5 < dist < 1.5


# ── Ranker Tests ───────────────────────────────────────────────────────────

class TestRanker:
    def test_ranking_order(self):
        profiles = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
        cosine_scores = [0.3, 0.9, 0.1]
        distances = [None, None, None]
        results = rank_results(profiles, cosine_scores, distances, top_k=3)
        assert results[0]["name"] == "B"
        assert results[-1]["name"] == "C"

    def test_top_k_limit(self):
        profiles = [{"name": f"Biz{i}"} for i in range(10)]
        cosine_scores = [0.1 * i for i in range(10)]
        distances = [None] * 10
        results = rank_results(profiles, cosine_scores, distances, top_k=3)
        assert len(results) == 3

    def test_proximity_influence(self):
        profiles = [{"name": "Close"}, {"name": "Far"}]
        cosine_scores = [0.5, 0.5]  # Same relevance
        distances = [0.1, 100.0]     # Very different distances
        results = rank_results(profiles, cosine_scores, distances, proximity_weight=0.5)
        assert results[0]["name"] == "Close"


# ── Engine Integration Tests ──────────────────────────────────────────────

class TestRecommendationEngine:
    def test_engine_basic(self):
        engine = RecommendationEngine()
        profiles = [
            {"name": "Juan's Bakery", "description": "bread and pastry"},
            {"name": "Auto Repair", "description": "car fix vulcanizing"},
        ]
        engine.set_profiles(profiles)
        results = engine.recommend("bakery bread")
        assert len(results) > 0
        assert results[0]["name"] == "Juan's Bakery"

    def test_engine_empty_profiles(self):
        engine = RecommendationEngine()
        results = engine.recommend("bakery")
        assert results == []

    def test_engine_with_location(self):
        engine = RecommendationEngine()
        profiles = [
            {"name": "Bakery A", "description": "bread", "latitude": 14.27, "longitude": 121.12},
            {"name": "Bakery B", "description": "bread", "latitude": 14.50, "longitude": 121.50},
        ]
        engine.set_profiles(profiles)
        results = engine.recommend("bread", user_lat=14.27, user_lon=121.12)
        assert results[0]["name"] == "Bakery A"  # Closer bakery should rank higher

    def test_engine_config_weights(self):
        config = AcuityConfig(relevance_weight=1.0, proximity_weight=0.0)
        engine = RecommendationEngine(config=config)
        assert engine.relevance_weight == 1.0
        assert engine.proximity_weight == 0.0
