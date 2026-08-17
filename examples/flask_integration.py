"""
ACUITY Framework — Flask Integration Example

Demonstrates how a third-party Flask web application can integrate
the ACUITY framework for business extraction, verification, and
recommendation via REST API endpoints.

This is similar to how the original ACUITY thesis application uses
the framework, but shows it can work with ANY Flask app.

Usage:
    pip install flask acuity-framework
    python examples/flask_integration.py
"""

from flask import Flask, request, jsonify

from acuity.extraction import ExtractionPipeline
from acuity.recommendation import RecommendationEngine
from acuity.verification import BPLOVerifier
from acuity.config import AcuityConfig


# ── Application Setup ──────────────────────────────────────────────────────
app = Flask(__name__)

# Configure ACUITY
config = AcuityConfig(
    ner_backend="crf",
    ner_model_path=None,  # Set your model path here
    relevance_weight=0.6,
    proximity_weight=0.4,
    default_top_k=10,
    fuzzy_match_threshold_verified=0.8,
    fuzzy_match_threshold_pending=0.6,
)

# Initialize framework components
pipeline = ExtractionPipeline(config=config)
engine = RecommendationEngine(config=config)
verifier = BPLOVerifier(config=config)

# In a real app, you'd load these from your database
SAMPLE_REGISTRY = [
    {"name": "Juan's Bakeshop", "address": "Mamatid, Cabuyao"},
    {"name": "JC Automotive Repair", "address": "Banay-Banay, Cabuyao"},
]
verifier.load_registry_from_list(SAMPLE_REGISTRY)


# ── API Routes ─────────────────────────────────────────────────────────────

@app.route("/api/extract", methods=["POST"])
def extract_profiles():
    """Extract business profiles from raw text posts.

    Request body: {"texts": ["post 1", "post 2", ...]}
    """
    data = request.get_json()
    texts = data.get("texts", [])

    if not texts:
        return jsonify({"error": "No texts provided"}), 400

    profiles = pipeline.extract_from_texts(texts)

    # Optionally verify against registry
    verified_profiles = verifier.verify_batch(profiles)

    return jsonify({
        "count": len(verified_profiles),
        "profiles": verified_profiles,
    })


@app.route("/api/recommend", methods=["GET"])
def recommend():
    """Recommend businesses based on a search query.

    Query params:
        q: Search query (required)
        lat: User latitude (optional)
        lon: User longitude (optional)
        top_k: Number of results (optional, default 10)
    """
    query = request.args.get("q", "")
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    user_lat = request.args.get("lat", type=float)
    user_lon = request.args.get("lon", type=float)
    top_k = request.args.get("top_k", type=int)

    results = engine.recommend(
        query=query,
        user_lat=user_lat,
        user_lon=user_lon,
        top_k=top_k,
    )

    return jsonify({
        "query": query,
        "count": len(results),
        "results": results,
    })


@app.route("/api/verify", methods=["POST"])
def verify_business():
    """Verify a business name against the BPLO registry.

    Request body: {"name": "Business Name"}
    """
    data = request.get_json()
    name = data.get("name", "")

    if not name:
        return jsonify({"error": "Business name is required"}), 400

    result = verifier.verify(name)
    return jsonify(result)


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pre-load some sample profiles for the recommendation engine
    sample_profiles = [
        {"name": "Mang Juan's Bakery", "description": "Fresh pandesal and ensaymada"},
        {"name": "JC Auto Repair", "description": "Vulcanizing and oil change"},
        {"name": "Lina's Laundry", "description": "Wash and fold services"},
    ]
    engine.set_profiles(sample_profiles)

    print("\n" + "=" * 60)
    print("ACUITY Framework — Flask Integration Demo")
    print("=" * 60)
    print("\nEndpoints:")
    print("  POST /api/extract   — Extract profiles from text")
    print("  GET  /api/recommend — Recommend businesses")
    print("  POST /api/verify    — Verify a business name")
    print()

    app.run(debug=True, port=5050)
