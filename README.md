# ACUITY Framework

**A**utomated **C**ommunity **U**nstructured **I**nformation to **T**argeted visibilit**Y**

A Python machine learning framework for extracting, verifying, and recommending local micro-enterprise profiles from unstructured community posts (e.g., Facebook groups, forums).

---

## 🚀 Installation

```bash
# Core framework (no heavy dependencies)
pip install acuity-framework

# With NLP support (nltk for CRF-based NER)
pip install acuity-framework[nlp]

# With Transformer NER (requires PyTorch)
pip install acuity-framework[transformers]

# With Facebook scraper
pip install acuity-framework[scraper]

# Everything
pip install acuity-framework[all]
```

### Local Development Install

```bash
git clone https://github.com/acuity-framework/acuity-framework.git
cd acuity-framework
pip install -e ".[dev]"
```

---

## 📦 Modules

| Module | Description |
|--------|-------------|
| `acuity.extraction` | NLP pipeline: preprocessing → NER → rule-based extraction → profile construction |
| `acuity.recommendation` | TF-IDF + cosine similarity + Haversine proximity ranking |
| `acuity.verification` | Business legitimacy verification via fuzzy matching (Levenshtein) |
| `acuity.scraper` | Facebook community group post scraper (optional) |

---

## 🔧 Quick Start

### 1. Extract Business Profiles from Text

```python
from acuity.extraction import ExtractionPipeline

pipeline = ExtractionPipeline()
profiles = pipeline.extract_from_texts([
    "Mang Juan's Bakery sa Mamatid, open 8am-5pm, 0917-123-4567, pandesal ₱5",
    "JC Auto Repair, vulcanizing, Brgy Banay-Banay, 0918-987-6543",
])

for p in profiles:
    print(f"{p['business_name']}: {p['phones']}, {p['hours']}")
```

### 2. Verify Against a Government Registry

```python
from acuity.verification import BPLOVerifier

verifier = BPLOVerifier()
verifier.load_registry_from_list([
    {"name": "Juan's Bakeshop", "address": "Mamatid"},
    {"name": "JC Automotive Repair", "address": "Banay-Banay"},
])

result = verifier.verify("Mang Juan's Bakery")
print(f"Status: {result['status']}, Score: {result['score']}")
# Output: Status: Pending Verification, Score: 0.65
```

### 3. Recommend Businesses

```python
from acuity.recommendation import RecommendationEngine

engine = RecommendationEngine()
engine.set_profiles([
    {"name": "Juan's Bakery", "description": "Fresh bread daily", "latitude": 14.27, "longitude": 121.12},
    {"name": "Auto Repair", "description": "Vulcanizing and oil change", "latitude": 14.26, "longitude": 121.11},
])

results = engine.recommend("bakery bread", user_lat=14.27, user_lon=121.12)
for r in results:
    print(f"{r['name']}: score={r['final_score']}, dist={r['distance_km']}km")
```

---

## 🔌 Extensibility (v3.0)

ACUITY v3.0 introduces **three pluggable extension points** via abstract base classes. You can inject custom implementations without modifying the framework's source code. All extension points are optional — existing code continues to work unchanged.

### Custom NER Backend

Replace the built-in CRF/Transformer NER with your own implementation:

```python
from acuity.extraction.interfaces import NERBackend
from acuity.extraction import ExtractionPipeline

class MyNERBackend(NERBackend):
    def extract_entities(self, text: str) -> dict:
        # Your custom entity extraction logic
        return {
            "business_name": ["Detected Name"],
            "categories": ["food"],
            "locations": ["Manila"],
        }

# Inject it — existing config-based NER is used when ner_backend=None (default)
pipeline = ExtractionPipeline(ner_backend=MyNERBackend())
profiles = pipeline.extract_from_texts(["Sample post text"])
```

### Custom Data Source

Replace the Facebook scraper with any data source (CSV, database, API, etc.):

```python
from acuity.scraper.interfaces import DataSource
from acuity.extraction import ExtractionPipeline

class MyDataSource(DataSource):
    def fetch_posts(self, sources: list[str], max_posts: int = 500) -> list[dict]:
        # Your custom data fetching logic
        return [{"text": "Post content", "poster": "Author Name"}]

# Inject it and use extract_from_source() for fetch + extract in one call
pipeline = ExtractionPipeline(data_source=MyDataSource())
profiles = pipeline.extract_from_source(sources=["my_source_id"])
```

### Custom Ranking Strategy

Replace TF-IDF + cosine similarity with your own text-relevance scoring:

```python
from acuity.recommendation.interfaces import RankingStrategy
from acuity.recommendation import RecommendationEngine

class MyRanking(RankingStrategy):
    def compute_scores(self, profiles: list[dict], query: str) -> list[float]:
        # Your custom relevance scoring logic
        return [1.0 if query.lower() in str(p).lower() else 0.0 for p in profiles]

# Inject it — Haversine proximity is still used alongside (it's a fixed formula)
engine = RecommendationEngine(ranking_strategy=MyRanking())
engine.set_profiles(profiles)
results = engine.recommend("bakery")
```

> **Note:** Haversine distance, the pipeline stage order (preprocess → NER → rules → postprocess), and Levenshtein fuzzy matching are intentionally **not** abstracted — they are fixed, correct algorithms with no legitimate variation.

See [`examples/demo_extensibility.py`](examples/demo_extensibility.py) for a complete end-to-end demo using all three extension points.

---

## ⚙️ Configuration

All settings are controlled via the `AcuityConfig` dataclass:

```python
from acuity.config import AcuityConfig

config = AcuityConfig(
    # NER settings
    ner_backend="crf",                    # "crf" or "transformer"
    ner_model_path="./models/crf.pkl",    # Path to your trained model

    # Recommendation weights
    relevance_weight=0.6,
    proximity_weight=0.4,
    default_top_k=10,

    # Verification thresholds
    fuzzy_match_threshold_verified=0.8,
    fuzzy_match_threshold_pending=0.6,
)
```

---

## 🌐 Integrating with Your Web Application

ACUITY is framework-agnostic. Here's how to use it with Flask:

```python
from flask import Flask, request, jsonify
from acuity.recommendation import RecommendationEngine

app = Flask(__name__)
engine = RecommendationEngine()

@app.route("/api/recommend")
def recommend():
    query = request.args.get("q", "")
    results = engine.recommend(query)
    return jsonify(results)
```

See [`examples/flask_integration.py`](examples/flask_integration.py) for a complete working example.

---

## 🧪 Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## 📁 Project Structure

```
acuity-framework/
├── pyproject.toml          # Package configuration
├── README.md
├── LICENSE
├── acuity/
│   ├── __init__.py         # Public API
│   ├── config.py           # AcuityConfig dataclass
│   ├── utils.py            # Levenshtein similarity utilities
│   ├── extraction/         # NLP extraction pipeline
│   │   ├── pipeline.py     # ExtractionPipeline class
│   │   ├── interfaces.py   # NERBackend ABC (extensibility)
│   │   ├── preprocessing.py
│   │   ├── ner_crf.py
│   │   ├── ner_transformer.py
│   │   ├── rules.py
│   │   └── postprocessing.py
│   ├── recommendation/     # Recommendation engine
│   │   ├── engine.py       # RecommendationEngine class
│   │   ├── interfaces.py   # RankingStrategy ABC (extensibility)
│   │   ├── vectorizer.py   # TF-IDF vectorizer
│   │   ├── similarity.py   # Cosine similarity
│   │   ├── proximity.py    # Haversine distance (fixed, not abstracted)
│   │   └── ranker.py       # Combined ranking
│   ├── verification/       # Business verification
│   │   └── bplo.py         # BPLOVerifier class
│   └── scraper/            # Data collection (optional)
│       ├── scraper.py      # FacebookScraper class
│       ├── interfaces.py   # DataSource ABC (extensibility)
│       └── utils.py
├── examples/
│   ├── basic_extraction.py
│   ├── basic_recommendation.py
│   ├── flask_integration.py
│   ├── custom_ner_backend.py       # Example: KeywordNERBackend
│   ├── custom_data_source.py       # Example: CSVDataSource
│   ├── custom_ranking_strategy.py  # Example: KeywordMatchRanking
│   └── demo_extensibility.py       # Combined end-to-end demo
└── tests/
    ├── test_extraction.py
    ├── test_recommendation.py
    └── test_verification.py
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🎓 Academic Reference

This framework was developed as part of an academic thesis at the College of Computing Studies. The core algorithms implement:

- **TF-IDF Vectorization** with log-normalised term frequency and inverse document frequency
- **Cosine Similarity** for textual relevance scoring
- **Haversine Formula** for geographic proximity computation
- **CRF (Conditional Random Field)** for Named Entity Recognition with BIO tagging
- **Levenshtein Distance** for fuzzy string matching in business verification
