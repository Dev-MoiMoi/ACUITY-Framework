import os
import json
from typing import List, Dict
from flask import Flask, render_template_string, request, jsonify

# --- ACUITY Core Imports ---
from acuity.config import AcuityConfig
from acuity.extraction.pipeline import ExtractionPipeline
from acuity.recommendation import RecommendationEngine

# --- ACUITY Extension Interfaces ---
from acuity.extraction.interfaces import NERBackend
from acuity.scraper.interfaces import DataSource
from acuity.recommendation.interfaces import RankingStrategy

# ==============================================================================
# EXTENSION POINT 1: Custom Data Source
# ==============================================================================
class JSONDataSource(DataSource):
    """Bypasses the Facebook scraper to read posts directly from a local JSON file."""
    def fetch_posts(self, sources: List[str], max_posts: int = 500) -> List[Dict]:
        posts = []
        for source_path in sources:
            if not os.path.exists(source_path): 
                continue
            with open(source_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    if len(posts) >= max_posts: 
                        break
                    posts.append({
                        "text": item.get("content", ""),
                        "poster": item.get("author", "Unknown"),
                        "scraped_at": item.get("date", "")
                    })
        return posts

# ==============================================================================
# EXTENSION POINT 2: Custom NER Backend
# ==============================================================================
class RuleBasedNERBackend(NERBackend):
    """Bypasses the ML models (CRF/Transformers) in favor of simple keyword heuristics."""
    def __init__(self):
        self.known_categories = ["tech", "food", "repair", "services", "retail", "bakery", "auto"]
        self.known_locations = ["manila", "cabuyao", "laguna", "makati", "quezon city", "mamatid"]

    def extract_entities(self, text: str) -> Dict:
        text_lower = text.lower()
        categories = [cat for cat in self.known_categories if cat in text_lower]
        locations = [loc.title() for loc in self.known_locations if loc in text_lower]
        
        # Simple heuristic for business name
        business_names = []
        words = text.split(',')
        if words:
            # Grab title-cased words at the start
            potential_name = " ".join([w for w in words[0].split() if w.istitle() or "'" in w])
            if potential_name:
                business_names.append(potential_name.strip())
                
        return {
            "business_name": business_names,
            "categories": categories,
            "locations": locations,
        }

# ==============================================================================
# EXTENSION POINT 3: Custom Ranking Strategy
# ==============================================================================
class JaccardRankingStrategy(RankingStrategy):
    """Bypasses TF-IDF + Cosine Similarity and uses the Jaccard index (word overlap)."""
    def compute_scores(self, profiles: List[Dict], query: str) -> List[float]:
        query_words = set(query.lower().split())
        if not query_words:
            return [0.0] * len(profiles)
            
        scores = []
        for profile in profiles:
            name = profile.get("name", "") or profile.get("business_name", "")
            desc = profile.get("description", "")
            cats = " ".join(profile.get("categories", []))
            
            profile_text = f"{name} {desc} {cats}".lower()
            profile_words = set(profile_text.split())
            
            if not profile_words:
                scores.append(0.0)
                continue
                
            intersection = query_words.intersection(profile_words)
            union = query_words.union(profile_words)
            scores.append(len(intersection) / len(union))
            
        return scores


# ==============================================================================
# FLASK APPLICATION SETUP
# ==============================================================================
app = Flask(__name__)

# Initialize Framework with Custom Extensions Injected
config = AcuityConfig(completeness_threshold=1)
pipeline = ExtractionPipeline(
    config=config,
    ner_backend=RuleBasedNERBackend(),   # Inject NER
    data_source=JSONDataSource()         # Inject Data Source
)

engine = RecommendationEngine(
    config=AcuityConfig(relevance_weight=1.0, proximity_weight=0.0), 
    ranking_strategy=JaccardRankingStrategy() # Inject Ranking
)

# Set up some dummy JSON data for the DataSource to read
DATA_FILE = "webapp_sample_data.json"
def setup_sample_data():
    sample_data = [
        {"content": "Mang Juan's Bakery, located at Mamatid. Fresh pandesal everyday!", "author": "Juan", "date": "2026-09-24"},
        {"content": "Kirt's Tech Repair Shop, Cabuyao City. We fix broken laptops and phones!", "author": "Kirt", "date": "2026-09-23"},
        {"content": "Lina's Laundry Services, Brgy Marinig. Wash and fold.", "author": "Lina", "date": "2026-09-22"}
    ]
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f)
setup_sample_data()


# ==============================================================================
# SINGLE-PAGE FRONTEND
# ==============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ACUITY Framework - Defense Demo</title>
    <style>
        body { font-family: -apple-system, system-ui, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; background: #f5f7fa; color: #333; }
        .card { background: white; padding: 25px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 25px; border-left: 5px solid #3498db; }
        .card-green { border-left-color: #2ecc71; }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 5px; }
        h2 { color: #2c3e50; margin-top: 0; }
        .subtitle { text-align: center; color: #7f8c8d; margin-bottom: 40px; font-size: 1.1em; }
        .badge { background: #e74c3c; color: white; padding: 4px 8px; border-radius: 4px; font-size: 13px; font-family: monospace; }
        button { background: #3498db; color: white; border: none; padding: 12px 20px; border-radius: 5px; cursor: pointer; font-size: 15px; font-weight: bold; transition: background 0.2s;}
        button:hover { background: #2980b9; }
        button.green-btn { background: #2ecc71; }
        button.green-btn:hover { background: #27ae60; }
        input[type="text"] { padding: 12px; width: calc(100% - 130px); border: 2px solid #ecf0f1; border-radius: 5px; font-size: 15px; outline: none; }
        input[type="text"]:focus { border-color: #3498db; }
        pre { background: #282c34; color: #abb2bf; padding: 15px; border-radius: 5px; border: none; overflow-x: auto; font-size: 14px; margin-top: 15px;}
        .flex-row { display: flex; gap: 10px; }
    </style>
</head>
<body>
    <h1>🎓 ACUITY Extensibility Web Demo</h1>
    <p class="subtitle">A visual proof that the architecture supports fully custom injected modules.</p>

    <div class="card">
        <h2>1. Extraction Pipeline</h2>
        <p>Instead of the Facebook scraper and CRF models, this step delegates entirely to the custom injected <span class="badge">JSONDataSource</span> and <span class="badge">RuleBasedNERBackend</span> components.</p>
        <button onclick="extract()">Trigger Extraction Pipeline</button>
        <div id="extraction-result"></div>
    </div>

    <div class="card card-green">
        <h2>2. Recommendation Engine</h2>
        <p>Instead of TF-IDF and Cosine Similarity, this step scores relevance using the custom injected <span class="badge">JaccardRankingStrategy</span>.</p>
        <div class="flex-row">
            <input type="text" id="query" placeholder="e.g., tech repair">
            <button class="green-btn" onclick="recommend()">Search Profiles</button>
        </div>
        <div id="recommendation-result"></div>
    </div>

    <script>
        async function extract() {
            const resultDiv = document.getElementById('extraction-result');
            resultDiv.innerHTML = "<p style='color:#7f8c8d; font-weight:bold;'>Running custom extraction pipeline...</p>";
            const res = await fetch('/api/extract', { method: 'POST' });
            const data = await res.json();
            resultDiv.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
        }

        async function recommend() {
            const q = document.getElementById('query').value;
            if(!q) return alert("Please enter a query");
            
            const resultDiv = document.getElementById('recommendation-result');
            resultDiv.innerHTML = "<p style='color:#7f8c8d; font-weight:bold;'>Ranking with Jaccard Index...</p>";
            const res = await fetch(`/api/recommend?q=${encodeURIComponent(q)}`);
            const data = await res.json();
            resultDiv.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/extract', methods=['POST'])
def api_extract():
    # Framework triggers the injected Data Source and NER
    profiles = pipeline.extract_from_source(sources=[DATA_FILE])
    
    # Push profiles into the recommendation engine for step 2
    rec_profiles = []
    for p in profiles:
        rec_profiles.append({
            "name": p.get("business_name", "Unknown"),
            "categories": p.get("categories", []),
            "latitude": 14.27,
            "longitude": 121.12,
        })
    engine.set_profiles(rec_profiles)
    
    return jsonify({
        "System Output": f"Extracted {len(profiles)} profiles bypassing defaults.",
        "Active DataSource": "JSONDataSource",
        "Active NER Backend": "RuleBasedNERBackend",
        "Extracted Profiles": profiles
    })

@app.route('/api/recommend', methods=['GET'])
def api_recommend():
    query = request.args.get('q', '')
    
    # Framework triggers the injected Ranking Strategy
    results = engine.recommend(query, user_lat=14.27, user_lon=121.12, top_k=3)
    
    return jsonify({
        "System Output": f"Scored and ranked candidates using Jaccard Similarity.",
        "Active Ranking Strategy": "JaccardRankingStrategy",
        "Query": query,
        "Ranked Results": results
    })

if __name__ == '__main__':
    print("=====================================================")
    print("Starting ACUITY Defense Web App Demo")
    print("Open http://127.0.0.1:5000 in your browser.")
    print("=====================================================")
    app.run(debug=True, port=5000)
