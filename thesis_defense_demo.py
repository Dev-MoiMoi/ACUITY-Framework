import sys
import os
import json
from typing import List, Dict

# Import the core framework
from acuity.config import AcuityConfig
from acuity.extraction.pipeline import ExtractionPipeline
from acuity.recommendation import RecommendationEngine
from acuity.verification import BPLOVerifier

# Import the 3 extension interfaces we will implement
from acuity.extraction.interfaces import NERBackend
from acuity.scraper.interfaces import DataSource
from acuity.recommendation.interfaces import RankingStrategy

# ==============================================================================
# EXTENSION POINT 1: Custom Data Source
# Instead of scraping Facebook, we read from a local JSON dataset.
# ==============================================================================
class JSONDataSource(DataSource):
    """Custom Data Source that reads community posts from a JSON file."""
    
    def fetch_posts(self, sources: List[str], max_posts: int = 500) -> List[Dict]:
        print("    --> [System Log: Triggering Custom Extension] JSONDataSource.fetch_posts()")
        posts = []
        for source_path in sources:
            if not os.path.exists(source_path):
                print(f"    --> [Error] Could not find {source_path}")
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
# Instead of the default CRF/Transformer NER, we use a custom heuristic rule-based NER.
# ==============================================================================
class RuleBasedNERBackend(NERBackend):
    """Custom NER that uses specific heuristics and dictionaries for this demo."""
    
    def __init__(self):
        self.known_categories = ["tech", "food", "repair", "services", "retail"]
        self.known_locations = ["manila", "cabuyao", "laguna", "makati", "quezon city"]

    def extract_entities(self, text: str) -> Dict:
        print("    --> [System Log: Triggering Custom Extension] RuleBasedNERBackend.extract_entities()")
        text_lower = text.lower()
        
        # Heuristic 1: Extract categories based on known dictionary
        categories = [cat for cat in self.known_categories if cat in text_lower]
        
        # Heuristic 2: Extract locations based on known dictionary
        locations = [loc.title() for loc in self.known_locations if loc in text_lower]
        
        # Heuristic 3: Extract business name (simple heuristic: first few title-cased words before a comma)
        business_names = []
        words = text.split(',')
        if words:
            potential_name = " ".join([w for w in words[0].split() if w.istitle()])
            if potential_name:
                business_names.append(potential_name)
                
        return {
            "business_name": business_names,
            "categories": categories,
            "locations": locations,
        }


# ==============================================================================
# EXTENSION POINT 3: Custom Ranking Strategy
# Instead of default TF-IDF + Cosine Similarity, we use a simple Jaccard Index.
# ==============================================================================
class JaccardRankingStrategy(RankingStrategy):
    """Custom Ranking that scores based on Jaccard similarity of words."""
    
    def compute_scores(self, profiles: List[Dict], query: str) -> List[float]:
        print("    --> [System Log: Triggering Custom Extension] JaccardRankingStrategy.compute_scores()")
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
            jaccard_score = len(intersection) / len(union)
            scores.append(jaccard_score)
            
        return scores


# ==============================================================================
# MAIN DEMO SYSTEM
# ==============================================================================
def create_sample_data(json_file_path: str):
    """Creates a sample JSON file for the custom data source to read."""
    sample_data = [
        {
            "content": "Kirt's Tech Repair Shop, Cabuyao. We fix laptops and phones! Open daily.",
            "author": "Kirt",
            "date": "2026-09-24"
        },
        {
            "content": "Laguna Food Hub, offering the best silog meals in Laguna. Visit us!",
            "author": "Chef Joy",
            "date": "2026-09-23"
        }
    ]
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=4)


def run_demo():
    print("="*70)
    print(" 🎓 ACUITY FRAMEWORK: THESIS DEFENSE EXTENSIBILITY DEMO")
    print("="*70)
    print("This demo proves the framework's architecture by bypassing all defaults")
    print("and injecting custom implementations into the 3 extension points.")
    print("="*70)

    # 1. Setup sample data
    data_file = "defense_sample_data.json"
    create_sample_data(data_file)
    
    # 2. Initialize Framework with Custom Extensions
    print("\n[INITIALIZING FRAMEWORK WITH EXTENSIONS...]")
    config = AcuityConfig(completeness_threshold=1)
    
    # Injecting custom implementations!
    pipeline = ExtractionPipeline(
        config=config,
        ner_backend=RuleBasedNERBackend(),  # Extension 1
        data_source=JSONDataSource()        # Extension 2
    )
    
    engine = RecommendationEngine(
        config=AcuityConfig(relevance_weight=1.0, proximity_weight=0.0), # Rely purely on our custom ranking for demo
        ranking_strategy=JaccardRankingStrategy() # Extension 3
    )
    
    while True:
        print("\n" + "-"*50)
        print(" DEMO MENU:")
        print(" 1. Run Pipeline (Extract from Custom Data Source & Custom NER)")
        print(" 2. Run Recommendation (Search using Custom Ranking Strategy)")
        print(" 3. Exit")
        choice = input("Select an option: ").strip()
        
        if choice == '1':
            print("\nExecuting Pipeline.extract_from_source()...")
            # This triggers JSONDataSource and RuleBasedNERBackend
            extracted_profiles = pipeline.extract_from_source(sources=[data_file])
            
            print(f"\n✅ Extracted {len(extracted_profiles)} profiles:")
            for idx, p in enumerate(extracted_profiles, 1):
                print(f"  {idx}. {p.get('business_name', 'Unnamed')}")
                print(f"     Categories: {p.get('categories', [])}")
                print(f"     Locations:  {p.get('locations', [])}")
                
            # Feed them to the recommendation engine for step 2
            rec_profiles = []
            for p in extracted_profiles:
                rec_profiles.append({
                    "name": p.get("business_name", "Unknown"),
                    "categories": p.get("categories", []),
                    "latitude": 14.27,
                    "longitude": 121.12,
                })
            engine.set_profiles(rec_profiles)
            
        elif choice == '2':
            query = input("\nEnter search query (e.g., 'tech repair cabuyao'): ").strip()
            print("\nExecuting RecommendationEngine.recommend()...")
            # This triggers JaccardRankingStrategy
            results = engine.recommend(query, user_lat=14.27, user_lon=121.12, top_k=3)
            
            print(f"\n✅ Found {len(results)} matches for '{query}':")
            for idx, r in enumerate(results, 1):
                print(f"  {idx}. {r['name']}")
                print(f"     Custom Relevance Score (Jaccard): {r['relevance_score']:.4f}")
                
        elif choice == '3':
            print("\nExiting demo. Good luck with the defense!")
            if os.path.exists(data_file):
                os.remove(data_file)
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    run_demo()
