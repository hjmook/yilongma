"""
Golden Set Evaluation Script
Compares Fast vs Thinking model responses against real Elon Musk interview answers
"""

import requests
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np

# Try to import advanced NLP libraries, with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Using basic similarity metrics.")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not installed. Using basic similarity metrics.")

try:
    from difflib import SequenceMatcher
    DIFFLIB_AVAILABLE = True
except ImportError:
    DIFFLIB_AVAILABLE = False


FAST_SERVER = "http://localhost:5001/predict"
THINKING_SERVER = "http://localhost:5055/predict"
GOLDEN_FILE = "golden.json"


def load_golden_responses() -> List[Dict]:
    """Load golden responses from JSON file"""
    with open(GOLDEN_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['qa_pairs']


def get_model_response(server_url: str, query: str, user_id: str) -> Dict:
    """Get response from a model server"""
    try:
        start = time.time()
        response = requests.post(
            server_url,
            json={"input": query, "user_id": user_id, "use_rag": True},
            timeout=60
        )
        latency = (time.time() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "output": data.get("output", ""),
                "rag_used": data.get("rag_used", False),
                "latency_ms": latency
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "latency_ms": latency
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "latency_ms": 0
        }


def calculate_text_similarity(text1: str, text2: str) -> Dict[str, float]:
    """Calculate various similarity metrics between two texts"""
    similarities = {}
    
    # 1. Basic token overlap (Jaccard similarity)
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if tokens1 or tokens2:
        jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2)
        similarities['jaccard'] = jaccard
    else:
        similarities['jaccard'] = 0.0
    
    # 2. Sequence matcher (considers order)
    if DIFFLIB_AVAILABLE:
        matcher = SequenceMatcher(None, text1.lower(), text2.lower())
        similarities['sequence_match'] = matcher.ratio()
    
    # 3. Length ratio (penalize very different lengths)
    len1, len2 = len(text1), len(text2)
    if len1 and len2:
        length_ratio = min(len1, len2) / max(len1, len2)
        similarities['length_ratio'] = length_ratio
    else:
        similarities['length_ratio'] = 0.0
    
    return similarities


def calculate_semantic_similarity(text1: str, text2: str, model) -> float:
    """Calculate semantic similarity using sentence transformers"""
    if not SENTENCE_TRANSFORMER_AVAILABLE or not SKLEARN_AVAILABLE:
        return 0.0
    
    try:
        # Encode texts to embeddings
        embeddings = model.encode([text1, text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    except Exception as e:
        print(f"⚠️  Semantic similarity error: {e}")
        return 0.0


def calculate_style_metrics(text: str, golden_text: str) -> Dict[str, float]:
    """Calculate style-based metrics"""
    metrics = {}
    
    # Average word length (Elon tends to use simple words)
    words = text.split()
    golden_words = golden_text.split()
    
    if words:
        avg_word_len = np.mean([len(w) for w in words])
        golden_avg_word_len = np.mean([len(w) for w in golden_words]) if golden_words else 0
        
        # Normalize difference (closer to 1 is better)
        if golden_avg_word_len > 0:
            word_len_similarity = 1 - abs(avg_word_len - golden_avg_word_len) / max(avg_word_len, golden_avg_word_len)
            metrics['word_length_similarity'] = max(0, word_len_similarity)
        else:
            metrics['word_length_similarity'] = 0.0
    
    # Sentence count
    text_sentences = text.count('.') + text.count('!') + text.count('?')
    golden_sentences = golden_text.count('.') + golden_text.count('!') + golden_text.count('?')
    
    if max(text_sentences, golden_sentences) > 0:
        sentence_similarity = 1 - abs(text_sentences - golden_sentences) / max(text_sentences, golden_sentences)
        metrics['sentence_structure_similarity'] = max(0, sentence_similarity)
    else:
        metrics['sentence_structure_similarity'] = 0.0
    
    return metrics


def evaluate_response(model_response: str, golden_response: str, semantic_model=None) -> Dict:
    """Evaluate a single response against golden response"""
    
    # Text-based similarities
    text_sims = calculate_text_similarity(model_response, golden_response)
    
    # Semantic similarity (if available)
    semantic_sim = 0.0
    if semantic_model:
        semantic_sim = calculate_semantic_similarity(model_response, golden_response, semantic_model)
    
    # Style metrics
    style_metrics = calculate_style_metrics(model_response, golden_response)
    
    # Calculate overall score (weighted average)
    if SENTENCE_TRANSFORMER_AVAILABLE:
        # With semantic similarity
        overall = (
            text_sims.get('jaccard', 0) * 0.2 +
            text_sims.get('sequence_match', 0) * 0.2 +
            semantic_sim * 0.4 +
            style_metrics.get('word_length_similarity', 0) * 0.1 +
            style_metrics.get('sentence_structure_similarity', 0) * 0.1
        )
    else:
        # Without semantic similarity
        overall = (
            text_sims.get('jaccard', 0) * 0.4 +
            text_sims.get('sequence_match', 0) * 0.3 +
            style_metrics.get('word_length_similarity', 0) * 0.15 +
            style_metrics.get('sentence_structure_similarity', 0) * 0.15
        )
    
    return {
        'overall_similarity': overall,
        'text_similarity': text_sims,
        'semantic_similarity': semantic_sim,
        'style_metrics': style_metrics
    }


def run_evaluation():
    """Main evaluation function"""
    
    print("=" * 80)
    print("GOLDEN SET EVALUATION: Fast vs Thinking Model")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load semantic model if available
    semantic_model = None
    if SENTENCE_TRANSFORMER_AVAILABLE:
        print("Loading semantic similarity model...")
        try:
            semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Semantic model loaded\n")
        except Exception as e:
            print(f"⚠️  Could not load semantic model: {e}\n")
    
    # Load golden responses
    print(f"Loading golden responses from {GOLDEN_FILE}...")
    golden_pairs = load_golden_responses()
    print(f"✅ Loaded {len(golden_pairs)} golden Q&A pairs\n")
    
    results = {
        "test_date": datetime.now().isoformat(),
        "total_pairs": len(golden_pairs),
        "evaluations": [],
        "summary": {
            "fast": {
                "avg_overall_similarity": 0,
                "avg_text_similarity": 0,
                "avg_semantic_similarity": 0,
                "avg_response_time": 0,
                "successful_responses": 0
            },
            "thinking": {
                "avg_overall_similarity": 0,
                "avg_text_similarity": 0,
                "avg_semantic_similarity": 0,
                "avg_response_time": 0,
                "successful_responses": 0
            }
        }
    }
    
    # Process each golden pair
    for i, pair in enumerate(golden_pairs, 1):
        query = pair['query']
        golden_response = pair['response']
        
        print(f"\n{'='*80}")
        print(f"Question {i}/{len(golden_pairs)}")
        print(f"{'='*80}")
        print(f"❓ {query}")
        print(f"\n🎯 Golden Response:")
        print(f"   {golden_response[:150]}{'...' if len(golden_response) > 150 else ''}")
        
        # Get responses from both models
        print(f"\n⚡ Testing Fast Model...")
        fast_result = get_model_response(FAST_SERVER, query, f"eval_fast_{i}")
        
        print(f"🧠 Testing Thinking Model...")
        thinking_result = get_model_response(THINKING_SERVER, query, f"eval_thinking_{i}")
        
        evaluation_result = {
            "question_id": i,
            "query": query,
            "golden_response": golden_response,
            "fast": {},
            "thinking": {}
        }
        
        # Evaluate Fast Model
        if fast_result["success"]:
            print(f"\n📊 Fast Model Response:")
            print(f"   {fast_result['output'][:150]}{'...' if len(fast_result['output']) > 150 else ''}")
            
            fast_eval = evaluate_response(fast_result['output'], golden_response, semantic_model)
            evaluation_result["fast"] = {
                "response": fast_result['output'],
                "latency_ms": fast_result['latency_ms'],
                "evaluation": fast_eval
            }
            
            print(f"   ├─ Overall Similarity: {fast_eval['overall_similarity']:.3f}")
            print(f"   ├─ Jaccard Similarity: {fast_eval['text_similarity'].get('jaccard', 0):.3f}")
            if semantic_model:
                print(f"   ├─ Semantic Similarity: {fast_eval['semantic_similarity']:.3f}")
            print(f"   └─ Response Time: {fast_result['latency_ms']:.0f}ms")
            
            results["summary"]["fast"]["successful_responses"] += 1
            results["summary"]["fast"]["avg_overall_similarity"] += fast_eval['overall_similarity']
            results["summary"]["fast"]["avg_response_time"] += fast_result['latency_ms']
        else:
            print(f"   ❌ Error: {fast_result['error']}")
            evaluation_result["fast"] = {"error": fast_result['error']}
        
        # Evaluate Thinking Model
        if thinking_result["success"]:
            print(f"\n📊 Thinking Model Response:")
            print(f"   {thinking_result['output'][:150]}{'...' if len(thinking_result['output']) > 150 else ''}")
            
            thinking_eval = evaluate_response(thinking_result['output'], golden_response, semantic_model)
            evaluation_result["thinking"] = {
                "response": thinking_result['output'],
                "latency_ms": thinking_result['latency_ms'],
                "evaluation": thinking_eval
            }
            
            print(f"   ├─ Overall Similarity: {thinking_eval['overall_similarity']:.3f}")
            print(f"   ├─ Jaccard Similarity: {thinking_eval['text_similarity'].get('jaccard', 0):.3f}")
            if semantic_model:
                print(f"   ├─ Semantic Similarity: {thinking_eval['semantic_similarity']:.3f}")
            print(f"   └─ Response Time: {thinking_result['latency_ms']:.0f}ms")
            
            results["summary"]["thinking"]["successful_responses"] += 1
            results["summary"]["thinking"]["avg_overall_similarity"] += thinking_eval['overall_similarity']
            results["summary"]["thinking"]["avg_response_time"] += thinking_result['latency_ms']
        else:
            print(f"   ❌ Error: {thinking_result['error']}")
            evaluation_result["thinking"] = {"error": thinking_result['error']}
        
        # Show winner for this question
        if fast_result["success"] and thinking_result["success"]:
            fast_score = evaluation_result["fast"]["evaluation"]["overall_similarity"]
            thinking_score = evaluation_result["thinking"]["evaluation"]["overall_similarity"]
            
            if fast_score > thinking_score:
                print(f"\n🏆 Winner: Fast Model (score: {fast_score:.3f} vs {thinking_score:.3f})")
            elif thinking_score > fast_score:
                print(f"\n🏆 Winner: Thinking Model (score: {thinking_score:.3f} vs {fast_score:.3f})")
            else:
                print(f"\n🤝 Tie (score: {fast_score:.3f})")
        
        results["evaluations"].append(evaluation_result)
        
        time.sleep(1)  # Brief pause between queries
    
    # Calculate final averages
    n_fast = results["summary"]["fast"]["successful_responses"]
    n_thinking = results["summary"]["thinking"]["successful_responses"]
    
    if n_fast > 0:
        results["summary"]["fast"]["avg_overall_similarity"] /= n_fast
        results["summary"]["fast"]["avg_response_time"] /= n_fast
    
    if n_thinking > 0:
        results["summary"]["thinking"]["avg_overall_similarity"] /= n_thinking
        results["summary"]["thinking"]["avg_response_time"] /= n_thinking
    
    # Print final summary
    print("\n\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    print(f"\n⚡ Fast Model:")
    print(f"   ├─ Successful Responses: {n_fast}/{len(golden_pairs)}")
    print(f"   ├─ Avg Overall Similarity: {results['summary']['fast']['avg_overall_similarity']:.3f}")
    print(f"   └─ Avg Response Time: {results['summary']['fast']['avg_response_time']:.0f}ms")
    
    print(f"\n🧠 Thinking Model:")
    print(f"   ├─ Successful Responses: {n_thinking}/{len(golden_pairs)}")
    print(f"   ├─ Avg Overall Similarity: {results['summary']['thinking']['avg_overall_similarity']:.3f}")
    print(f"   └─ Avg Response Time: {results['summary']['thinking']['avg_response_time']:.0f}ms")
    
    # Determine overall winner
    print(f"\n" + "=" * 80)
    if n_fast > 0 and n_thinking > 0:
        fast_score = results['summary']['fast']['avg_overall_similarity']
        thinking_score = results['summary']['thinking']['avg_overall_similarity']
        
        if fast_score > thinking_score:
            improvement = ((fast_score - thinking_score) / thinking_score) * 100
            print(f"🏆 WINNER: Fast Model")
            print(f"   {improvement:.1f}% more similar to real Elon responses")
        elif thinking_score > fast_score:
            improvement = ((thinking_score - fast_score) / fast_score) * 100
            print(f"🏆 WINNER: Thinking Model")
            print(f"   {improvement:.1f}% more similar to real Elon responses")
        else:
            print(f"🤝 TIE: Both models equally similar to real Elon")
    print("=" * 80)
    
    # Save detailed results
    filename = f"golden_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Detailed results saved to: {filename}")
    
    # Optionally create a CSV for easier analysis
    create_csv_summary(results, filename.replace('.json', '.csv'))


def create_csv_summary(results: Dict, csv_filename: str):
    """Create a CSV summary of results for easier analysis"""
    try:
        import csv
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Question_ID',
                'Query',
                'Fast_Similarity',
                'Thinking_Similarity',
                'Fast_Response_Time_ms',
                'Thinking_Response_Time_ms',
                'Winner'
            ])
            
            # Data rows
            for eval_result in results['evaluations']:
                qid = eval_result['question_id']
                query = eval_result['query'][:50] + '...' if len(eval_result['query']) > 50 else eval_result['query']
                
                fast_sim = eval_result['fast'].get('evaluation', {}).get('overall_similarity', 0)
                thinking_sim = eval_result['thinking'].get('evaluation', {}).get('overall_similarity', 0)
                
                fast_time = eval_result['fast'].get('latency_ms', 0)
                thinking_time = eval_result['thinking'].get('latency_ms', 0)
                
                winner = 'Fast' if fast_sim > thinking_sim else ('Thinking' if thinking_sim > fast_sim else 'Tie')
                
                writer.writerow([
                    qid,
                    query,
                    f"{fast_sim:.3f}",
                    f"{thinking_sim:.3f}",
                    f"{fast_time:.0f}",
                    f"{thinking_time:.0f}",
                    winner
                ])
        
        print(f"✅ CSV summary saved to: {csv_filename}")
    except Exception as e:
        print(f"⚠️  Could not create CSV: {e}")


if __name__ == "__main__":
    print("\n📊 Golden Set Evaluation Script")
    print("=" * 80)
    print("This will compare both models against real Elon Musk responses")
    print("\nRequired:")
    print("  • Fast Model running on http://localhost:5001")
    print("  • Thinking Model running on http://localhost:5055")
    print("  • golden.json file in current directory")
    
    if not SENTENCE_TRANSFORMER_AVAILABLE:
        print("\n💡 Tip: Install sentence-transformers for semantic similarity:")
        print("   pip install sentence-transformers scikit-learn")
    
    print("\n" + "=" * 80)
    input("\nPress Enter to start evaluation...")
    
    run_evaluation()
