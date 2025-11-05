"""
Baseline Model Evaluation Script
Tests the baseline model (no RAG, no analyzer) against golden responses
"""

import requests
import json
import time
import re
import csv
from datetime import datetime
from typing import List, Dict, Tuple
import statistics

# Try to import advanced NLP libraries, with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_AVAILABLE = True
    print("✅ Loading semantic similarity model (all-MiniLM-L6-v2)...")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("   Model loaded successfully!")
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Using basic similarity metrics only.")
    print("   Install with: pip install sentence-transformers scikit-learn")

from difflib import SequenceMatcher

# Configuration
BASELINE_SERVER = "http://localhost:5002/predict"
GOLDEN_FILE = "golden.json"

# ============================================================================
# ELON-SPECIFIC LINGUISTIC MARKERS
# ============================================================================

ELON_CONVERSATIONAL_MARKERS = {
    'yeah', 'so', 'i mean', 'like', 'you know', 'i think', 'obviously',
    'clearly', 'definitely', 'probably', 'basically', 'essentially',
    'sort of', 'kind of', 'right', 'ok', 'well', 'um', 'uh'
}

ELON_CONFIDENCE_MARKERS = {
    'definitely', 'obviously', 'clearly', 'certainly', 'absolutely',
    'without a doubt', 'for sure', 'no question', 'unequivocally'
}

ELON_TECHNICAL_TERMS = {
    'rocket', 'spacecraft', 'orbital', 'propellant', 'methane', 'raptor',
    'starship', 'neural', 'autopilot', 'battery', 'electrode', 'tesla',
    'spacex', 'neuralink', 'mars', 'sustainable', 'energy', 'ai', 'fsd',
    'lithium', 'ion', 'semiconductor', 'manufacturing', 'production',
    'falcon', 'dragon', 'starlink', 'boring', 'hyperloop', 'gigafactory'
}

# Domain-specific stopwords
DOMAIN_STOPWORDS = {
    'model', 'data', 'training', 'learning', 'algorithm', 'neural network',
    'processing', 'optimization', 'performance', 'accuracy'
}

# ============================================================================
# ENHANCED SIMILARITY METRICS
# ============================================================================

def calculate_vocabulary_richness(text: str) -> float:
    """Calculate vocabulary diversity (unique words / total words)"""
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0
    return len(set(words)) / len(words)


def count_conversational_markers(text: str) -> float:
    """Count Elon-specific conversational markers"""
    text_lower = text.lower()
    words = text_lower.split()
    if not words:
        return 0
    marker_count = sum(1 for marker in ELON_CONVERSATIONAL_MARKERS if marker in text_lower)
    return (marker_count / len(words)) * 100


def count_confidence_markers(text: str) -> float:
    """Count confidence markers"""
    text_lower = text.lower()
    words = text_lower.split()
    if not words:
        return 0
    confidence_count = sum(1 for marker in ELON_CONFIDENCE_MARKERS if marker in text_lower)
    return (confidence_count / len(words)) * 100


def count_technical_terms(text: str) -> float:
    """Count Elon-specific technical terms"""
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0
    tech_count = sum(1 for word in words if word in ELON_TECHNICAL_TERMS)
    return (tech_count / len(words)) * 100


def calculate_style_similarity(text1: str, text2: str) -> float:
    """Calculate style similarity based on multiple features"""
    def get_style_features(text):
        words = text.lower().split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        num_sentences = len(sentences)
        avg_sentence_length = len(words) / max(num_sentences, 1)
        
        return {
            'avg_word_length': avg_word_length,
            'num_sentences': num_sentences,
            'avg_sentence_length': avg_sentence_length,
            'vocab_richness': calculate_vocabulary_richness(text),
            'conversational_markers': count_conversational_markers(text),
            'confidence_markers': count_confidence_markers(text),
            'technical_terms': count_technical_terms(text)
        }
    
    style1 = get_style_features(text1)
    style2 = get_style_features(text2)
    
    # Calculate similarity for each feature
    word_length_sim = 1 - min(abs(style1['avg_word_length'] - style2['avg_word_length']) / 10, 1)
    sentence_count_sim = 1 - min(abs(style1['num_sentences'] - style2['num_sentences']) / max(style1['num_sentences'], style2['num_sentences'], 1), 1)
    sentence_length_sim = 1 - min(abs(style1['avg_sentence_length'] - style2['avg_sentence_length']) / max(style1['avg_sentence_length'], style2['avg_sentence_length'], 1), 1)
    vocab_sim = 1 - min(abs(style1['vocab_richness'] - style2['vocab_richness']), 1)
    conv_sim = 1 - min(abs(style1['conversational_markers'] - style2['conversational_markers']) / 10, 1)
    conf_sim = 1 - min(abs(style1['confidence_markers'] - style2['confidence_markers']) / 5, 1)
    tech_sim = 1 - min(abs(style1['technical_terms'] - style2['technical_terms']) / 10, 1)
    
    # Weighted average
    style_similarity = (
        word_length_sim * 0.15 +
        sentence_count_sim * 0.10 +
        sentence_length_sim * 0.15 +
        vocab_sim * 0.15 +
        conv_sim * 0.25 +
        conf_sim * 0.10 +
        tech_sim * 0.10
    )
    
    return max(0, min(1, style_similarity))


def jaccard_similarity(text1: str, text2: str, remove_stopwords: bool = True) -> float:
    """Calculate Jaccard similarity with optional stopword removal"""
    def tokenize(text):
        words = set(re.findall(r'\b\w+\b', text.lower()))
        if remove_stopwords:
            words = {w for w in words if w not in DOMAIN_STOPWORDS}
        return words
    
    words1 = tokenize(text1)
    words2 = tokenize(text2)
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0
    
    jaccard = len(intersection) / len(union)
    
    # Length normalization
    len1 = len(text1.split())
    len2 = len(text2.split())
    length_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
    
    normalized_jaccard = jaccard * 0.8 + length_ratio * 0.2
    return normalized_jaccard


def sequence_match_similarity(text1: str, text2: str) -> float:
    """Calculate sequence matching similarity"""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def semantic_similarity(text1: str, text2: str, apply_paraphrase_penalty: bool = True) -> float:
    """Calculate semantic similarity with optional paraphrase penalty"""
    if not SEMANTIC_AVAILABLE:
        return None
    
    embeddings = semantic_model.encode([text1, text2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    similarity = float(similarity)
    
    if apply_paraphrase_penalty:
        jaccard = jaccard_similarity(text1, text2, remove_stopwords=False)
        if similarity > 0.8 and jaccard < 0.3:
            penalty = (0.8 - jaccard) * 0.15
            similarity = max(0, similarity - penalty)
    
    return similarity


def calculate_overall_similarity(golden: str, response: str) -> Dict:
    """Calculate comprehensive similarity metrics"""
    jaccard_sim = jaccard_similarity(golden, response)
    sequence_sim = sequence_match_similarity(golden, response)
    style_sim = calculate_style_similarity(golden, response)
    semantic_sim = semantic_similarity(golden, response) if SEMANTIC_AVAILABLE else None
    
    # Calculate overall similarity
    if SEMANTIC_AVAILABLE:
        overall_similarity = (
            semantic_sim * 0.35 +
            jaccard_sim * 0.25 +
            sequence_sim * 0.25 +
            style_sim * 0.15
        )
    else:
        overall_similarity = (
            jaccard_sim * 0.35 +
            sequence_sim * 0.35 +
            style_sim * 0.30
        )
    
    return {
        'overall': overall_similarity,
        'jaccard': jaccard_sim,
        'sequence': sequence_sim,
        'style': style_sim,
        'semantic': semantic_sim
    }


# ============================================================================
# MODEL INTERACTION
# ============================================================================

def load_golden_responses() -> List[Dict]:
    """Load golden responses from JSON file"""
    with open(GOLDEN_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['qa_pairs']


def get_model_response(query: str, user_id: str = "evaluator") -> Tuple[str, int]:
    """
    Get response from baseline model server.
    Returns (response_text, response_time_ms)
    """
    try:
        start = time.time()
        response = requests.post(
            BASELINE_SERVER,
            json={"input": query, "user_id": user_id},
            timeout=120
        )
        response_time = int((time.time() - start) * 1000)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('output', ''), response_time
        else:
            return f"Error: {response.status_code}", response_time
    except Exception as e:
        return f"Error: {str(e)}", 0


# ============================================================================
# EVALUATION LOGIC
# ============================================================================

def evaluate_baseline(golden_pairs: List[Dict]) -> Dict:
    """
    Evaluate baseline model against golden responses.
    Returns comprehensive results.
    """
    results = []
    scores = {'overall': [], 'jaccard': [], 'sequence': [], 'style': [], 'semantic': [], 'time': []}
    
    total = len(golden_pairs)
    
    print("\n" + "="*80)
    print("BASELINE MODEL EVALUATION")
    print("="*80)
    print(f"Total Questions: {total}")
    print(f"Model: final_combined_v11 (No RAG, No Analyzer)")
    print(f"Semantic Similarity: {'✅ Enabled' if SEMANTIC_AVAILABLE else '❌ Disabled'}")
    print("="*80 + "\n")
    
    for idx, pair in enumerate(golden_pairs, 1):
        query = pair['query']
        golden = pair.get('golden_response') or pair.get('response')
        
        print(f"\n{'='*80}")
        print(f"Question {idx}/{total}")
        print(f"{'='*80}")
        print(f"❓ {query}\n")
        print(f"🎯 Golden Response:")
        print(f"   {golden[:200]}{'...' if len(golden) > 200 else ''}\n")
        
        # Test Baseline Model
        print("📝 Testing Baseline Model...")
        response, response_time = get_model_response(query)
        metrics = calculate_overall_similarity(golden, response)
        
        print(f"📊 Baseline Model Response:")
        print(f"   {response[:200]}{'...' if len(response) > 200 else ''}")
        print(f"   ├─ Overall Similarity: {metrics['overall']:.3f}")
        print(f"   ├─ Jaccard Similarity: {metrics['jaccard']:.3f}")
        print(f"   ├─ Sequence Match: {metrics['sequence']:.3f}")
        if metrics['semantic'] is not None:
            print(f"   ├─ Semantic Similarity: {metrics['semantic']:.3f}")
        print(f"   ├─ Style Similarity: {metrics['style']:.3f}")
        print(f"   └─ Response Time: {response_time}ms")
        
        # Store results
        results.append({
            'question_num': idx,
            'query': query,
            'golden_response': golden,
            'baseline_response': response,
            'metrics': metrics,
            'response_time': response_time
        })
        
        # Accumulate scores
        for key in ['overall', 'jaccard', 'sequence', 'style']:
            scores[key].append(metrics[key])
        
        if metrics['semantic'] is not None:
            scores['semantic'].append(metrics['semantic'])
        
        scores['time'].append(response_time)
    
    # Calculate statistics
    stats = {
        'avg_overall': statistics.mean(scores['overall']),
        'avg_jaccard': statistics.mean(scores['jaccard']),
        'avg_sequence': statistics.mean(scores['sequence']),
        'avg_style': statistics.mean(scores['style']),
        'avg_semantic': statistics.mean(scores['semantic']) if scores['semantic'] else None,
        'avg_time': statistics.mean(scores['time']),
        'std_overall': statistics.stdev(scores['overall']) if len(scores['overall']) > 1 else 0,
        'min_overall': min(scores['overall']),
        'max_overall': max(scores['overall']),
        'success_count': len([s for s in scores['overall'] if s > 0]),
        'total_count': len(scores['overall'])
    }
    
    return {
        'timestamp': datetime.now().isoformat(),
        'total_questions': total,
        'semantic_enabled': SEMANTIC_AVAILABLE,
        'statistics': stats,
        'detailed_results': results
    }


def print_final_summary(evaluation_results: Dict):
    """Print comprehensive final summary"""
    stats = evaluation_results['statistics']
    
    print("\n" + "="*80)
    print("BASELINE MODEL - FINAL RESULTS")
    print("="*80 + "\n")
    
    print("📊 OVERALL PERFORMANCE:")
    print(f"   ├─ Successful Responses: {stats['success_count']}/{stats['total_count']}")
    print(f"   ├─ Avg Overall Similarity: {stats['avg_overall']:.3f} (±{stats['std_overall']:.3f})")
    print(f"   ├─ Min Similarity: {stats['min_overall']:.3f}")
    print(f"   ├─ Max Similarity: {stats['max_overall']:.3f}")
    print(f"   └─ Avg Response Time: {stats['avg_time']:.0f}ms\n")
    
    print("📈 METRIC BREAKDOWN:")
    print(f"   ├─ Avg Jaccard (Word Match): {stats['avg_jaccard']:.3f}")
    print(f"   ├─ Avg Sequence (Word Order): {stats['avg_sequence']:.3f}")
    if stats['avg_semantic'] is not None:
        print(f"   ├─ Avg Semantic (Meaning): {stats['avg_semantic']:.3f}")
    print(f"   └─ Avg Style (Personality): {stats['avg_style']:.3f}\n")
    
    print("="*80)
    print("✅ BASELINE MODEL EVALUATION COMPLETE")
    print("="*80 + "\n")


def save_results(evaluation_results: Dict):
    """Save results to JSON and CSV files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_file = f"baseline_evaluation_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    print(f"📄 Detailed results saved to: {json_file}")
    
    # Save CSV
    csv_file = f"baseline_evaluation_{timestamp}.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header row
        writer.writerow([
            'Question', 'Query', 'Golden_Response', 'Baseline_Response',
            'Overall_Similarity', 'Jaccard', 'Sequence', 'Style', 'Semantic', 'Response_Time_ms'
        ])
        
        # Data rows
        for result in evaluation_results['detailed_results']:
            writer.writerow([
                result['question_num'],
                result['query'],
                result['golden_response'][:100] + '...' if len(result['golden_response']) > 100 else result['golden_response'],
                result['baseline_response'][:100] + '...' if len(result['baseline_response']) > 100 else result['baseline_response'],
                result['metrics']['overall'],
                result['metrics']['jaccard'],
                result['metrics']['sequence'],
                result['metrics']['style'],
                result['metrics']['semantic'] or '',
                result['response_time']
            ])
        
        # Add summary section
        writer.writerow([])
        writer.writerow(['SUMMARY STATISTICS'])
        writer.writerow([])
        
        stats = evaluation_results['statistics']
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Avg Overall Similarity', stats['avg_overall']])
        writer.writerow(['Std Deviation', stats['std_overall']])
        writer.writerow(['Min Similarity', stats['min_overall']])
        writer.writerow(['Max Similarity', stats['max_overall']])
        writer.writerow(['Avg Jaccard', stats['avg_jaccard']])
        writer.writerow(['Avg Sequence', stats['avg_sequence']])
        writer.writerow(['Avg Style', stats['avg_style']])
        if stats['avg_semantic'] is not None:
            writer.writerow(['Avg Semantic', stats['avg_semantic']])
        writer.writerow(['Avg Response Time (ms)', stats['avg_time']])
        writer.writerow(['Success Rate', f"{stats['success_count']}/{stats['total_count']}"])
    
    print(f"📊 CSV export saved to: {csv_file}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "📝"*40)
    print("BASELINE MODEL EVALUATION")
    print("Testing final_combined_v11 without RAG or Query Analyzer")
    print("📝"*40 + "\n")
    
    try:
        # Check server health
        print("🔍 Checking baseline server...")
        try:
            health_response = requests.get("http://localhost:5002/health", timeout=5)
            if health_response.status_code == 200:
                print("✅ Baseline server is healthy\n")
            else:
                print("⚠️  Server responded but returned non-200 status")
        except:
            print("❌ Cannot connect to baseline server!")
            print("   Make sure it's running: python baseline_model_server.py")
            return
        
        # Load golden responses
        print("📚 Loading golden responses...")
        golden_pairs = load_golden_responses()
        print(f"   Loaded {len(golden_pairs)} question-answer pairs\n")
        
        # Run evaluation
        evaluation_results = evaluate_baseline(golden_pairs)
        
        # Print summary
        print_final_summary(evaluation_results)
        
        # Save results
        save_results(evaluation_results)
        
        print("✅ Evaluation complete!\n")
        
    except FileNotFoundError:
        print(f"❌ Error: {GOLDEN_FILE} not found!")
        print("   Please create a golden.json file with question-answer pairs")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
