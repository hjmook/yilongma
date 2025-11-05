"""
Enhanced Golden Set Evaluation Script
Compares Fast vs Thinking model responses against real Elon Musk interview answers
with improved similarity metrics based on personality authenticity research
"""

import requests
import json
import time
import re
import csv
from datetime import datetime
from typing import List, Dict, Tuple
from collections import Counter
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
FAST_SERVER = "http://localhost:5001/predict"
THINKING_SERVER = "http://localhost:5055/predict"
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

# Domain-specific stopwords (common ML/AI jargon to filter out)
DOMAIN_STOPWORDS = {
    'model', 'data', 'training', 'learning', 'algorithm', 'neural network',
    'processing', 'optimization', 'performance', 'accuracy'
}

# ============================================================================
# ENHANCED SIMILARITY METRICS
# ============================================================================

def calculate_vocabulary_richness(text: str) -> float:
    """
    Calculate vocabulary diversity (unique words / total words).
    Elon mixes simple words with technical jargon.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0
    return len(set(words)) / len(words)


def count_conversational_markers(text: str) -> float:
    """
    Count Elon-specific conversational markers like 'yeah', 'so', 'i mean'.
    Returns normalized count (markers per 100 words).
    """
    text_lower = text.lower()
    words = text_lower.split()
    if not words:
        return 0
    
    marker_count = sum(1 for marker in ELON_CONVERSATIONAL_MARKERS if marker in text_lower)
    return (marker_count / len(words)) * 100


def count_confidence_markers(text: str) -> float:
    """
    Count confidence markers like 'definitely', 'obviously', 'clearly'.
    Returns normalized count.
    """
    text_lower = text.lower()
    words = text_lower.split()
    if not words:
        return 0
    
    confidence_count = sum(1 for marker in ELON_CONFIDENCE_MARKERS if marker in text_lower)
    return (confidence_count / len(words)) * 100


def count_technical_terms(text: str) -> float:
    """
    Count Elon-specific technical terms.
    Returns normalized count.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0
    
    tech_count = sum(1 for word in words if word in ELON_TECHNICAL_TERMS)
    return (tech_count / len(words)) * 100


def calculate_style_similarity(text1: str, text2: str) -> float:
    """
    Enhanced style similarity based on:
    - Average word length (Elon uses simple words)
    - Sentence structure (short, punchy sentences)
    - Vocabulary richness (mixing simple + technical)
    - Conversational markers ('yeah', 'so', 'i mean')
    - Confidence markers ('definitely', 'obviously')
    - Technical term usage
    """
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
    
    # Calculate similarity for each feature (inverse of normalized difference)
    word_length_sim = 1 - min(abs(style1['avg_word_length'] - style2['avg_word_length']) / 10, 1)
    sentence_count_sim = 1 - min(abs(style1['num_sentences'] - style2['num_sentences']) / max(style1['num_sentences'], style2['num_sentences'], 1), 1)
    sentence_length_sim = 1 - min(abs(style1['avg_sentence_length'] - style2['avg_sentence_length']) / max(style1['avg_sentence_length'], style2['avg_sentence_length'], 1), 1)
    
    # Vocabulary richness similarity
    vocab_sim = 1 - min(abs(style1['vocab_richness'] - style2['vocab_richness']), 1)
    
    # Conversational markers similarity (normalized difference)
    conv_sim = 1 - min(abs(style1['conversational_markers'] - style2['conversational_markers']) / 10, 1)
    
    # Confidence markers similarity
    conf_sim = 1 - min(abs(style1['confidence_markers'] - style2['confidence_markers']) / 5, 1)
    
    # Technical terms similarity
    tech_sim = 1 - min(abs(style1['technical_terms'] - style2['technical_terms']) / 10, 1)
    
    # Weighted average of style features
    # Prioritize conversational markers and sentence structure for Elon's style
    style_similarity = (
        word_length_sim * 0.15 +
        sentence_count_sim * 0.10 +
        sentence_length_sim * 0.15 +
        vocab_sim * 0.15 +
        conv_sim * 0.25 +  # High weight - very characteristic of Elon
        conf_sim * 0.10 +
        tech_sim * 0.10
    )
    
    return max(0, min(1, style_similarity))


def jaccard_similarity(text1: str, text2: str, remove_stopwords: bool = True) -> float:
    """
    Calculate Jaccard similarity between two texts (word-level).
    Optionally removes domain-specific stopwords to focus on content words.
    Normalized by response length to handle length variance.
    """
    def tokenize(text):
        # Extract words, convert to lowercase
        words = set(re.findall(r'\b\w+\b', text.lower()))
        
        if remove_stopwords:
            # Remove domain-specific stopwords
            words = {w for w in words if w not in DOMAIN_STOPWORDS}
        
        return words
    
    words1 = tokenize(text1)
    words2 = tokenize(text2)
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0
    
    # Basic Jaccard
    jaccard = len(intersection) / len(union)
    
    # Length normalization factor
    # Penalize if response lengths are very different
    len1 = len(text1.split())
    len2 = len(text2.split())
    length_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
    
    # Combine Jaccard with length normalization (80% Jaccard, 20% length similarity)
    normalized_jaccard = jaccard * 0.8 + length_ratio * 0.2
    
    return normalized_jaccard


def sequence_match_similarity(text1: str, text2: str) -> float:
    """
    Calculate sequence matching similarity (preserves word order).
    Important for capturing Elon's specific phrasing patterns.
    """
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def semantic_similarity(text1: str, text2: str, apply_paraphrase_penalty: bool = True) -> float:
    """
    Calculate semantic similarity using sentence transformers.
    Optionally applies paraphrase penalty to reduce bias toward memorization.
    """
    if not SEMANTIC_AVAILABLE:
        return None
    
    embeddings = semantic_model.encode([text1, text2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    similarity = float(similarity)
    
    if apply_paraphrase_penalty:
        # If semantically very similar but words are different, reduce score slightly
        # This penalizes "polished rephrasing" vs authentic word choice
        jaccard = jaccard_similarity(text1, text2, remove_stopwords=False)
        
        # If semantic similarity is high (>0.8) but Jaccard is low (<0.3),
        # it's likely a paraphrase rather than authentic style match
        if similarity > 0.8 and jaccard < 0.3:
            penalty = (0.8 - jaccard) * 0.15  # Up to 7.5% penalty
            similarity = max(0, similarity - penalty)
    
    return similarity


def calculate_overall_similarity(golden: str, response: str) -> Dict:
    """
    Calculate comprehensive similarity metrics.
    Returns dict with individual metrics and overall score.
    """
    jaccard_sim = jaccard_similarity(golden, response)
    sequence_sim = sequence_match_similarity(golden, response)
    style_sim = calculate_style_similarity(golden, response)
    semantic_sim = semantic_similarity(golden, response) if SEMANTIC_AVAILABLE else None
    
    # Calculate overall similarity score
    # REVISED FORMULA based on personality authenticity evaluation best practices
    if SEMANTIC_AVAILABLE:
        # With semantic model: 35% semantic + 25% Jaccard + 25% sequence + 15% style
        # Reduced semantic weight to minimize memorization bias
        # Increased Jaccard & sequence to prioritize authentic word choice and phrasing
        overall_similarity = (
            semantic_sim * 0.35 +
            jaccard_sim * 0.25 +
            sequence_sim * 0.25 +
            style_sim * 0.15
        )
    else:
        # Without semantic model: 35% Jaccard + 35% sequence + 30% style
        # Elevated sequence match priority for personality assessment
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


def get_model_response(server_url: str, query: str, user_id: str = "evaluator") -> Tuple[str, int]:
    """
    Get response from a model server.
    Returns (response_text, response_time_ms)
    """
    try:
        start = time.time()
        response = requests.post(
            server_url,
            json={"input": query, "user_id": user_id, "use_rag": True},  # Fixed: 'input' not 'query'
            timeout=120  # Increased to 120 seconds for model loading + RAG processing
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

def evaluate_models(golden_pairs: List[Dict]) -> Dict:
    """
    Evaluate both models against golden responses.
    Returns comprehensive results.
    """
    results = []
    fast_scores = {'overall': [], 'jaccard': [], 'sequence': [], 'style': [], 'semantic': [], 'time': []}
    thinking_scores = {'overall': [], 'jaccard': [], 'sequence': [], 'style': [], 'semantic': [], 'time': []}
    
    total = len(golden_pairs)
    
    print("\n" + "="*80)
    print("STARTING GOLDEN SET EVALUATION")
    print("="*80)
    print(f"Total Questions: {total}")
    print(f"Semantic Similarity: {'✅ Enabled' if SEMANTIC_AVAILABLE else '❌ Disabled (basic metrics only)'}")
    print("="*80 + "\n")
    
    for idx, pair in enumerate(golden_pairs, 1):
        query = pair['query']
        golden = pair.get('golden_response') or pair.get('response')  # Support both key names
        
        print(f"\n{'='*80}")
        print(f"Question {idx}/{total}")
        print(f"{'='*80}")
        print(f"❓ {query}\n")
        print(f"🎯 Golden Response:")
        print(f"   {golden[:200]}{'...' if len(golden) > 200 else ''}\n")
        
        # Test Fast Model
        print("⚡ Testing Fast Model...")
        fast_response, fast_time = get_model_response(FAST_SERVER, query)
        fast_metrics = calculate_overall_similarity(golden, fast_response)
        
        print(f"📊 Fast Model Response:")
        print(f"   {fast_response[:200]}{'...' if len(fast_response) > 200 else ''}")
        print(f"   ├─ Overall Similarity: {fast_metrics['overall']:.3f}")
        print(f"   ├─ Jaccard Similarity: {fast_metrics['jaccard']:.3f}")
        print(f"   ├─ Sequence Match: {fast_metrics['sequence']:.3f}")
        if fast_metrics['semantic'] is not None:
            print(f"   ├─ Semantic Similarity: {fast_metrics['semantic']:.3f}")
        print(f"   ├─ Style Similarity: {fast_metrics['style']:.3f}")
        print(f"   └─ Response Time: {fast_time}ms\n")
        
        # Test Thinking Model
        print("🧠 Testing Thinking Model...")
        thinking_response, thinking_time = get_model_response(THINKING_SERVER, query)
        thinking_metrics = calculate_overall_similarity(golden, thinking_response)
        
        print(f"📊 Thinking Model Response:")
        print(f"   {thinking_response[:200]}{'...' if len(thinking_response) > 200 else ''}")
        print(f"   ├─ Overall Similarity: {thinking_metrics['overall']:.3f}")
        print(f"   ├─ Jaccard Similarity: {thinking_metrics['jaccard']:.3f}")
        print(f"   ├─ Sequence Match: {thinking_metrics['sequence']:.3f}")
        if thinking_metrics['semantic'] is not None:
            print(f"   ├─ Semantic Similarity: {thinking_metrics['semantic']:.3f}")
        print(f"   ├─ Style Similarity: {thinking_metrics['style']:.3f}")
        print(f"   └─ Response Time: {thinking_time}ms\n")
        
        # Determine winner
        winner = "Fast" if fast_metrics['overall'] > thinking_metrics['overall'] else "Thinking"
        winner_emoji = "⚡" if winner == "Fast" else "🧠"
        diff = abs(fast_metrics['overall'] - thinking_metrics['overall'])
        
        print(f"🏆 Winner: {winner_emoji} {winner} Model (score: {max(fast_metrics['overall'], thinking_metrics['overall']):.3f} vs {min(fast_metrics['overall'], thinking_metrics['overall']):.3f}, diff: {diff:.3f})")
        
        # Store results
        results.append({
            'question_num': idx,
            'query': query,
            'golden_response': golden,
            'fast_response': fast_response,
            'fast_metrics': fast_metrics,
            'fast_time': fast_time,
            'thinking_response': thinking_response,
            'thinking_metrics': thinking_metrics,
            'thinking_time': thinking_time,
            'winner': winner
        })
        
        # Accumulate scores
        for key in ['overall', 'jaccard', 'sequence', 'style']:
            fast_scores[key].append(fast_metrics[key])
            thinking_scores[key].append(thinking_metrics[key])
        
        if fast_metrics['semantic'] is not None:
            fast_scores['semantic'].append(fast_metrics['semantic'])
            thinking_scores['semantic'].append(thinking_metrics['semantic'])
        
        fast_scores['time'].append(fast_time)
        thinking_scores['time'].append(thinking_time)
    
    # Calculate final statistics
    def calc_stats(scores):
        return {
            'avg_overall': statistics.mean(scores['overall']),
            'avg_jaccard': statistics.mean(scores['jaccard']),
            'avg_sequence': statistics.mean(scores['sequence']),
            'avg_style': statistics.mean(scores['style']),
            'avg_semantic': statistics.mean(scores['semantic']) if scores['semantic'] else None,
            'avg_time': statistics.mean(scores['time']),
            'std_overall': statistics.stdev(scores['overall']) if len(scores['overall']) > 1 else 0,
            'success_count': len([s for s in scores['overall'] if s > 0]),
            'total_count': len(scores['overall'])
        }
    
    fast_stats = calc_stats(fast_scores)
    thinking_stats = calc_stats(thinking_scores)
    
    # Determine overall winner
    overall_winner = "Thinking" if thinking_stats['avg_overall'] > fast_stats['avg_overall'] else "Fast"
    improvement = abs(thinking_stats['avg_overall'] - fast_stats['avg_overall']) / min(fast_stats['avg_overall'], thinking_stats['avg_overall']) * 100
    
    return {
        'timestamp': datetime.now().isoformat(),
        'total_questions': total,
        'semantic_enabled': SEMANTIC_AVAILABLE,
        'fast_model': fast_stats,
        'thinking_model': thinking_stats,
        'overall_winner': overall_winner,
        'improvement_pct': improvement,
        'detailed_results': results,
        'win_counts': {
            'fast': sum(1 for r in results if r['winner'] == 'Fast'),
            'thinking': sum(1 for r in results if r['winner'] == 'Thinking')
        }
    }


def print_final_summary(evaluation_results: Dict):
    """Print comprehensive final summary"""
    fast = evaluation_results['fast_model']
    thinking = evaluation_results['thinking_model']
    winner = evaluation_results['overall_winner']
    improvement = evaluation_results['improvement_pct']
    win_counts = evaluation_results['win_counts']
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80 + "\n")
    
    print("⚡ FAST MODEL:")
    print(f"   ├─ Successful Responses: {fast['success_count']}/{fast['total_count']}")
    print(f"   ├─ Avg Overall Similarity: {fast['avg_overall']:.3f} (±{fast['std_overall']:.3f})")
    print(f"   ├─ Avg Jaccard: {fast['avg_jaccard']:.3f}")
    print(f"   ├─ Avg Sequence: {fast['avg_sequence']:.3f}")
    if fast['avg_semantic'] is not None:
        print(f"   ├─ Avg Semantic: {fast['avg_semantic']:.3f}")
    print(f"   ├─ Avg Style: {fast['avg_style']:.3f}")
    print(f"   ├─ Avg Response Time: {fast['avg_time']:.0f}ms")
    print(f"   └─ Wins: {win_counts['fast']}/{evaluation_results['total_questions']}\n")
    
    print("🧠 THINKING MODEL:")
    print(f"   ├─ Successful Responses: {thinking['success_count']}/{thinking['total_count']}")
    print(f"   ├─ Avg Overall Similarity: {thinking['avg_overall']:.3f} (±{thinking['std_overall']:.3f})")
    print(f"   ├─ Avg Jaccard: {thinking['avg_jaccard']:.3f}")
    print(f"   ├─ Avg Sequence: {thinking['avg_sequence']:.3f}")
    if thinking['avg_semantic'] is not None:
        print(f"   ├─ Avg Semantic: {thinking['avg_semantic']:.3f}")
    print(f"   ├─ Avg Style: {thinking['avg_style']:.3f}")
    print(f"   ├─ Avg Response Time: {thinking['avg_time']:.0f}ms")
    print(f"   └─ Wins: {win_counts['thinking']}/{evaluation_results['total_questions']}\n")
    
    print("="*80)
    winner_emoji = "🧠" if winner == "Thinking" else "⚡"
    print(f"🏆 OVERALL WINNER: {winner_emoji} {winner.upper()} MODEL")
    print(f"   {improvement:.1f}% {'more' if winner == 'Thinking' else 'less'} similar to real Elon responses")
    print(f"   Won {max(win_counts['fast'], win_counts['thinking'])}/{evaluation_results['total_questions']} questions")
    print("="*80 + "\n")


def save_results(evaluation_results: Dict):
    """Save results to JSON and CSV files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON (detailed)
    json_file = f"golden_evaluation_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    print(f"📄 Detailed results saved to: {json_file}")
    
    # Save CSV (for spreadsheet analysis)
    csv_file = f"golden_evaluation_{timestamp}.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header row
        writer.writerow([
            'Question', 'Query', 'Winner',
            'Fast_Overall', 'Fast_Jaccard', 'Fast_Sequence', 'Fast_Style', 'Fast_Semantic', 'Fast_Time',
            'Thinking_Overall', 'Thinking_Jaccard', 'Thinking_Sequence', 'Thinking_Style', 'Thinking_Semantic', 'Thinking_Time'
        ])
        
        # Data rows
        for result in evaluation_results['detailed_results']:
            writer.writerow([
                result['question_num'],
                result['query'],
                result['winner'],
                result['fast_metrics']['overall'],
                result['fast_metrics']['jaccard'],
                result['fast_metrics']['sequence'],
                result['fast_metrics']['style'],
                result['fast_metrics']['semantic'] or '',
                result['fast_time'],
                result['thinking_metrics']['overall'],
                result['thinking_metrics']['jaccard'],
                result['thinking_metrics']['sequence'],
                result['thinking_metrics']['style'],
                result['thinking_metrics']['semantic'] or '',
                result['thinking_time']
            ])
        
        # Add empty row separator
        writer.writerow([])
        
        # Add summary statistics rows
        fast = evaluation_results['fast_model']
        thinking = evaluation_results['thinking_model']
        
        writer.writerow(['SUMMARY STATISTICS'])
        writer.writerow([])
        
        # Fast Model Averages
        writer.writerow([
            'FAST MODEL AVERAGES',
            '',
            '',
            fast['avg_overall'],
            fast['avg_jaccard'],
            fast['avg_sequence'],
            fast['avg_style'],
            fast['avg_semantic'] if fast['avg_semantic'] is not None else '',
            fast['avg_time']
        ])
        
        # Thinking Model Averages
        writer.writerow([
            'THINKING MODEL AVERAGES',
            '',
            '',
            thinking['avg_overall'],
            thinking['avg_jaccard'],
            thinking['avg_sequence'],
            thinking['avg_style'],
            thinking['avg_semantic'] if thinking['avg_semantic'] is not None else '',
            thinking['avg_time']
        ])
        
        writer.writerow([])
        
        # Standard Deviations
        writer.writerow([
            'STANDARD DEVIATION',
            '',
            'Fast',
            fast['std_overall']
        ])
        writer.writerow([
            '',
            '',
            'Thinking',
            thinking['std_overall']
        ])
        
        writer.writerow([])
        
        # Win counts
        writer.writerow([
            'WIN COUNTS',
            '',
            'Fast',
            evaluation_results['win_counts']['fast']
        ])
        writer.writerow([
            '',
            '',
            'Thinking',
            evaluation_results['win_counts']['thinking']
        ])
        
        writer.writerow([])
        
        # Overall winner
        writer.writerow([
            'OVERALL WINNER',
            '',
            evaluation_results['overall_winner'].upper(),
            f"{evaluation_results['improvement_pct']:.1f}% improvement"
        ])
    
    print(f"📊 CSV export saved to: {csv_file}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "🎯"*40)
    print("ELON MUSK CHATBOT - GOLDEN SET EVALUATION")
    print("Enhanced Similarity Metrics for Personality Authenticity")
    print("🎯"*40 + "\n")
    
    try:
        # Load golden responses
        print("📚 Loading golden responses...")
        golden_pairs = load_golden_responses()
        print(f"   Loaded {len(golden_pairs)} question-answer pairs\n")
        
        # Run evaluation
        evaluation_results = evaluate_models(golden_pairs)
        
        # Print summary
        print_final_summary(evaluation_results)
        
        # Save results
        save_results(evaluation_results)
        
        print("✅ Evaluation complete!\n")
        
    except FileNotFoundError:
        print(f"❌ Error: {GOLDEN_FILE} not found!")
        print("   Please create a golden.json file with this format:")
        print("""
{
  "qa_pairs": [
    {
      "query": "Why did you start SpaceX?",
      "golden_response": "[actual Elon response from interview]"
    }
  ]
}
""")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
