"""
Model Comparison Script
Evaluates Fast vs Thinking model on test queries
"""

import requests
import json
import time
from datetime import datetime
from typing import  Dict

# Test queries covering different types
TEST_QUERIES = [
    # Recent factual
    "What are your plans for this year?",
    "How many Starlink satellites are in orbit currently?",
    "What happened with Twitter's rebranding to X?",
    
    # Domain knowledge
    "Why did you start SpaceX?",
    "What's your approach to solving traffic with Boring Company?",
    "Explain your vision for sustainable energy",
    
    # Conversational
    "How are you doing today?",
    "What do you think about AI safety?",
    "Are you worried about competition?",
    
    # Complex multi-part
    "What's your strategy for Tesla's expansion in China and how does it relate to the supply chain?",
    "How do you balance running multiple companies while also working on Neuralink?",
]

FAST_SERVER = "http://localhost:5001/predict"
THINKING_SERVER = "http://localhost:5055/predict"


def evaluate_model(server_url: str, query: str, user_id: str = "evaluator") -> Dict:
    """Send query to model and measure response"""
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
                "num_chunks": data.get("num_chunks", 0),
                "query_type": data.get("query_type", "unknown"),
                "total_latency_ms": latency,
                "server_latency": data.get("latency", {}),
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "total_latency_ms": latency
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "total_latency_ms": 0
        }


def compare_models():
    """Run comparison between Fast and Thinking models"""
    
    print("=" * 80)
    print("MODEL COMPARISON: Fast vs Thinking")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = {
        "test_date": datetime.now().isoformat(),
        "queries": [],
        "summary": {
            "fast": {"total_time": 0, "rag_usage": 0, "avg_response_length": 0},
            "thinking": {"total_time": 0, "rag_usage": 0, "avg_response_length": 0}
        }
    }
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}/{len(TEST_QUERIES)}")
        print(f"{'='*80}")
        print(f"📝 {query}\n")
        
        # Evaluate both models
        print("⚡ Testing Fast Model...")
        fast_result = evaluate_model(FAST_SERVER, query, f"eval_fast_{i}")
        
        print("🧠 Testing Thinking Model...")
        thinking_result = evaluate_model(THINKING_SERVER, query, f"eval_thinking_{i}")
        
        # Display results
        print("\n" + "-" * 80)
        print("FAST MODEL:")
        print("-" * 80)
        if fast_result["success"]:
            print(f"Response: {fast_result['output'][:200]}...")
            print(f"RAG Used: {fast_result['rag_used']} ({fast_result['num_chunks']} chunks)")
            print(f"Latency: {fast_result['total_latency_ms']:.0f}ms")
            results["summary"]["fast"]["total_time"] += fast_result["total_latency_ms"]
            results["summary"]["fast"]["rag_usage"] += (1 if fast_result["rag_used"] else 0)
            results["summary"]["fast"]["avg_response_length"] += len(fast_result["output"])
        else:
            print(f"❌ Error: {fast_result['error']}")
        
        print("\n" + "-" * 80)
        print("THINKING MODEL:")
        print("-" * 80)
        if thinking_result["success"]:
            print(f"Response: {thinking_result['output'][:200]}...")
            print(f"RAG Used: {thinking_result['rag_used']} ({thinking_result['num_chunks']} chunks)")
            print(f"Latency: {thinking_result['total_latency_ms']:.0f}ms")
            if thinking_result.get("server_latency"):
                lat = thinking_result["server_latency"]
                print(f"  ├─ Analysis: {lat.get('analysis_ms', 0):.0f}ms")
                print(f"  ├─ Retrieval: {lat.get('retrieval_ms', 0):.0f}ms")
                print(f"  └─ Generation: {lat.get('generation_ms', 0):.0f}ms")
            results["summary"]["thinking"]["total_time"] += thinking_result["total_latency_ms"]
            results["summary"]["thinking"]["rag_usage"] += (1 if thinking_result["rag_used"] else 0)
            results["summary"]["thinking"]["avg_response_length"] += len(thinking_result["output"])
        else:
            print(f"❌ Error: {thinking_result['error']}")
        
        # Store detailed results
        results["queries"].append({
            "query": query,
            "fast": fast_result,
            "thinking": thinking_result
        })
        
        time.sleep(1)  # Brief pause between queries
    
    # Calculate averages
    n = len(TEST_QUERIES)
    results["summary"]["fast"]["avg_latency_ms"] = results["summary"]["fast"]["total_time"] / n
    results["summary"]["fast"]["rag_usage_percent"] = (results["summary"]["fast"]["rag_usage"] / n) * 100
    results["summary"]["fast"]["avg_response_length"] = results["summary"]["fast"]["avg_response_length"] / n
    
    results["summary"]["thinking"]["avg_latency_ms"] = results["summary"]["thinking"]["total_time"] / n
    results["summary"]["thinking"]["rag_usage_percent"] = (results["summary"]["thinking"]["rag_usage"] / n) * 100
    results["summary"]["thinking"]["avg_response_length"] = results["summary"]["thinking"]["avg_response_length"] / n
    
    # Print summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nFast Model:")
    print(f"  • Avg Latency: {results['summary']['fast']['avg_latency_ms']:.0f}ms")
    print(f"  • RAG Usage: {results['summary']['fast']['rag_usage_percent']:.0f}%")
    print(f"  • Avg Response Length: {results['summary']['fast']['avg_response_length']:.0f} chars")
    
    print(f"\nThinking Model:")
    print(f"  • Avg Latency: {results['summary']['thinking']['avg_latency_ms']:.0f}ms")
    print(f"  • RAG Usage: {results['summary']['thinking']['rag_usage_percent']:.0f}%")
    print(f"  • Avg Response Length: {results['summary']['thinking']['avg_response_length']:.0f} chars")
    
    # Save detailed results
    filename = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Detailed results saved to: {filename}")


if __name__ == "__main__":
    print("Starting model comparison...")
    print("Make sure both servers are running:")
    print("  • Fast Model: http://localhost:5001")
    print("  • Thinking Model: http://localhost:5055\n")
    
    input("Press Enter to start evaluation...")
    
    compare_models()