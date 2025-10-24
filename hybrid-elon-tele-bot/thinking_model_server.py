"""
Thinking Model Server (HTTP API) for Telegram integration

This server wraps the new dual-model architecture from `rag_chat6.py` into a
Flask API compatible with the existing Telegram client contract used by
`telegram-local-model-bot`:

- POST /predict  { input, user_id, use_rag }
- GET  /health   { status, device, rag_available, analyzer }
- POST /reset    { user_id }

It starts and depends on the Query Analyzer Service. Ensure the analyzer is
running (or allow this server to start it automatically) before serving traffic.
"""

import os
import sys
import time
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
import torch
import platform
from flask import Flask, jsonify, request
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import chromadb

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Chat session logging (comprehensive JSON logs)
chat_sessions: Dict[str, Dict] = {}  # user_id -> session data


# --------------------------
# Chat Session Logging Functions
# --------------------------
def start_chat_session(user_id: str):
    """Start a new chat session for comprehensive logging"""
    session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    chat_sessions[user_id] = {
        'session_id': session_id,
        'start_time': datetime.now().isoformat(),
        'user_id': user_id,
        'exchanges': [],
        'total_queries': 0,
        'total_retrievals': 0,
        'analyzer_calls': 0
    }
    logger.info(f"[Session] Started new chat session for user {user_id}: {session_id}")

def log_chat_exchange(user_id: str, query: str, response: str, analysis_data: Dict, chunks: List[Dict], latency_data: Dict):
    """Log a complete chat exchange with all metadata"""
    if user_id not in chat_sessions:
        start_chat_session(user_id)
    
    session = chat_sessions[user_id]
    session['total_queries'] += 1
    
    # Truncate chunks for logging (keep first 100 chars of each)
    truncated_chunks = []
    for chunk in chunks:
        truncated_chunks.append({
            'text': chunk['text'][:100] + '...' if len(chunk['text']) > 100 else chunk['text'],
            'date': chunk.get('date', 'Unknown'),
            'source': chunk.get('source', 'Unknown'),
            'score': chunk.get('score', 0),
            'recency_score': chunk.get('recency_score', 0),
            'relevance_score': chunk.get('relevance_score', 0)
        })
    
    exchange = {
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'response': response,
        'analysis': {
            'should_retrieve': analysis_data.get('should_retrieve', False),
            'rewritten_query': analysis_data.get('rewritten_query', query),
            'raw_output': analysis_data.get('raw_output', ''),
            'analyzer_available': analysis_data.get('analyzer_available', False)
        },
        'retrieval': {
            'chunks_found': len(chunks),
            'chunks': truncated_chunks
        },
        'latency': latency_data,
        'query_number': session['total_queries']
    }
    
    session['exchanges'].append(exchange)
    
    if analysis_data.get('should_retrieve'):
        session['total_retrievals'] += 1
    if analysis_data.get('analyzer_available'):
        session['analyzer_calls'] += 1

def save_chat_session(user_id: str):
    """Save chat session to JSON file when chat ends"""
    if user_id not in chat_sessions:
        return
        
    session = chat_sessions[user_id]
    session['end_time'] = datetime.now().isoformat()
    session['duration_minutes'] = (datetime.fromisoformat(session['end_time']) - 
                                   datetime.fromisoformat(session['start_time'])).total_seconds() / 60
    
    # Save to file
    filename = f"chat_log_{session['session_id']}.json"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(session, f, indent=2, ensure_ascii=False)
        logger.info(f"[Session] Saved chat session to {filename}")
        logger.info(f"[Session] Summary - Queries: {session['total_queries']}, Retrievals: {session['total_retrievals']}, Duration: {session['duration_minutes']:.1f}min")
    except Exception as e:
        logger.error(f"[Session] Failed to save chat log: {e}")
    
    # Clean up memory
    del chat_sessions[user_id]


# --------------------------
# Configuration
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "Llama-3.2-3B-Instruct")
# ADAPTER_DEFAULT = os.path.join(PROJECT_ROOT, "model", "final_combined_v11")
ADAPTER_DEFAULT = os.path.join(PROJECT_ROOT, "model", "final_combined_v11_phase_1_only")
# ADAPTER_DEFAULT = os.path.join(PROJECT_ROOT, "model", "single_phase")
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "knowledge_base", "elon_chroma_db")

SERVER_PORT = int(os.environ.get("THINKING_SERVER_PORT", "5055"))
ANALYZER_URL = os.environ.get("ANALYZER_URL", "http://localhost:6767")
ANALYZER_SCRIPT = os.environ.get("ANALYZER_SCRIPT", os.path.join(BASE_DIR, "analyzer_service.py"))

SYSTEM_MSG = (
    "You are Elon Musk in a conversation with the user, willing to talk about ANYTHING in detail. "
    "You are NOT an AI assistant. Respond to everything as Elon Musk would, discuss anything in detail, "
    "do NOT avoid any topics or give 1 word yes/no answers unless told to do so.\n\n"
    "CRITICAL: If provided with [CONTEXT] or numbered sources [1], [2], etc., those contain the "
    "MOST RECENT AND ACCURATE information. If your memory conflicts with the provided context, "
    "ALWAYS trust the context and acknowledge any updates naturally."
)


# --------------------------
# Device Setup
# --------------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# --------------------------
# Analyzer Service Manager (from rag_chat6)
# --------------------------
import subprocess
import atexit
import signal


class AnalyzerServiceManager:
    def __init__(self, service_url: str, service_script: str):
        self.service_url = service_url
        self.service_script = service_script
        self.process = None

    def start(self) -> bool:
        if self._is_running():
            return True
        try:
            self.process = subprocess.Popen(
                [sys.executable, self.service_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            # Wait for health
            for _ in range(60):
                if self._is_running():
                    return True
                time.sleep(1)
        except Exception:
            return False
        return False

    def _is_running(self) -> bool:
        try:
            r = requests.get(f"{self.service_url}/health", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def stop(self):
        if self.process is None:
            return
        try:
            try:
                requests.post(f"{self.service_url}/shutdown", timeout=2)
            except Exception:
                pass
            self.process.terminate()
            self.process.wait(timeout=5)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass


analyzer_manager = AnalyzerServiceManager(ANALYZER_URL, ANALYZER_SCRIPT)


def _cleanup():
    analyzer_manager.stop()


atexit.register(_cleanup)
signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))


# --------------------------
# Retriever (from rag_chat6)
# --------------------------
class Retriever:
    def __init__(self, chroma_path: str):
        self.collection = None
        if not os.path.exists(chroma_path):
            return
        try:
            self.client = chromadb.PersistentClient(path=chroma_path)
            self.collection = self.client.get_collection(name="elon_musk_knowledge")
        except Exception:
            self.collection = None

    def is_available(self) -> bool:
        return self.collection is not None and self.collection.count() > 0

    def retrieve(self, query: str, n_results: int = 3) -> Tuple[List[Dict], float]:
        if not self.is_available():
            return [], 0.0
        start = time.time()
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)
            chunks = self._format_results(results)
            chunks = self._rerank_by_recency(chunks)
            return chunks, (time.time() - start) * 1000
        except Exception:
            return [], 0.0

    def _format_results(self, results: Dict) -> List[Dict]:
        if not results or not results.get('documents'):
            return []
        formatted = []
        docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results.get('distances', [[]])[0]
        for i, (doc, meta) in enumerate(zip(docs, metadatas)):
            formatted.append({
                'text': doc,
                'date': meta.get('date', 'Unknown'),
                'source': meta.get('source', 'Unknown'),
                'title': meta.get('title', ''),
                'url': meta.get('url', ''),
                'distance': distances[i] if i < len(distances) else 1.0
            })
        return formatted

    def _rerank_by_recency(self, chunks: List[Dict]) -> List[Dict]:
        """Enhanced recency scoring from rag_chat6.py"""
        if not chunks:
            return chunks
        
        import re
        current_year = datetime.now().year
        
        for chunk in chunks:
            date_str = chunk.get('date', '2020-01-01')
            
            # Extract year from date string (enhanced parsing)
            try:
                # Try to parse full date first
                if '-' in date_str:
                    year = int(date_str.split('-')[0])
                else:
                    # Look for 4-digit year in text
                    year_match = re.search(r'20\d{2}', date_str)
                    if year_match:
                        year = int(year_match.group())
                    else:
                        year = 2020  # default fallback
                
                # Enhanced recency scoring (stronger preference for recent content)
                years_old = current_year - year
                if years_old <= 0:  # Current or future year
                    recency_score = 1.0
                elif years_old == 1:  # Last year
                    recency_score = 0.8
                elif years_old == 2:  # 2 years ago
                    recency_score = 0.6
                elif years_old <= 4:  # 3-4 years ago
                    recency_score = 0.4
                else:  # 5+ years ago
                    recency_score = max(0.1, 1.0 - years_old * 0.15)
                    
            except Exception:
                recency_score = 0.3  # default for unparseable dates
            
            # Combined scoring: 60% relevance, 40% recency (stronger recency weight)
            relevance = 1.0 / (1.0 + chunk.get('distance', 1.0))
            chunk['score'] = relevance * 0.6 + recency_score * 0.4
            chunk['recency_score'] = recency_score
            chunk['relevance_score'] = relevance
        
        # Sort by combined score
        return sorted(chunks, key=lambda x: x['score'], reverse=True)


# --------------------------
# Response Generator (from rag_chat6)
# --------------------------
class ResponseGenerator:
    def __init__(self, adapter_path: str, base_model_path: str, system_msg: str):
        self.system_msg = system_msg
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        has_gpu = torch.cuda.is_available()
        on_mac = platform.system() == "Darwin"

        if has_gpu and not on_mac:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map={"": device},
                torch_dtype=torch.float16 if has_gpu or on_mac else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        if device == "mps":
            torch.mps.set_per_process_memory_fraction(0.0)

    def _format_prompt(self, query: str, chat_history: List[Dict], chunks: Optional[List[Dict]], use_citations: bool) -> List[Dict]:
        """Enhanced prompt formatting from rag_chat6.py"""
        if not chunks:
            return [
                {"role": "system", "content": self.system_msg},
                *chat_history,
                {"role": "user", "content": query},
            ]
        
        # Enhanced context block formatting (from rag_chat6.py)
        context_block = "═══════════════════════════════════════════════════\n"
        context_block += "📰 CURRENT INFORMATION (from recent sources)\n"
        context_block += "═══════════════════════════════════════════════════\n\n"
        
        for i, chunk in enumerate(chunks, 1):
            context_block += f"[{i}] {chunk['text']}\n"
            if chunk.get('date') != 'Unknown':
                context_block += f"    📅 {chunk['date']}"
                # Add recency and relevance info for debugging
                if chunk.get('recency_score') is not None:
                    context_block += f" (relevance: {chunk.get('relevance_score', 0):.2f}, recency: {chunk.get('recency_score', 0):.2f})"
                context_block += "\n"
            if chunk.get('source') != 'Unknown':
                context_block += f"    🔗 {chunk['source']}\n"
            context_block += "\n"
        
        context_block += "═══════════════════════════════════════════════════\n\n"
        
        # Enhanced instructions (more detailed guidance)
        if use_citations:
            instructions = """INSTRUCTIONS:
1. Use the information above to inform your response as Elon Musk
2. When stating facts from above, add citation like [1] or [2] after the fact
3. Respond naturally as Elon - don't sound robotic or mention "retrieved information"
4. If the context helps answer the question, integrate it seamlessly
5. If the context doesn't help, just respond normally as Elon would
6. Always maintain Elon's personality, opinions, and speaking style"""
        else:
            instructions = """INSTRUCTIONS:
1. Use the above information to inform your response as Elon Musk
2. Respond naturally and maintain Elon's personality, opinions, and speaking style  
3. Integrate relevant facts seamlessly into your response
4. Never mention that you're using "retrieved information" or "sources"
5. If the context doesn't help answer the question, respond normally as Elon would
6. Be detailed, opinionated, and engaging as Elon would be"""
        
        enhanced_system = f"{self.system_msg}\n\n{context_block}\n{instructions}"
        
        return [
            {"role": "system", "content": enhanced_system},
            *chat_history,
            {"role": "user", "content": query},
        ]

    @torch.inference_mode()
    def generate(self, query: str, chat_history: List[Dict], chunks: Optional[List[Dict]] = None) -> Tuple[str, float]:
        start = time.time()
        messages = self._format_prompt(query, chat_history, chunks, use_citations=False)
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        if device == "cuda":
            autocast_context = torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            from contextlib import nullcontext
            autocast_context = nullcontext()
        with autocast_context:
            output = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        response = self.tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        return response, (time.time() - start) * 1000


# --------------------------
# Flask App
# --------------------------
app = Flask(__name__)

generator: Optional[ResponseGenerator] = None
retriever: Optional[Retriever] = None
chat_histories: Dict[str, List[Dict]] = {}
ready: bool = False


def ensure_analyzer() -> bool:
    return analyzer_manager.start()


def load_components(adapter_path: str = ADAPTER_DEFAULT):
    global generator, retriever, ready
    if retriever is None:
        retriever_local = CHROMA_DB_PATH
        if not os.path.exists(retriever_local):
            retriever_local = os.path.join(BASE_DIR, "elon_chroma_db")
        retriever_inst = Retriever(retriever_local)
        retriever = retriever_inst
    if generator is None:
        generator = ResponseGenerator(adapter_path, BASE_MODEL_PATH, SYSTEM_MSG)
    ready = True


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy" if ready else "initializing",
        "ready": ready,
        "device": device,
        "rag_available": retriever.is_available() if retriever else False,
        "analyzer": "up" if analyzer_manager._is_running() else "down",
    })


@app.route('/reset', methods=['POST'])
def reset():
    data = request.get_json() or {}
    user_id = data.get('user_id', 'default')
    chat_histories[user_id] = []
    return jsonify({"status": "success", "message": f"Chat history reset for user {user_id}"})


@app.route('/end_session', methods=['POST'])
def end_session():
    """End chat session and save comprehensive logs"""
    data = request.get_json() or {}
    user_id = data.get('user_id', 'default')
    save_chat_session(user_id)
    return jsonify({"status": "success", "message": f"Chat session ended and saved for user {user_id}"})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        if 'input' not in data:
            return jsonify({"error": "Missing 'input' field"}), 400
        
        user_input = data['input']
        user_id = data.get('user_id', 'default')
        use_rag = bool(data.get('use_rag', True))

        logger.info(f"[Request] user_id={user_id}, query='{user_input[:100]}...', use_rag={use_rag}")

        if user_id not in chat_histories:
            chat_histories[user_id] = []
        chat_history = chat_histories[user_id]

        # Enhanced analyzer handling with robust error management
        should_retrieve = use_rag  # default fallback
        final_query = user_input
        raw_output = ""
        analysis_latency = 0.0
        analyzer_available = False
        
        if not ensure_analyzer():
            logger.warning("[Analyzer] Service unavailable - falling back to default RAG behavior")
            should_retrieve = use_rag
            final_query = f"Elon Musk {user_input}" if should_retrieve else user_input
        else:
            # Call analyzer with comprehensive error handling
            analysis_start = time.time()
            try:
                logger.info(f"[Analyzer] Calling service for query: {user_input[:100]}...")
                resp = requests.post(
                    f"{ANALYZER_URL}/analyze",
                    json={"query": user_input, "chat_history": chat_history},
                    timeout=30,
                )
                analysis_latency = (time.time() - analysis_start) * 1000
                
                if resp.status_code == 200:
                    j = resp.json()
                    should_retrieve = bool(j.get('should_retrieve', use_rag)) and use_rag
                    final_query = j.get('rewritten_query', user_input)
                    raw_output = j.get('raw_output', '')
                    analysis_method = j.get('analysis_method', 'Unknown')
                    analyzer_available = True
                    
                    decision = "RETRIEVE" if should_retrieve else "NO_RETRIEVE"
                    logger.info(f"[Analyzer] Decision: {decision} | Method: {analysis_method} | Latency: {analysis_latency:.0f}ms")
                    
                    if should_retrieve and final_query != user_input:
                        logger.info(f"[Analyzer] Query rewritten: '{final_query}'")
                    
                    # Log raw analyzer output (like rag_chat6.py)
                    if raw_output:
                        logger.info(f"[Analyzer] Raw output ({analysis_method}):")
                        logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                        logger.info(f"{raw_output}")
                        logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                else:
                    logger.warning(f"[Analyzer] HTTP error {resp.status_code} - falling back to default RAG")
                    should_retrieve = use_rag
                    final_query = f"Elon Musk {user_input}" if should_retrieve else user_input
                    
            except requests.exceptions.Timeout:
                analysis_latency = (time.time() - analysis_start) * 1000
                logger.warning(f"[Analyzer] Timeout after {analysis_latency:.0f}ms - falling back to RAG")
                should_retrieve = use_rag
                final_query = f"Elon Musk {user_input}"
                
            except requests.exceptions.ConnectionError:
                analysis_latency = (time.time() - analysis_start) * 1000
                logger.warning(f"[Analyzer] Connection error after {analysis_latency:.0f}ms - falling back to RAG")
                should_retrieve = use_rag
                final_query = f"Elon Musk {user_input}"
                
            except Exception as e:
                analysis_latency = (time.time() - analysis_start) * 1000
                logger.error(f"[Analyzer] Unexpected error after {analysis_latency:.0f}ms: {e} - falling back to RAG")
                should_retrieve = use_rag
                final_query = f"Elon Musk {user_input}"

        # Load components lazily
        load_components()

        # Enhanced retrieval with detailed logging
        chunks: List[Dict] = []
        retrieval_latency = 0.0
        
        if should_retrieve and retriever and retriever.is_available():
            try:
                chunks, retrieval_latency = retriever.retrieve(final_query, n_results=3)
                if chunks:
                    logger.info(f"[Retrieval] Found {len(chunks)} chunks | Latency: {retrieval_latency:.0f}ms")
                    # Log chunk scores for debugging
                    for i, chunk in enumerate(chunks):
                        score_info = ""
                        if chunk.get('score') is not None:
                            score_info = f" (score: {chunk['score']:.3f})"
                        logger.debug(f"  [{i+1}] {chunk['date']} - {chunk['source'][:50]}...{score_info}")
                else:
                    logger.info(f"[Retrieval] No chunks found | Latency: {retrieval_latency:.0f}ms")
            except Exception as e:
                logger.error(f"[Retrieval] Error: {e}")
                chunks = []
                retrieval_latency = 0.0
        elif should_retrieve:
            logger.warning("[Retrieval] RAG requested but retriever unavailable")

        # Generation with error handling
        try:
            response_text, gen_latency = generator.generate(user_input, chat_history, chunks if chunks else None)
            logger.info(f"[Generation] Success | Latency: {gen_latency:.0f}ms")
            logger.info(f"[Response] {response_text[:200]}...")
        except Exception as e:
            logger.error(f"[Generation] Error: {e}")
            return jsonify({"error": f"Generation failed: {str(e)}"}), 500

        # Update history (keep last 10 exchanges like original)
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response_text})
        if len(chat_history) > 20:
            chat_histories[user_id] = chat_history[-20:]
        else:
            chat_histories[user_id] = chat_history

        # Summary logging
        total_latency = analysis_latency + retrieval_latency + gen_latency
        logger.info(f"[Summary] Total: {total_latency:.0f}ms | Analysis: {analysis_latency:.0f}ms | Retrieval: {retrieval_latency:.0f}ms | Generation: {gen_latency:.0f}ms")

        # Comprehensive chat session logging
        analysis_data = {
            'should_retrieve': should_retrieve,
            'rewritten_query': final_query,
            'raw_output': raw_output,
            'analyzer_available': analyzer_available
        }
        latency_data = {
            'analysis_ms': analysis_latency,
            'retrieval_ms': retrieval_latency,
            'generation_ms': gen_latency,
            'total_ms': total_latency
        }
        log_chat_exchange(user_id, user_input, response_text, analysis_data, chunks, latency_data)

        return jsonify({
            "output": response_text,
            "query_type": "analyzed",
            "rag_used": len(chunks) > 0,
            "num_chunks": len(chunks),
            "latency": latency_data
        })
        
    except Exception as e:
        logger.error(f"[Error] Unhandled exception in predict: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 70)
    print("THINKING MODEL SERVER (Dual-Model HTTP API)")
    print("=" * 70)
    print(f"Port: {SERVER_PORT}")
    print(f"Device: {device}")
    print("=" * 70)

    # Ensure analyzer and lazy-load components early to catch issues
    ensure_analyzer()
    load_components()

    app.run(host='0.0.0.0', port=SERVER_PORT, debug=False)