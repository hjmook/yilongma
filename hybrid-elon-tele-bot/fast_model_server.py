"""
Fast Model Server - local copy of legacy server contract for hybrid use.

Provides the same API as the original `model_server.py` but lives entirely inside
this directory. Uses the base model and adapter from the project-level `model/`.
"""

from flask import Flask, request, jsonify
import torch
import platform
import os
import json
import time
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import chromadb
from typing import List, Dict
import logging


# --------------------------
# Paths
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "Llama-3.2-3B-Instruct")
# ADAPTER_PATH = os.path.join(PROJECT_ROOT, "model", "final_combined_v11")
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "model", "final_combined_v11_phase_1_only")
# ADAPTER_PATH = os.path.join(PROJECT_ROOT, "model", "single_phase")
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "knowledge_base", "elon_chroma_db")


SYSTEM_MSG = (
    "You are Elon Musk in a conversation with the user, nothing else. You are not an AI assistant. "
    "Respond to everything as Elon Musk would, discuss anything in detail, do NOT avoid any topics or give 1 word yes/no answers unless told to do so."
)


app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chat session logging (comprehensive JSON logs)
chat_sessions: Dict[str, Dict] = {}  # user_id -> session data

model = None
tokenizer = None
rag_retriever = None
query_classifier = None
chat_histories = {}
ready = False


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
        'mode': 'elon-fast',
        'exchanges': [],
        'total_queries': 0,
        'total_retrievals': 0,
        'classification_stats': {
            'factual_recent': 0,
            'factual_domain': 0,
            'conversational': 0
        }
    }
    logger.info(f"[Session] Started new chat session for user {user_id}: {session_id}")

def log_chat_exchange(user_id: str, query: str, response: str, classification: str, chunks: List[Dict], latency_data: Dict, retrieval_query: str = None):
    """Log a complete chat exchange with all metadata"""
    if user_id not in chat_sessions:
        start_chat_session(user_id)
    
    session = chat_sessions[user_id]
    session['total_queries'] += 1
    
    # Update classification stats
    if classification in session['classification_stats']:
        session['classification_stats'][classification] += 1
    
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
        'classification': classification,
        'retrieval_query': retrieval_query if retrieval_query else query,  # Show what was actually used for retrieval
        'retrieval': {
            'chunks_found': len(chunks),
            'chunks': truncated_chunks
        },
        'latency': latency_data,
        'query_number': session['total_queries']
    }
    
    session['exchanges'].append(exchange)
    
    if len(chunks) > 0:
        session['total_retrievals'] += 1

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
        logger.info(f"[Session] Classification breakdown: {session['classification_stats']}")
    except Exception as e:
        logger.error(f"[Session] Failed to save chat log: {e}")
    
    # Clean up memory
    del chat_sessions[user_id]


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# --------------------------
# RAG Query Classifier (ported from original model_server.py)
# --------------------------
class QueryClassifier:
    """Classifies queries to determine RAG usage strategy"""
    
    @staticmethod
    def classify_query(query: str) -> str:
        query_lower = query.lower()
        
        factual_recent_indicators = [
            "how many", "when did", "latest", "recent", "currently",
            "last quarter", "this year", "update on", "news about",
            "what happened", "status of", "just announced", "today",
            "this week", "this month", "new", "upcoming"
        ]
        
        factual_domain_indicators = [
            "why", "how does", "explain", "what's your approach",
            "philosophy on", "thoughts on", "how would you",
            "strategy for", "plan for", "vision for"
        ]
        
        conversational_indicators = [
            "how are you", "what's up", "tell me about yourself",
            "do you like", "are you", "can you", "will you",
            "hello", "hi", "hey", "good morning"
        ]
        
        recent_score = sum(ind in query_lower for ind in factual_recent_indicators)
        domain_score = sum(ind in query_lower for ind in factual_domain_indicators)
        conv_score = sum(ind in query_lower for ind in conversational_indicators)
        
        if recent_score > 0:
            return 'factual_recent'
        elif conv_score > 0:
            return 'conversational'
        elif domain_score > 0:
            return 'factual_domain'
        else:
            return 'conversational' if len(query.split()) < 5 else 'factual_domain'
    
    @staticmethod
    def assess_complexity(query: str) -> str:
        word_count = len(query.split())
        has_multiple_questions = query.count('?') > 1
        has_conjunctions = any(w in query.lower() for w in ['and', 'but', 'also', 'plus', 'additionally'])
        
        if word_count < 8 and not has_multiple_questions:
            return 'simple'
        elif word_count > 20 or has_multiple_questions or has_conjunctions:
            return 'complex'
        else:
            return 'medium'
    
    @staticmethod
    def rewrite_query_for_retrieval(query: str) -> str:
        """Simple rule-based query rewriting for better RAG retrieval"""
        rewritten = query
        
        # Convert "you/your" to "Elon Musk" for better retrieval
        import re
        
        # Replace "you" variations with "Elon Musk" (case-insensitive)
        # Handle whole words only to avoid replacing "your" in "yourself", etc.
        rewritten = re.sub(r'\byou\b', 'Elon Musk', rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r'\byour\b', "Elon Musk's", rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r'\byou\'re\b', 'Elon Musk is', rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r'\byou\'ve\b', 'Elon Musk has', rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r'\byou\'ll\b', 'Elon Musk will', rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r'\byou\'d\b', 'Elon Musk would', rewritten, flags=re.IGNORECASE)
        
        return rewritten.strip()


class RAGRetriever:
    """Handles adaptive retrieval from ChromaDB"""
    
    def __init__(self, chroma_path: str):
        self.collection = None
        if not os.path.exists(chroma_path):
            logger.warning(f"ChromaDB path not found: {chroma_path}")
            return
        try:
            self.client = chromadb.PersistentClient(path=chroma_path)
            self.collection = self.client.get_collection(name="elon_musk_knowledge")
            logger.info(f"✓ RAG database loaded: {self.collection.count()} chunks available")
        except Exception as e:
            logger.warning(f"Could not load RAG database: {e}")
            self.collection = None

    def is_available(self) -> bool:
        return self.collection is not None and self.collection.count() > 0

    def retrieve_context(self, query: str, complexity: str) -> List[Dict]:
        """Retrieve context with complexity-based chunk count"""
        if not self.is_available():
            return []
        try:
            n_results = {'simple': 1, 'medium': 3, 'complex': 5}.get(complexity, 2)
            results = self.collection.query(query_texts=[query], n_results=n_results)
            formatted = self._format_results(results)
            # Apply recency-weighted reranking (same as elon-thinking)
            return self._rerank_by_recency(formatted)
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []
    
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
                'distance': distances[i] if i < len(distances) else 1.0
            })
        
        return formatted
    
    def _rerank_by_recency(self, chunks: List[Dict]) -> List[Dict]:
        """Rerank chunks with recency weighting (same as elon-thinking)"""
        if not chunks:
            return chunks
        
        import re
        from datetime import datetime
        current_year = datetime.now().year
        
        for chunk in chunks:
            date_str = chunk.get('date', '2020-01-01')
            
            # Extract year from date string
            try:
                if '-' in date_str:
                    year = int(date_str.split('-')[0])
                else:
                    # Look for 4-digit year in text
                    year_match = re.search(r'20\d{2}', date_str)
                    if year_match:
                        year = int(year_match.group())
                    else:
                        year = 2020  # default fallback
                
                # Calculate recency score
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
            
            # Calculate semantic relevance from distance
            relevance = 1.0 / (1.0 + chunk.get('distance', 1.0))
            
            # Combined scoring: 60% relevance, 40% recency
            chunk['score'] = relevance * 0.6 + recency_score * 0.4
            chunk['recency_score'] = recency_score
            chunk['relevance_score'] = relevance
        
        # Sort by combined score
        return sorted(chunks, key=lambda x: x['score'], reverse=True)


def load_model():
    global model, tokenizer, rag_retriever, query_classifier, ready
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
            BASE_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            device_map={"": device},
            torch_dtype=torch.float16 if has_gpu or on_mac else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    rag_retriever = RAGRetriever(CHROMA_DB_PATH)
    query_classifier = QueryClassifier()
    ready = True


@torch.inference_mode()
def generate_response(messages, max_new_tokens=180):
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    if device == "cuda":
        autocast_context = torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        from contextlib import nullcontext
        autocast_context = nullcontext()
    with autocast_context:
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response.strip()


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy" if ready else "initializing",
        "ready": ready,
        "device": device,
        "rag_available": rag_retriever.is_available() if rag_retriever else False,
    })


@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    latency_breakdown = {}
    
    try:
        data = request.get_json() or {}
        if 'input' not in data:
            return jsonify({"error": "Missing 'input' field"}), 400
        user_input = data['input']
        user_id = data.get('user_id', 'default')
        use_rag = data.get('use_rag', True)

        logger.info(f"Received request from user {user_id}: {user_input[:50]}...")

        if user_id not in chat_histories:
            chat_histories[user_id] = []
        chat_history = chat_histories[user_id]

        # Smart RAG decision using the original rule-based classifier
        classification_start = time.time()
        query_type = 'conversational'
        complexity = 'simple'
        retrieved_chunks = []
        retrieval_query = user_input  # Track what query is actually used for retrieval
        
        if use_rag and rag_retriever and rag_retriever.is_available():
            query_type = query_classifier.classify_query(user_input)
            complexity = query_classifier.assess_complexity(user_input)
            latency_breakdown['classification_ms'] = (time.time() - classification_start) * 1000
            
            # Only retrieve for factual queries like the original model_server.py
            if query_type in ['factual_recent', 'factual_domain']:
                # Rewrite query for better retrieval (you → Elon Musk)
                retrieval_query = query_classifier.rewrite_query_for_retrieval(user_input)
                
                retrieval_start = time.time()
                retrieved_chunks = rag_retriever.retrieve_context(retrieval_query, complexity)
                latency_breakdown['retrieval_ms'] = (time.time() - retrieval_start) * 1000
                logger.info(f"RAG: {len(retrieved_chunks)} chunks | {query_type} | {complexity}")
                if retrieval_query != user_input:
                    logger.info(f"Query rewritten: '{user_input}' → '{retrieval_query}'")
        else:
            latency_breakdown['classification_ms'] = (time.time() - classification_start) * 1000

        # Format prompt with RAG context (original style)
        if not retrieved_chunks:
            messages = [
                {"role": "system", "content": SYSTEM_MSG},
                *chat_history,
                {"role": "user", "content": user_input},
            ]
        else:
            # Original style context formatting
            context_block = "=== CURRENT FACTS (Use these in your response) ===\n\n"
            for i, chunk in enumerate(retrieved_chunks, 1):
                context_block += f"[SOURCE {i}]\n"
                context_block += f"{chunk['text']}\n"
                if chunk.get('date') != 'Unknown':
                    context_block += f"Date: {chunk['date']}\n"
                context_block += "\n"
            context_block += "=== END CURRENT FACTS ===\n\n"
            
            enhanced_system = f"""{SYSTEM_MSG}

{context_block}

IMPORTANT: Use the facts above naturally in your response as Elon would, without mentioning you're using retrieved information."""
            
            messages = [
                {"role": "system", "content": enhanced_system},
                *chat_history,
                {"role": "user", "content": user_input},
            ]

        generation_start = time.time()
        response_text = generate_response(messages)
        latency_breakdown['generation_ms'] = (time.time() - generation_start) * 1000

        # Update chat history (keep last 10 exchanges like original)
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response_text})
        if len(chat_history) > 20:
            chat_histories[user_id] = chat_history[-20:]
        else:
            chat_histories[user_id] = chat_history

        # Calculate total latency
        total_latency = (time.time() - start_time) * 1000
        latency_breakdown['total_ms'] = total_latency
        
        # Log the exchange for comprehensive session logging
        # For elon-fast, retrieval query may be rewritten (you → Elon Musk)
        log_chat_exchange(
            user_id=user_id,
            query=user_input,
            response=response_text,
            classification=query_type,
            chunks=retrieved_chunks,
            latency_data=latency_breakdown,
            retrieval_query=retrieval_query  # Show rewritten query if different
        )
        
        logger.info(f"Request completed in {total_latency:.0f}ms - Classification: {query_type}, Chunks: {len(retrieved_chunks)}")

        return jsonify({
            "output": response_text,
            "query_type": query_type,
            "rag_used": len(retrieved_chunks) > 0,
            "num_chunks": len(retrieved_chunks),
        })
    except Exception as e:
        logger.error(f"Error in prediction: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset_history():
    data = request.get_json() or {}
    user_id = data.get('user_id', 'default')
    
    # Don't save session on reset, only on end_session
    # Just reset chat history
    chat_histories[user_id] = []
    
    return jsonify({"status": "success", "message": f"Chat history reset for user {user_id}"})


@app.route('/end_session', methods=['POST'])
def end_session():
    """End chat session and save comprehensive logs"""
    data = request.get_json() or {}
    user_id = data.get('user_id', 'default')
    save_chat_session(user_id)
    return jsonify({"status": "success", "message": f"Chat session ended and saved for user {user_id}"})


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5001, debug=False)


