"""
Detailed Model Server (HTTP API) for Telegram integration

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
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
import torch
import platform
from flask import Flask, jsonify, request
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import chromadb


# --------------------------
# Configuration
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "Llama-3.2-3B-Instruct")
ADAPTER_DEFAULT = os.path.join(PROJECT_ROOT, "model", "final_combined_v11")
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "knowledge_base", "elon_chroma_db")

SERVER_PORT = int(os.environ.get("DETAILED_SERVER_PORT", "5055"))
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
        if not chunks:
            return chunks
        current_year = datetime.now().year
        for chunk in chunks:
            date_str = chunk.get('date', '2020-01-01')
            try:
                year = int(date_str[:4])
                recency_score = max(0, 1.0 - (current_year - year) * 0.1)
            except Exception:
                recency_score = 0.5
            relevance = 1.0 / (1.0 + chunk.get('distance', 1.0))
            chunk['score'] = relevance * 0.7 + recency_score * 0.3
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
        if not chunks:
            return [
                {"role": "system", "content": self.system_msg},
                *chat_history,
                {"role": "user", "content": query},
            ]
        context_block = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        context_block += "📰 CURRENT INFORMATION (from recent sources)\n"
        context_block += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        for i, chunk in enumerate(chunks, 1):
            context_block += f"[{i}] {chunk['text']}\n"
            if chunk.get('date') != 'Unknown':
                context_block += f"    📅 {chunk['date']}\n"
            context_block += "\n"
        context_block += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        instructions = (
            "INSTRUCTIONS:\n1. Use the information above to inform your response  "
            "\n2. Respond naturally as Elon Musk would\n3. Incorporate relevant facts seamlessly\n4. If the information doesn't help, just respond normally"
        )
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


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        if 'input' not in data:
            return jsonify({"error": "Missing 'input' field"}), 400
        user_input = data['input']
        user_id = data.get('user_id', 'default')
        use_rag = bool(data.get('use_rag', True))

        if user_id not in chat_histories:
            chat_histories[user_id] = []
        chat_history = chat_histories[user_id]

        # Ensure analyzer and components are ready
        if not ensure_analyzer():
            # Analyzer down; proceed with best-effort (no analysis) RAG
            should_retrieve = use_rag
            final_query = f"Elon Musk {user_input}" if should_retrieve else user_input
            raw_output = ""
        else:
            # Call analyzer
            t0 = time.time()
            try:
                resp = requests.post(
                    f"{ANALYZER_URL}/analyze",
                    json={"query": user_input, "chat_history": chat_history},
                    timeout=30,
                )
                if resp.status_code == 200:
                    j = resp.json()
                    should_retrieve = bool(j.get('should_retrieve', use_rag)) and use_rag
                    final_query = j.get('rewritten_query', user_input)
                    raw_output = j.get('raw_output', '')
                    print(f"[Analysis] user='{user_input}' should_retrieve={should_retrieve} final_query='{final_query}' latency_ms={(time.time()-t0)*1000:.0f}")
                    if raw_output:
                        print(f"[Analysis raw]\n{raw_output}\n---")
                else:
                    should_retrieve = use_rag
                    final_query = f"Elon Musk {user_input}" if should_retrieve else user_input
                    raw_output = ""
            except Exception:
                should_retrieve = use_rag
                final_query = f"Elon Musk {user_input}" if should_retrieve else user_input
                raw_output = ""

        # Load components lazily
        load_components()

        # Retrieval
        chunks: List[Dict] = []
        if should_retrieve and retriever and retriever.is_available():
            chunks, retrieval_latency = retriever.retrieve(final_query, n_results=3)
            print(f"[Retrieval] chunks={len(chunks)} latency_ms={retrieval_latency:.0f}")

        # Generation
        response_text, gen_latency = generator.generate(user_input, chat_history, chunks if chunks else None)
        print(f"[Generation] latency_ms={gen_latency:.0f}")
        print(f"[Elon] {response_text}")

        # Update history
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response_text})
        if len(chat_history) > 20:
            chat_histories[user_id] = chat_history[-20:]
        else:
            chat_histories[user_id] = chat_history

        return jsonify({
            "output": response_text,
            "query_type": "analyzed",
            "rag_used": len(chunks) > 0,
            "num_chunks": len(chunks),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 70)
    print("DETAILED MODEL SERVER (Dual-Model HTTP API)")
    print("=" * 70)
    print(f"Port: {SERVER_PORT}")
    print(f"Device: {device}")
    print("=" * 70)

    # Ensure analyzer and lazy-load components early to catch issues
    ensure_analyzer()
    load_components()

    app.run(host='0.0.0.0', port=SERVER_PORT, debug=False)


