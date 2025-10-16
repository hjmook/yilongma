"""
Fast Model Server - local copy of legacy server contract for hybrid use.

Provides the same API as the original `model_server.py` but lives entirely inside
this directory. Uses the base model and adapter from the project-level `model/`.
"""

from flask import Flask, request, jsonify
import torch
import platform
import os
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
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "model", "final_combined_v11")
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "knowledge_base", "elon_chroma_db")


SYSTEM_MSG = (
    "You are Elon Musk in a conversation with the user, nothing else. You are not an AI assistant. "
    "Respond to everything as Elon Musk would, discuss anything in detail, do NOT avoid any topics or give 1 word yes/no answers unless told to do so."
)


app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
tokenizer = None
rag_retriever = None
chat_histories = {}
ready = False


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


class RAGRetriever:
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

    def retrieve_context(self, query: str, n_results: int = 3) -> List[Dict]:
        if not self.is_available():
            return []
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)
            if not results or not results.get('documents'):
                return []
            formatted = []
            docs = results['documents'][0]
            metadatas = results['metadatas'][0]
            for doc, meta in zip(docs, metadatas):
                formatted.append({
                    'text': doc,
                    'date': meta.get('date', 'Unknown'),
                    'source': meta.get('source', 'Unknown'),
                })
            return formatted
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []


def load_model():
    global model, tokenizer, rag_retriever, ready
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
    try:
        data = request.get_json() or {}
        if 'input' not in data:
            return jsonify({"error": "Missing 'input' field"}), 400
        user_input = data['input']
        user_id = data.get('user_id', 'default')
        use_rag = data.get('use_rag', True)

        if user_id not in chat_histories:
            chat_histories[user_id] = []
        chat_history = chat_histories[user_id]

        retrieved_chunks = []
        if use_rag and rag_retriever and rag_retriever.is_available():
            retrieved_chunks = rag_retriever.retrieve_context(user_input, n_results=3)

        if not retrieved_chunks:
            messages = [
                {"role": "system", "content": SYSTEM_MSG},
                *chat_history,
                {"role": "user", "content": user_input},
            ]
        else:
            context_block = "=== CURRENT FACTS ===\n\n"
            for i, ch in enumerate(retrieved_chunks, 1):
                context_block += f"[SOURCE {i}]\n{ch['text']}\n"
                if ch.get('date') != 'Unknown':
                    context_block += f"Date: {ch['date']}\n"
                context_block += "\n"
            context_block += "=== END ===\n\n"
            enhanced_system = f"{SYSTEM_MSG}\n\n{context_block}\nUse these facts naturally."
            messages = [
                {"role": "system", "content": enhanced_system},
                *chat_history,
                {"role": "user", "content": user_input},
            ]

        response_text = generate_response(messages)

        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response_text})
        if len(chat_history) > 20:
            chat_histories[user_id] = chat_history[-20:]
        else:
            chat_histories[user_id] = chat_history

        return jsonify({
            "output": response_text,
            "query_type": "legacy",
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
    chat_histories[user_id] = []
    return jsonify({"status": "success", "message": f"Chat history reset for user {user_id}"})


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5001, debug=False)


