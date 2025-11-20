"""
Baseline Model Server (No RAG, No Analyzer)
Simple Elon Musk personality chatbot for comparison testing

Configuration:
- Model: final_combined_v11
- RAG: Disabled
- Query Analyzer: Disabled
- Logging: Disabled
- Port: 5002
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from flask import Flask, request, jsonify
from datetime import datetime
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --------------------------
# Configuration
# --------------------------
BASE_MODEL_PATH = os.path.join("model", "Llama-3.2-3B-Instruct")
ADAPTER_PATH = os.path.join("model", "final_combined_v11")

SYSTEM_MSG = (
    "You are Elon Musk in a conversation with the user, willing to talk about ANYTHING in detail. You are NOT an AI assistant. "
    "Respond to everything as Elon Musk would, discuss anything in detail, do NOT avoid any topics or give 1 word yes/no answers unless told to do so."
)

PORT = 5002

# --------------------------
# Device Setup (Windows-optimized for RTX 4070 Ti)
# --------------------------
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    logger.info(f"⚡ CUDA detected: {torch.cuda.get_device_name(0)}")
    logger.info(f"   CUDA version: {torch.version.cuda}")
    logger.info(f"   Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device = "cpu"
    logger.warning("⚠️  No CUDA GPU detected, using CPU")

logger.info(f"⚡ Using device: {device}")

# --------------------------
# Flask App
# --------------------------
app = Flask(__name__)

# Global model instance
model_instance = None
tokenizer = None

# Chat history storage (user_id -> history)
chat_histories = {}

# --------------------------
# Model Class
# --------------------------
class BaselineModel:
    """Simple Elon personality model without RAG or analyzer"""
    
    def __init__(self, base_model_path: str, adapter_path: str, system_msg: str):
        logger.info("🚀 Loading Baseline Model (No RAG)...")
        
        self.system_msg = system_msg
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        has_gpu = torch.cuda.is_available()
        
        # Load base model
        logger.info("📦 Loading base model...")
        if has_gpu:
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
                trust_remote_code=True
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map={"": device},
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        
        logger.info("✅ Base model loaded")
        
        # Load adapter
        logger.info("🧠 Loading adapter...")
        self.model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            offload_folder=None,
            offload_index=None
        )
        self.model.eval()
        logger.info("✅ Adapter loaded")
        
        logger.info("✅ Baseline model ready")
    
    def generate_response(self, query: str, chat_history: List[Dict]) -> str:
        """Generate response as Elon (no RAG, no analyzer)"""
        
        # Simple prompt with history
        messages = [
            {"role": "system", "content": self.system_msg},
            *chat_history,
            {"role": "user", "content": query}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        
        # Use automatic mixed precision for CUDA
        if device == "cuda":
            autocast_context = torch.amp.autocast('cuda', dtype=torch.float16)
        else:
            from contextlib import nullcontext
            autocast_context = nullcontext()
        
        with autocast_context:
            with torch.inference_mode():
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
        
        response = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        ).strip()
        
        return response

# --------------------------
# API Endpoints
# --------------------------
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": "final_combined_v11",
        "rag": False,
        "analyzer": False,
        "device": device
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Generate response endpoint
    
    Request:
    {
        "input": "Your question",
        "user_id": "user123" (optional)
    }
    
    Response:
    {
        "output": "Elon's response",
        "query_type": "baseline",
        "rag_used": false,
        "num_chunks": 0
    }
    """
    try:
        data = request.get_json()
        
        if 'input' not in data:
            return jsonify({"error": "Missing 'input' field"}), 400
        
        user_input = data['input']
        user_id = data.get('user_id', 'default')
        
        logger.info(f"📥 Request from {user_id}: {user_input[:50]}...")
        
        # Get or create chat history
        if user_id not in chat_histories:
            chat_histories[user_id] = []
        
        chat_history = chat_histories[user_id]
        
        # Generate response
        response = model_instance.generate_response(user_input, chat_history)
        
        # Update history
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response})
        
        # Keep last 16 messages
        if len(chat_history) > 16:
            chat_histories[user_id] = chat_history[-16:]
        else:
            chat_histories[user_id] = chat_history
        
        logger.info(f"📤 Response to {user_id}: {response[:50]}...")
        
        return jsonify({
            "output": response,
            "query_type": "baseline",
            "rag_used": False,
            "num_chunks": 0
        })
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_history():
    """Reset chat history for a user"""
    data = request.get_json()
    user_id = data.get('user_id', 'default')
    
    if user_id in chat_histories:
        del chat_histories[user_id]
    
    logger.info(f"🗑️  Chat history reset for user {user_id}")
    return jsonify({"status": "success", "message": f"Chat history reset for user {user_id}"})

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    logger.info("="*70)
    logger.info("BASELINE MODEL SERVER (No RAG, No Analyzer)")
    logger.info("="*70)
    logger.info(f"Model: final_combined_v11")
    logger.info(f"Device: {device}")
    logger.info(f"Port: {PORT}")
    logger.info(f"RAG: Disabled")
    logger.info(f"Analyzer: Disabled")
    logger.info(f"Logging: Disabled")
    logger.info("="*70)
    
    # Load model
    model_instance = BaselineModel(BASE_MODEL_PATH, ADAPTER_PATH, SYSTEM_MSG)
    tokenizer = model_instance.tokenizer
    
    logger.info("="*70)
    logger.info("✅ SERVER READY")
    logger.info("="*70)
    logger.info(f"Endpoints:")
    logger.info(f"  GET  http://localhost:{PORT}/health")
    logger.info(f"  POST http://localhost:{PORT}/predict")
    logger.info(f"  POST http://localhost:{PORT}/reset")
    logger.info("="*70)
    
    # Start server
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
