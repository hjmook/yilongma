"""
Raw Llama 3.2-3B-Instruct Chat (No Fine-tuning)
Run the base model without any PEFT adapters
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datetime import datetime
from typing import List, Dict

# Configuration
BASE_MODEL_PATH = os.path.join("model", "Llama-3.2-3B-Instruct")

# System message (you can customize this)
SYSTEM_MSG = "You are a helpful AI assistant based on Llama 3.2."

# Device Setup (Windows-optimized for RTX 4070 Ti)
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    print(f"⚡ CUDA detected: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device = "cpu"
    print(f"⚠️  No CUDA GPU detected, using CPU (will be slower)")

print(f"⚡ Using device: {device}")

class RawLlamaChat:
    """Raw Llama 3.2-3B-Instruct without fine-tuning"""
    
    def __init__(self, base_model_path: str, system_msg: str):
        print(f"\n🚀 Loading Raw Llama 3.2-3B-Instruct...\n")
        
        self.system_msg = system_msg
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        has_gpu = torch.cuda.is_available()
        
        # Load base model
        print(f"📦 Loading base model...")
        if has_gpu:
            # Use 4-bit quantization for GPU
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # CPU fallback
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map={"": device},
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        
        self.model.eval()
        print(f"✅ Raw Llama 3.2-3B-Instruct loaded\n")
    
    def generate_response(self, query: str, chat_history: List[Dict]) -> str:
        """Generate response using raw base model"""
        
        # Format prompt
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

def start_conversation(llama_chat: RawLlamaChat):
    """Main conversation loop"""
    print("\n" + "="*70)
    print("💬 Raw Llama 3.2-3B-Instruct Chat")
    print("="*70)
    print("Commands:")
    print("  'exit' / 'quit' - End conversation")
    print("  'clear'         - Clear conversation history")
    print("  'system'        - Change system message\n")
    
    chat_history = []
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ["exit", "quit"]:
            print("\n👋 Goodbye!")
            break
        
        if user_input.lower() == "clear":
            chat_history = []
            print("\n🗑️  Conversation history cleared\n")
            continue
        
        if user_input.lower() == "system":
            new_system = input("Enter new system message: ").strip()
            if new_system:
                llama_chat.system_msg = new_system
                print(f"\n✅ System message updated\n")
            continue
        
        if not user_input:
            continue
        
        # Generate response
        response = llama_chat.generate_response(user_input, chat_history)
        print(f"Assistant: {response}\n")
        
        # Update history
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response})
        
        # Keep manageable
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]

if __name__ == "__main__":
    print("\n" + "="*70)
    print("RAW LLAMA 3.2-3B-INSTRUCT (No Fine-tuning)")
    print("="*70)
    
    # Optionally customize system message
    print("\nDefault system message:")
    print(f'"{SYSTEM_MSG}"')
    
    custom = input("\nUse custom system message? (y/n): ").strip().lower()
    if custom == 'y':
        SYSTEM_MSG = input("Enter system message: ").strip() or SYSTEM_MSG
    
    print("\n" + "="*70)
    
    # Load model
    llama_chat = RawLlamaChat(BASE_MODEL_PATH, SYSTEM_MSG)
    
    print("="*70)
    print("✅ READY TO CHAT")
    print("="*70)
    print(f"Platform: Windows")
    print(f"Device: {device.upper()}")
    print(f"Model: Raw Llama 3.2-3B-Instruct (no adapters)")
    print("="*70)
    
    # Start conversation
    start_conversation(llama_chat)
