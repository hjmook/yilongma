import torch
import platform
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import chromadb
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import json
import time

# --------------------------
# Configuration
# --------------------------
BASE_MODEL_PATH = os.path.join("model", "Llama-3.2-3B-Instruct")
PHASE1_ADAPTER_PATH = os.path.join("model", "final_combined_v11_phase_1_only")  # Loads only Phase 1

SYSTEM_MSG = (
    "You are Elon Musk in a conversation with the user, willing to talk about ANYTHING in detail. You are NOT an AI assistant. "
    "Respond to everything as Elon Musk would, discuss anything in detail, do NOT avoid any topics or give 1 word yes/no answers unless told to do so.\n\n"
    "CRITICAL: If provided with [CONTEXT] or numbered sources [1], [2], etc., those contain the MOST RECENT AND ACCURATE information. "
    "If your memory conflicts with the provided context, ALWAYS trust the context and acknowledge any updates naturally."
)

CHROMA_DB_PATH = os.path.join("knowledge_base", "elon_chroma_db")

# Fallback to current directory if knowledge_base doesn't exist
if not os.path.exists(CHROMA_DB_PATH):
    CHROMA_DB_PATH = "./elon_chroma_db"

# --------------------------
# Device Setup
# --------------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"âš¡ Using device: {device}")

# --------------------------
# Retriever
# --------------------------
class Retriever:
    """RAG retriever for knowledge base"""
    
    def __init__(self, chroma_path: str):
        self.collection = None
        
        if not os.path.exists(chroma_path):
            print(f"âš ï¸  ChromaDB path not found: {chroma_path}")
            print(f"   Expected location: {os.path.abspath(chroma_path)}")
            return
        
        try:
            self.client = chromadb.PersistentClient(path=chroma_path)
            self.collection = self.client.get_collection(name="elon_musk_knowledge")
            print(f"âœ… RAG database loaded: {self.collection.count()} chunks available")
        except Exception as e:
            print(f"âš ï¸  Warning: Could not load RAG database: {e}")
            self.collection = None
    
    def is_available(self) -> bool:
        return self.collection is not None and self.collection.count() > 0
    
    def retrieve(self, query: str, n_results: int = 3) -> Tuple[List[Dict], float]:
        """
        Retrieve relevant chunks.
        
        Returns:
            (chunks, latency_ms)
        """
        if not self.is_available():
            return [], 0.0
        
        start_time = time.time()
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            formatted = self._format_results(results)
            formatted = self._rerank_by_recency(formatted)
            
            latency = (time.time() - start_time) * 1000
            
            return formatted, latency
        
        except Exception as e:
            print(f"âš ï¸  Retrieval error: {e}")
            return [], 0.0
    
    def _format_results(self, results: Dict) -> List[Dict]:
        """Format ChromaDB results"""
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
        """Rerank giving preference to recent content"""
        if not chunks:
            return chunks
        
        current_year = datetime.now().year
        
        for chunk in chunks:
            date_str = chunk.get('date', '2020-01-01')
            try:
                year = int(date_str[:4])
                recency_score = max(0, 1.0 - (current_year - year) * 0.1)
            except:
                recency_score = 0.5
            
            relevance = 1.0 / (1.0 + chunk.get('distance', 1.0))
            chunk['score'] = relevance * 0.7 + recency_score * 0.3
        
        return sorted(chunks, key=lambda x: x['score'], reverse=True)

# --------------------------
# Phase 1 Only System
# --------------------------
class Phase1System:
    """System using ONLY Phase 1 adapter (identity layer)"""
    
    ANALYSIS_PROMPT_TEMPLATE = """You are a query analysis assistant for Elon Musk.

Your job: Decide if a query needs factual information retrieval from a knowledge base, and if so, rewrite it for search. 

Guidelines (EVEN IF TONE IS INFORMAL AND CONVERSATIONAL):
- Questions/Remarks about nouns related to Elon Musk (like SpaceX, Tesla, The Boring Company, DOGE, Grok, Trump) -> RETRIEVE
- Factual questions (what/when/how many/latest/recent/status) â†' RETRIEVE
- Follow-up questions with pronouns (it/that/there/he) â†' RETRIEVE (resolve pronouns using context)
- Questions about specific events, numbers, dates â†' RETRIEVE
- Ambiguous terms (like "DOGE" could be Dogecoin or Dept of Govt Efficiency) â†' RETRIEVE (disambiguate)
- Questions about personal life -> RETRIEVE
- Greetings â†' NO_RETRIEVE
- Personal philosophy about motivations â†' NO_RETRIEVE  


Output format (IMPORTANT - follow exactly):
- If retrieval NOT needed: "NO_RETRIEVE"
- If retrieval needed: "RETRIEVE: <rewritten query as clear, standalone search query>"

Examples:

Conversation: []
User: "Hey, how are you doing?"
Output: NO_RETRIEVE

---
Conversation: []
User: "What's the latest on Starship?"
Output: RETRIEVE: SpaceX Starship latest test flight results 2025

---
Conversation:
User: "Tell me about your time in DOGE"
Elon: "I invested in Dogecoin back in 2014..."
User: "No, I mean the department of government efficiency"
Elon: "Oh yeah, you want to talk about it?"

User: "Go ahead, tell me how was your time there"
Output: RETRIEVE: Elon Musk experience at Department of Government Efficiency role

---
Conversation:
User: "What do you think about Mars?"
Elon: "Mars is the future of humanity..."

User: "Why are you so obsessed with it?"
ELon: NO_RETRIEVE

---
Conversation: []
User: "Did you really work with Trump?"
Elon: RETRIEVE: Elon Musk Trump administration role DOGE 2025

---
Conversation:
User: "Tell me about Tesla"
Elon: "Tesla is revolutionizing transportation..."

User: "What about their latest earnings?"
Output: RETRIEVE: Tesla latest quarterly earnings report 2025

---

Now analyze this query:

Conversation:
{history}

User: "{query}"

Output:"""

    def __init__(self, base_model_path: str, adapter_path: str, system_msg: str):
        """Load base model + Phase 1 adapter only"""
        
        print(f"ðŸš€ Loading Phase 1 Only System...\n")
        
        self.system_msg = system_msg
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        has_gpu = torch.cuda.is_available()
        on_mac = platform.system() == "Darwin"
        
        # Load base model
        print(f"¦ Loading base model...")
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
                trust_remote_code=True
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map={"": device},
                torch_dtype=torch.float16 if has_gpu or on_mac else torch.float32,
                low_cpu_mem_usage=False,
                trust_remote_code=True
            )
        
        print(f"âœ… Base model loaded\n")
        
        # Load ONLY Phase 1 adapter
        print(f"ðŸ§  Loading Phase 1 adapter (identity layer, r=8)...")
        self.model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            is_trainable=False
        )
        self.model.eval()
        print(f"âœ… Phase 1 adapter loaded\n")
        
        # For compatibility with dual-model interface
        self.base_model = self.model
        self.finetuned_model = self.model
        
        # Set memory management for MPS
        if device == "mps":
            torch.mps.set_per_process_memory_fraction(0.0)
    
    def analyze_query(self, query: str, chat_history: List[Dict]) -> Tuple[bool, str, str, float]:
        """
        Analyze query using Phase 1 model to decide if retrieval needed.
        
        Returns:
            (should_retrieve, final_query, raw_output, latency_ms)
        """
        start_time = time.time()
        
        # Build prompt with conversation context
        history_str = self._format_history(chat_history)
        prompt = self.ANALYSIS_PROMPT_TEMPLATE.format(history=history_str, query=query)
        
        # Generate analysis
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        raw_output = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        ).strip()
        
        # Parse output
        should_retrieve, final_query = self._parse_analysis(raw_output, query)
        
        latency = (time.time() - start_time) * 1000
        
        return should_retrieve, final_query, raw_output, latency
    
    def generate_response(
        self, 
        query: str, 
        chat_history: List[Dict],
        chunks: Optional[List[Dict]] = None,
        use_citations: bool = True
    ) -> Tuple[str, float]:
        """
        Generate response using Phase 1 model as Elon.
        
        Returns:
            (response, latency_ms)
        """
        start_time = time.time()
        
        # Format prompt with context
        messages = self._format_prompt(query, chat_history, chunks, use_citations)
        
        # Generate
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        
        if device == "cuda":
            autocast_context = torch.cuda.amp.autocast(dtype=torch.float16)
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
        
        latency = (time.time() - start_time) * 1000
        
        return response, latency
    
    def _format_history(self, chat_history: List[Dict], max_turns: int = 10) -> str:
        """Format recent conversation history for prompt"""
        if not chat_history:
            return "[]"
        
        recent = chat_history[-(max_turns * 2):] if len(chat_history) > max_turns * 2 else chat_history
        
        formatted = []
        for msg in recent:
            role = "User" if msg['role'] == 'user' else "Assistant"
            formatted.append(f'{role}: "{msg["content"]}"')
        
        return "\n".join(formatted)
    
    def _parse_analysis(self, output: str, original_query: str) -> Tuple[bool, str]:
        """Parse model output into decision + query"""
        output = output.strip()
        
        if "NO_RETRIEVE" in output.upper() or output.upper().startswith("NO"):
            return False, original_query
        
        if "RETRIEVE:" in output.upper():
            parts = output.split(":", 1)
            if len(parts) == 2:
                rewritten = parts[1].strip().strip('"\'')
                if len(rewritten) > 5 and len(rewritten.split()) >= 2:
                    return True, rewritten
        
        negative_words = ["no", "don't", "not needed", "unnecessary", "skip"]
        if any(word in output.lower() for word in negative_words):
            return False, original_query
        
        meta_phrases = ["i think", "the user", "this query", "should", "needs", "wants"]
        if len(output.split()) > 3 and not any(phrase in output.lower() for phrase in meta_phrases):
            return True, output
        
        return True, original_query
    
    def _format_prompt(
        self, 
        query: str, 
        chat_history: List[Dict],
        chunks: Optional[List[Dict]],
        use_citations: bool
    ) -> List[Dict]:
        """Format prompt with optional RAG context"""
        
        if not chunks:
            return [
                {"role": "system", "content": self.system_msg},
                *chat_history,
                {"role": "user", "content": query}
            ]
        
        # Build context block
        context_block = "â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•\n"
        context_block += "ðŸ CURRENT INFORMATION (from recent sources)\n"
        context_block += "â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•\n\n"
        
        for i, chunk in enumerate(chunks, 1):
            context_block += f"[{i}] {chunk['text']}\n"
            if chunk.get('date') != 'Unknown':
                context_block += f"    ðŸ {chunk['date']}\n"
            context_block += "\n"
        
        context_block += "â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•\n\n"
        
        if use_citations:
            instructions = """INSTRUCTIONS:
1. Use the information above to inform your response
2. When stating facts from above, add citation like [1] or [2] after the fact
3. Respond naturally as Elon - don't sound robotic
4. If the information doesn't help, just respond normally

Example: "Yeah, I was involved with DOGE [1], but that wrapped up earlier this year [2]."
"""
        else:
            instructions = """INSTRUCTIONS:
1. Use the information above to inform your response  
2. Respond naturally as Elon Musk would
3. Incorporate relevant facts without explicitly mentioning sources
4. If the information doesn't help, just respond normally
"""
        
        enhanced_system = f"{self.system_msg}\n\n{context_block}\n{instructions}"
        
        return [
            {"role": "system", "content": enhanced_system},
            *chat_history,
            {"role": "user", "content": query}
        ]

# --------------------------
# Main Conversation Loop
# --------------------------
def start_conversation(
    system: Phase1System,
    retriever: Retriever,
    use_rag: bool = True,
    use_citations: bool = True,
    enable_logging: bool = True
):
    print("\n" + "="*70)
    print("ðŸ'¬ Phase 1 Only Elon Musk Chatbot" + (" (RAG Enhanced)" if use_rag else ""))
    print("="*70)
    print("Commands:")
    print("  'exit' / 'quit'     - End conversation")
    print("  'toggle rag'        - Switch RAG on/off")
    print("  'toggle citations'  - Switch citations on/off")
    print("  'toggle logging'    - Switch logging on/off")
    print("  'info'              - Show last query analysis")
    print("  'clear'             - Clear conversation history\n")
    
    chat_history = []
    last_analysis = None
    logs = []
    
    # Dynamic toggles
    rag_enabled = use_rag and retriever.is_available()
    citations_enabled = use_citations
    logging_enabled = enable_logging
    
    if use_rag and not retriever.is_available():
        print("âš ï¸  RAG database not available. Continuing without RAG.\n")
        rag_enabled = False
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ["exit", "quit"]:
            if logging_enabled and logs:
                log_file = f"phase1_chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(log_file, 'w') as f:
                    json.dump(logs, f, indent=2)
                print(f"\nðŸ'¾ Chat log saved to {log_file}")
            print("\nðŸ'‹ Goodbye!")
            break
        
        if user_input.lower() == "toggle rag":
            rag_enabled = not rag_enabled
            print(f"\n{'âœ…' if rag_enabled else 'âŒ'} RAG is now {'ON' if rag_enabled else 'OFF'}\n")
            continue
        
        if user_input.lower() == "toggle citations":
            citations_enabled = not citations_enabled
            print(f"\n{'âœ…' if citations_enabled else 'âŒ'} Citations are now {'ON' if citations_enabled else 'OFF'}\n")
            continue
        
        if user_input.lower() == "toggle logging":
            logging_enabled = not logging_enabled
            print(f"\n{'âœ…' if logging_enabled else 'âŒ'} Logging is now {'ON' if logging_enabled else 'OFF'}\n")
            continue
        
        if user_input.lower() == "clear":
            chat_history = []
            print("\nðŸ—'ï¸  Conversation history cleared\n")
            continue
        
        if user_input.lower() == "info":
            if last_analysis:
                print("\n" + "="*70)
                print("LAST QUERY ANALYSIS")
                print("="*70)
                print(f"Should retrieve: {last_analysis['should_retrieve']}")
                print(f"Original query: {last_analysis['original_query']}")
                print(f"Final query: {last_analysis['final_query']}")
                print(f"Model output: {last_analysis['raw_output']}")
                print(f"Analysis time: {last_analysis['analysis_latency']:.0f}ms")
                if last_analysis['chunks']:
                    print(f"\nRetrieved {len(last_analysis['chunks'])} chunks:")
                    for i, chunk in enumerate(last_analysis['chunks'], 1):
                        print(f"  [{i}] {chunk['date']} - {chunk['source']}")
                        print(f"      {chunk['text'][:80]}...")
                print(f"\nRetrieval time: {last_analysis['retrieval_latency']:.0f}ms")
                print(f"Generation time: {last_analysis['generation_latency']:.0f}ms")
                print(f"Total time: {last_analysis['total_latency']:.0f}ms")
                print("="*70 + "\n")
            else:
                print("\nNo analysis yet.\n")
            continue
        
        # Process query
        total_start = time.time()
        
        # Step 1: Query Analysis
        if rag_enabled:
            should_retrieve, final_query, raw_output, analysis_latency = system.analyze_query(
                user_input, 
                chat_history
            )
            
            print(f"[Analysis: {'RETRIEVE' if should_retrieve else 'NO_RETRIEVE'} | {analysis_latency:.0f}ms]")
            
            if should_retrieve and final_query != user_input:
                print(f"[Rewritten: {final_query}]")
        else:
            should_retrieve = False
            final_query = user_input
            raw_output = "RAG disabled"
            analysis_latency = 0.0
        
        # Step 2: Retrieval (if needed)
        chunks = []
        retrieval_latency = 0.0
        
        if should_retrieve and rag_enabled:
            chunks, retrieval_latency = retriever.retrieve(final_query, n_results=3)
            if chunks:
                print(f"[Retrieved {len(chunks)} chunks | {retrieval_latency:.0f}ms]")
            else:
                print(f"[No chunks found | {retrieval_latency:.0f}ms]")
        
        # Step 3: Generate response
        response, generation_latency = system.generate_response(
            user_input,
            chat_history,
            chunks if chunks else None,
            citations_enabled
        )
        
        total_latency = (time.time() - total_start) * 1000
        
        print(f"Elon: {response}\n")
        
        # Update history
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response})
        chat_history.append({"role": "system", "content": SYSTEM_MSG})
        
        # Keep manageable
        if len(chat_history) > 16:
            chat_history = chat_history[-16:]
        
        # Store analysis
        last_analysis = {
            'original_query': user_input,
            'should_retrieve': should_retrieve,
            'final_query': final_query,
            'raw_output': raw_output,
            'chunks': chunks,
            'analysis_latency': analysis_latency,
            'retrieval_latency': retrieval_latency,
            'generation_latency': generation_latency,
            'total_latency': total_latency
        }
        
        # Log if enabled
        if logging_enabled:
            logs.append({
                'timestamp': datetime.now().isoformat(),
                'query': user_input,
                'analysis': {
                    'should_retrieve': should_retrieve,
                    'rewritten_query': final_query,
                    'raw_output': raw_output
                },
                'retrieved_chunks': len(chunks),
                'response': response,
                'latency': {
                    'analysis_ms': analysis_latency,
                    'retrieval_ms': retrieval_latency,
                    'generation_ms': generation_latency,
                    'total_ms': total_latency
                }
            })

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    print("\n" + "="*70)
    print("PHASE 1 ONLY ELON MUSK CHATBOT (IDENTITY LAYER TEST)")
    print("="*70)
    print("\nThis version uses ONLY the Phase 1 adapter (r=8, identity learning)")
    print("to test identity consistency without pattern refinement.\n")
    
    # Select RAG mode
    print("Use RAG (knowledge base)?")
    print("1. Yes (recommended)")
    print("2. No (personality only)")
    
    rag_choice = input("\nEnter choice [1/2]: ").strip()
    use_rag = rag_choice != "2"
    
    # Select citation mode
    if use_rag:
        print("\nUse citations in responses?")
        print("1. Yes (show sources)")
        print("2. No (natural responses)")
        
        citation_choice = input("\nEnter choice [1/2]: ").strip()
        use_citations = citation_choice != "2"
    else:
        use_citations = False
    
    # Select logging mode
    print("\nEnable conversation logging?")
    print("1. Yes (save to JSON)")
    print("2. No")
    
    logging_choice = input("\nEnter choice [1/2]: ").strip()
    enable_logging = logging_choice != "2"
    
    # Load Retriever
    retriever = Retriever(CHROMA_DB_PATH)
    
    # Load Phase 1 Only System
    phase1_system = Phase1System(BASE_MODEL_PATH, PHASE1_ADAPTER_PATH, SYSTEM_MSG)
    
    print("\n" + "="*70)
    print("âœ… PHASE 1 ONLY SYSTEM READY")
    print("="*70)
    print(f"Architecture: Base model + Phase 1 adapter only")
    print(f"LoRA Configuration: r=8, alpha=16 (identity layer)")
    print(f"Purpose: Testing identity consistency without refinement")
    print(f"RAG: {'Enabled' if use_rag else 'Disabled'}")
    print(f"Citations: {'Enabled' if use_citations else 'Disabled'}")
    print(f"Logging: {'Enabled' if enable_logging else 'Disabled'}")
    print("="*70)
    
    # Start conversation
    start_conversation(
        system=phase1_system,
        retriever=retriever,
        use_rag=use_rag,
        use_citations=use_citations,
        enable_logging=enable_logging
    )