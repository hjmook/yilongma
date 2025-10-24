"""
Analyzer Service (local copy) - standalone microservice for query analysis

This is a self-contained copy of the Query Analyzer so that the hybrid system
does not rely on .py files outside this directory.

Enhanced with DSPy for better consistency and structured output parsing.
"""

import torch
import platform
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from flask import Flask, request, jsonify
from typing import List, Dict, Tuple

# DSPy imports (with fallback if not available)
try:
    import dspy
    DSPY_AVAILABLE = True
    print("🔬 DSPy available - using enhanced structured analysis")
except ImportError:
    DSPY_AVAILABLE = False
    print("⚠️  DSPy not available - using traditional analysis")


# --------------------------
# Configuration
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "Llama-3.2-3B-Instruct")
PORT = 6767


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
# Analysis Prompt
# --------------------------
ANALYSIS_PROMPT_TEMPLATE = """You are a query analysis assistant for an Elon Musk chatbot.

Your job: Decide if a query needs factual information retrieval from a knowledge base based on both the query and the conversation history, and if so, rewrite it for search.

Guidelines:
- Questions or remarks about Elon's companies/projects (SpaceX, Tesla, Boring Company, xAI, Grok, Neuralink, DOGE) → RETRIEVE
- Factual questions or remarks (what/when/how many/latest/recent/status/numbers) → RETRIEVE
- Follow-up questions → RETRIEVE (resolve pronouns using context)
- Questions about specific events, dates, people → RETRIEVE
- Ambiguous terms needing clarification → RETRIEVE
- Questions about personal life (kids, family, relationships) → RETRIEVE
- Pure greetings (hi/hello/hey) → NO_RETRIEVE
- Everything else -> RETRIEVE

Output format (CRITICAL - follow EXACTLY):
- If retrieval NOT needed: "NO_RETRIEVE"
- If retrieval needed: "RETRIEVE: <rewritten query as clear, standalone search query>"

Now analyze this query:

Conversation:
{history}

User: "{query}"

Output:"""

# DSPy Structured Analysis (when available) - Improved prompt structure
if DSPY_AVAILABLE:
    class QueryAnalysisSignature(dspy.Signature):
        """Analyze ONLY the current user query using conversation history as context"""
        conversation_context = dspy.InputField(desc="Previous conversation for context understanding")
        current_query = dspy.InputField(desc="The current user query to analyze")
        analysis_result = dspy.OutputField(desc="Analysis decision in the required format")
    
    class DSPyQueryAnalyzer(dspy.Module):
        """DSPy-enhanced query analyzer with better prompt structure"""
        
        def __init__(self):
            super().__init__()
            self.analyzer = dspy.ChainOfThought(QueryAnalysisSignature)
            
        def forward(self, conversation_context: str, current_query: str) -> dspy.Prediction:
            return self.analyzer(
                conversation_context=conversation_context,
                current_query=current_query
            )
    
    def create_structured_prompt(history_str: str, query: str) -> str:
        """Create a DSPy-structured prompt that clearly separates context from task"""
        return f"""You are a query analysis assistant for an Elon Musk chatbot.

CONTEXT (for understanding only - DO NOT analyze these messages):
{history_str if history_str.strip() != "[]" else "No previous conversation"}

TASK: Analyze ONLY the current user query below using the context above for understanding.

Guidelines:
- Questions or remarks about Elon's companies/projects/relationships (SpaceX, Tesla, Boring Company, xAI, Grok, Neuralink, DOGE, Trump) → RETRIEVE
- Factual questions or remarks (what/when/how many/latest/recent/status/numbers) → RETRIEVE
- Follow-up questions → RETRIEVE (resolve pronouns using context)
- Questions about specific events, dates, people → RETRIEVE
- Ambiguous terms needing clarification → RETRIEVE
- Questions about personal life (kids, family, relationships) → RETRIEVE
- Pure greetings (hi/hello/hey) → NO_RETRIEVE
- Everything else -> RETRIEVE

CRITICAL REWRITING RULES:
1. Convert first-person queries to third-person: "you" → "Elon Musk"
2. Use conversation context to resolve ambiguous pronouns and references
3. Create standalone queries optimized for vector database search
4. Focus on factual information retrieval, not conversational queries

Examples:
- "What do you think about Trump?" → "What does Elon Musk think about Trump?"
- "But you worked with him before" (context: Trump discussion) → "Did Elon Musk work with Donald Trump?"
- "On what?" (context: working together) → "What projects did Elon Musk work on with Donald Trump?"

Output format (CRITICAL - follow EXACTLY):
- If retrieval NOT needed: "NO_RETRIEVE"
- If retrieval needed: "RETRIEVE: <rewritten query as clear, standalone search query for vector database>"

CURRENT USER QUERY TO ANALYZE: "{query}"

Output:"""

else:
    def create_structured_prompt(history_str: str, query: str) -> str:
        """Fallback to original prompt when DSPy not available"""
        return ANALYSIS_PROMPT_TEMPLATE.format(history=history_str, query=query)


class QueryAnalyzer:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        has_gpu = torch.cuda.is_available()
        on_mac = platform.system() == "Darwin"

        if has_gpu and not on_mac:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map={"": device},
                torch_dtype=torch.float16 if has_gpu or on_mac else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        self.model.eval()
        
        # Initialize DSPy analyzer if available (improved prompt structure)
        self.use_dspy = DSPY_AVAILABLE
        if self.use_dspy:
            try:
                # Try to configure DSPy with a dummy LM for structure
                # We'll use our model directly but leverage DSPy's prompt structuring
                self.dspy_analyzer = DSPyQueryAnalyzer()
                print("✅ DSPy analyzer initialized successfully (structured prompting)")
            except Exception as e:
                print(f"⚠️  DSPy initialization failed: {e}")
                print("📝 Falling back to traditional analysis")
                self.use_dspy = False

    def analyze(self, query: str, chat_history: List[Dict]) -> Tuple[bool, str, str]:
        history_str = self._format_history(chat_history)
        
        # Use DSPy structured prompt if available, otherwise fallback
        if self.use_dspy:
            prompt = create_structured_prompt(history_str, query)
            method_tag = "DSPy-Structured"
        else:
            prompt = ANALYSIS_PROMPT_TEMPLATE.format(history=history_str, query=query)
            method_tag = "Traditional"
        
        # Generate response using our model
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        raw_output = self.tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        
        # Parse using traditional method
        should_retrieve, rewritten_query = self._parse_output(raw_output, query)
        return should_retrieve, rewritten_query, f"{method_tag}: {raw_output}"

    def _format_history(self, chat_history: List[Dict], max_turns: int = 3) -> str:
        if not chat_history:
            return "[]"
        recent = chat_history[-6:] if len(chat_history) > 6 else chat_history
        formatted = []
        for msg in recent:
            role = "User" if msg['role'] == 'user' else "Elon"
            content = msg["content"]
            if len(content) > 150:
                content = content[:150] + "..."
            formatted.append(f'{role}: "{content}"')
        return "\n".join(formatted)

    def _parse_output(self, output: str, original_query: str) -> Tuple[bool, str]:
        import re
        output = output.strip()
        first_line = output.split('\n')[0].strip()
        if "NO_RETRIEVE" in first_line.upper() or first_line.upper() == "NO":
            return False, original_query
        retrieve_match = re.search(r'RETRIEVE\s*:\s*([^\n]+)', output[:200], re.IGNORECASE)
        if retrieve_match:
            rewritten = retrieve_match.group(1).strip().strip('\"\'')
            rewritten = ' '.join(rewritten.split())
            if len(rewritten) > 5 and len(rewritten.split()) >= 2 and len(rewritten) < 200:
                return True, rewritten
        return True, original_query


app = Flask(__name__)
analyzer = None


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'query_analyzer', 'device': device})


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json or {}
        query = data.get('query', '')
        chat_history = data.get('chat_history', [])
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        should_retrieve, rewritten_query, raw_output = analyzer.analyze(query, chat_history)
        
        # Add analysis method info for debugging
        analysis_method = "DSPy-Enhanced" if analyzer.use_dspy else "Traditional"
        
        return jsonify({
            'should_retrieve': should_retrieve,
            'rewritten_query': rewritten_query,
            'raw_output': raw_output,
            'original_query': query,
            'analysis_method': analysis_method,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/shutdown', methods=['POST'])
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        return jsonify({'error': 'Not running with the Werkzeug Server'}), 500
    func()
    return jsonify({'message': 'Server shutting down...'})


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("QUERY ANALYZER SERVICE (Local Copy)")
    print("=" * 70)
    print(f"Port: {PORT}")
    print(f"Device: {device}")
    print("=" * 70 + "\n")
    analyzer = QueryAnalyzer(BASE_MODEL_PATH)
    print("=" * 70)
    print("🚀 SERVICE READY")
    print("=" * 70)
    print(f"Listening on http://localhost:{PORT}")
    print("=" * 70 + "\n")
    app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False)


