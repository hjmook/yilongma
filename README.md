# Dual-Phase LoRA Fine-Tuning with RAG for Persona-Driven Chatbot

## Project Overview

This project implements a conversational chatbot trained to emulate Elon Musk through a two-phase parameter-efficient fine-tuning approach combined with a Retrieval-Augmented Generation (RAG) system. The goal is to create a bot that captures both factual knowledge and conversational style while handling knowledge conflicts between training data and real-time information.

**Note:** This is an exploratory project with unvalidated hypotheses. Systematic evaluation and comparisons are ongoing work.

---

## Data Preparation

### Data Source
- **2,883 instruction-response pairs** extracted from interview transcripts and podcast audio
- Audio-to-text conversion: Whisper + NeMo for transcription and speaker diarization
- Source: Multiple Elon Musk interviews, podcasts, and forum discussions

### Data Transformation Pipeline

1. **Audio Processing:** Raw interview audio files → transcripts with speaker labels via Whisper + NeMo
2. **Pair Extraction:** Transcripts → sliding window contexts (window size: 10 turns)
   - **Instruction:** Conversation history (context from previous turns)
   - **Response:** Elon's reply to the last remark/question
3. **Standardization:** All pairs converted to `{"instruction": [...], "response": "..."}` format

### Data Characteristics
- Uneven distribution: Some interviews contributed more pairs than others
- Limited coverage: Some facts about Elon appear in too few examples to be reliably learned
- This limitation motivated the addition of RAG to supplement outdated or sparse knowledge

---

## Fine-Tuning Approach

### Architecture
- **Base Model:** Llama-3.2-3B-Instruct
- **Adaptation Method:** Low-Rank Adaptation (LoRA) applied to query, key, and value projections
- **Training Framework:** 4-bit quantization (BitsAndBytes) with gradient checkpointing for memory efficiency

### Dual-Phase Strategy

The training is split into two phases to separately target identity learning and conversational essence:

#### Phase 1: Identity Learning (5 epochs)
- **Goal:** Train the model to identify as Elon Musk with explicit persona markers
- **LoRA Config:** r=8, α=16, dropout=0.05
- **Identity Injection Rate:** 80% of training examples receive identity prefixes ("Elon, ...", "Hey Elon, ...")
- **Learning Rate:** 1e-4
- **Regularization:** weight_decay=0.05

**Hypothesis:** High injection of identity markers in early training helps the model establish a strong persona foundation.

#### Phase 2: Quality Enhancement (3 epochs)
- **Goal:** Train abstract patterns—speech style, reasoning patterns, conversational flow
- **LoRA Config:** r=32, α=64, dropout=0.1
- **Identity Injection Rate:** 20% (lower to focus on style rather than identity markers)
- **Learning Rate:** 5e-5
- **Built on:** Phase 1 adapters frozen and loaded; Phase 2 adapter stacked on top

**Hypothesis:** After identity is established, training on style without heavy persona injection captures the "essence" of conversational patterns.

### Training Details
- **Batch Size:** 2 (per device), gradient accumulation: 4 steps
- **Max Sequence Length:** 4,096 tokens
- **Label Masking:** Only the final response tokens are trained on; conversation history is masked (-100 labels)
- **Validation:** 15% held-out set, evaluated per epoch
- **Regularization:** weight_decay applied to prevent overfitting given the small dataset

### Outcomes
- **Phase 1:** Train Loss 12.16 → Eval Loss 2.41
- **Phase 2:** Train Loss 9.60 → Eval Loss 2.40
- Fine-tuning-only experiments showed overfitting (validation loss divergence + nonsensical responses) when aggressive hyperparameters were applied to the 2,883-pair dataset

---

## Retrieval-Augmented Generation (RAG)

### Motivation: Knowledge Conflict Problem
The conversational dataset contains outdated or incomplete information about Elon Musk's activities. A pure fine-tuned model would:
1. Hallucinate or provide outdated facts
2. Lack knowledge of recent developments in Tesla, SpaceX, etc.

RAG mitigates this by retrieving recent, verified information to ground responses.

### System Architecture

```
User Query
    ↓
[Query Analyzer] (Base Model)
    ├─ Decision: Retrieve or not?
    └─ If yes: Rewrite query for retrieval
         ↓
    [ChromaDB Retriever]
         ├─ Semantic search on query
         └─ Rerank by recency
         ↓
    [Context-Aware Response Generator] (Fine-tuned Model)
         ├─ Input: Query + Retrieved Context + Chat History
         └─ Output: Response
    ├─ No retrieval
         ↓
    [Response Generator] (Fine-tuned Model)
         ├─ Input: Query + Chat History only
         └─ Output: Response
```

### Components

#### 1. Query Analyzer (Base Model)
- Uses the **unmodified base model** (no fine-tuning) to make binary decisions
- **Decision Logic:**
  - **RETRIEVE:** Factual questions or remarks (what/when/how many), questions about companies (Tesla, SpaceX, DOGE), events
  - **NO_RETRIEVE:** Greetings, personal philosophy
- **Output:** Binary flag + optionally rewritten query for better retrieval

**Rationale:** The base model is unbiased and makes decisions based on semantic understanding rather than learned biases from fine-tuning.

#### 2. Knowledge Base (ChromaDB)
- **Data Sources:**
  - News API feeds (Tesla, SpaceX updates)
  - Recent news articles about Elon Musk
  - Verified timelines of events and announcements
- **Retrieval Method:** Semantic search (embeddings) with k=3 top results
- **Reranking:** Results weighted by recency (penalizing older documents)

#### 3. Context-Aware Response Generation
- Retrieved chunks are formatted and provided to the fine-tuned model during generation
- **Knowledge Conflict Handling:** Prompt engineering instructs the model to prioritize retrieved context over potentially outdated trained knowledge
  - System message contains explicit instructions: "If your memory conflicts with the provided context, trust the context"
  - This mechanism is designed to override the model's learned but outdated facts

---

## Model Stacking (Non-Merged Adapters)

Rather than merging Phase 1 and Phase 2 LoRA weights back into the base model, both adapters are kept separate and loaded together:

```
Base Model (Llama-3.2-3B, frozen)
    ↓
Phase 1 LoRA Adapter (identity learning, frozen during Phase 2)
    ↓
Phase 2 LoRA Adapter (quality enhancement, trainable)
```

**Benefits:**
- Reduced memory footprint (two small adapters instead of merged weights)
- Maintains computational efficiency during inference

---

## Inference Setup

### Hyperparameters
- **Max New Tokens:** 200
- **Temperature:** 0.7
- **Top-p:** 0.9, **Top-k:** 50
- **Repetition Penalty:** 1.1
- **Sampling:** Enabled for diversity

### Hardware
- 4-bit quantization reduces model size for deployment considerations
- Mixed precision (FP16/BF16) during generation for speed

---

## Limitations & Current State

1. **No Systematic Validation**
   - Dual-phase separation (identity vs. style) is a hypothesis based on loss curves and qualitative chat outputs, not rigorously validated
   - No A/B comparison between single-phase and dual-phase training
   - No quantitative metrics for response quality

2. **RAG Effectiveness**
   - Knowledge conflict resolution tested qualitatively only
   - Model sometimes contradicts retrieved facts despite prompt engineering
   - No systematic measurement of when/why retrieval helps vs. hurts

3. **Dataset Limitations**
   - 2,883 pairs is small for full model fine-tuning
   - Uneven distribution across sources
   - No test set for final evaluation

4. **No Baseline Comparisons**
   - Have not compared: fine-tuned only vs. RAG only vs. dual system
   - Cannot quantify the added value of combining both approaches

---

## Future Work

1. **Systematic Evaluation**
   - Implement metrics for response factuality, coherence, and persona consistency
   - A/B test dual-phase vs. single-phase training
   - Evaluate RAG contribution independently

2. **Knowledge Conflict Resolution**
   - Experiment with different prompt strategies to better override learned knowledge
   - Test alternative architectural choices (e.g., placing retrieved context in different positions)
   - Measure success rate of fact correction via retrieval

3. **Dataset Expansion**
   - Collect more interviews to reduce overfitting and improve coverage
   - Ensure balanced distribution across sources and topics

4. **Model Architecture**
   - Explore alternative adapter methods (DoRA, prefix tuning) for comparison
   - Investigate multi-adapter composition for more granular persona control

## References & Inspiration

This work was informed by recent research on persona-driven language models:
- **CharacterLLM: A Trainable Agent for Role-Playing** (Zhou et al., 2024) — Specialized architectures for character consistency
- **Persona Hub: Efficient LLM Personalization** (Kim et al., 2024) — Identity-first training strategies for small datasets
- **DoRA: Weight-Decomposed Low-Rank Adaptation** (Liu et al., 2024) — Alternative adapter methods for persona tasks

The dual-phase approach draws motivation from these works but is not a direct implementation of their methods.

---

## Author Notes

This project is exploratory in nature. The hypotheses about what each training phase learns have not been rigorously validated. As the project continues, systematic evaluation will clarify which design choices are effective and which require iteration.
