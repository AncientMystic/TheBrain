# TheBrain

TheBrain is a **local-first document intelligence and knowledge extraction engine**.  
It ingests large collections of mixed-format documents (PDF, HTML, Markdown, DOCX, EPUB, RTF, Jupyter Notebook, plain text, source code, and more), extracts structured knowledge using local LLMs and embedding models via **LM Studio**, **Ollama**, **Kobold.cpp**, or any **OpenAI-compatible backend**, and stores it across multiple SQLite databases.

The system includes **graph-based retrieval**, **verification-first reasoning**, **long-term memory**, **learned logic modules**, an **OpenAI-compatible server**, automatic **audit/governance**, optional **deep research mode**, autonomous report generation, **Recoll full-text search integration**, and a **curated verification standard corpus** for truth anchoring and Socratic/PSYOP vetting.

TheBrain now also incorporates **hyperbolic embeddings** (Poincaré ball model) for document and entity representation, **prime-even gated extraction** to intelligently reduce LLM calls, and **gated verification** that learns to trust or skip verification layers.

It also ships a **glassmorphism WebUI** (`python main.py --webui`, or double-click `start-webui.bat` / `start-webui.sh`) with nine tabs covering every feature — guided learning, chat, deep-graph mindmap, Recoll search, audit, server health, deep research, logic & memory, and full configuration.

## Screenshots

<table>
<tr>
<td width="50%" align="center"><img src="https://github.com/AncientMystic/TheBrain/blob/main/Screenshots/guided-learning-tab.JPG?raw=true" alt="Guided Learning tab" width="100%"><br><em>Guided Learning tab — folders, terminal log, per-document progress</em></td>
<td width="50%" align="center"><img src="https://github.com/AncientMystic/TheBrain/blob/main/Screenshots/deep-graph-tab.JPG?raw=true" alt="Deep Graph tab" width="100%"><br><em>Deep Graph tab — mindmap with right-click research actions</em></td>
</tr>
</table>

---

## Table of Contents

- [Screenshots](#screenshots)
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Install Dependencies](#install-dependencies)
  - [Tesseract OCR Setup](#tesseract-ocr-setup)
  - [ONNX NER Model](#onnx-ner-model)
  - [Recoll Setup (Optional)](#recoll-setup-optional)
- [Configuration](#configuration)
  - [Backend Providers](#backend-providers)
  - [LM Studio Endpoints](#lm-studio-endpoints)
  - [Optional Model Roles](#optional-model-roles)
  - [Concurrency and Chunking](#concurrency-and-chunking)
  - [Novelty Gating and Fast Extractor](#novelty-gating-and-fast-extractor)
  - [Performance / Quality Flags](#performance--quality-flags)
  - [Hyperbolic Embeddings & Prime-Even Gate](#hyperbolic-embeddings--prime-even-gate)
  - [Gated Verification](#gated-verification)
  - [Hyperbolic Conversation Summarization & Memory](#hyperbolic-conversation-summarization--memory)
  - [Recoll Settings](#recoll-settings)
  - [Environment Variables](#environment-variables)
- [Usage](#usage)
  - [Initialize Databases](#initialize-databases)
  - [Guided Learning](#guided-learning)
  - [Verified Sources and Admin Claims](#verified-sources-and-admin-claims)
  - [Audit and Standards Comparison](#audit-and-standards-comparison)
  - [Review Contradictions](#review-contradictions)
  - [Verification Facts JSON](#verification-facts-json)
  - [Chat](#chat)
  - [Deep Research Mode](#deep-research-mode)
  - [Recoll-Guided Autonomous Learning](#recoll-guided-autonomous-learning)
  - [Recoll Fast Mode](#recoll-fast-mode)
  - [Build Recoll Index](#build-recoll-index)
- [Server](#server)
- [WebUI](#webui)
- [Logic Learning](#logic-learning)
  - [Socratic/PSYOP Scoring](#socraticpsyop-scoring)
  - [Training Gate Models](#training-gate-models)
  - [Active Learning Review](#active-learning-review)
  - [Memory Consolidation](#memory-consolidation)
  - [Data Audit & Extraction Coverage](#data-audit--extraction-coverage)
  - [Reprocess Deficient Documents](#reprocess-deficient-documents)
  - [Hyperbolic Embedding Migration](#hyperbolic-embedding-migration)
- [Training Models](#training-models)
- [Directory Structure](#directory-structure)
- [Database Schemas](#database-schemas)
- [How It Works](#how-it-works)
  - [Document Processing](#document-processing)
  - [Reasoning and Chat](#reasoning-and-chat)
  - [Deep Research](#deep-research)
  - [Recoll-Guided Learning](#recoll-guided-learning)
  - [Recoll Fast Mode](#recoll-fast-mode)
  - [Standards and Auditing](#standards-and-auditing)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

---

## Features

- **Multi-format ingestion**  
  PDF, HTML, Markdown, DOCX, EPUB, RTF, Jupyter Notebook, plain text, source code, and more.

- **Full-document processing**  
  Every page and every chunk is processed—no truncation or arbitrary limits.

- **OCR fallback for scanned PDFs**  
  PyMuPDF + Tesseract OCR when text extraction fails. OCR batches process pages in parallel to keep memory bounded.

- **Deep LLM knowledge extraction**  
  Extracts atomic facts, typed entities, people, locations, dates, events, discoveries, gems, and relationships with source spans. Includes:
  - Adaptive system prompts based on document type.
  - Stronger source-span enforcement.
  - Relationship type vocabulary.
  - Hierarchical summarization for long documents.
  - Optional fast pre-extraction using ONNX NER, with LLM verification.
  - **Prime-even gated extraction** learns which chunks warrant full LLM extraction, reducing calls without quality loss.

- **Verification-first reasoning**  
  Implements multiple verification layers (SymStep, VeriCoT, FiDeLiS, R-CoT, ARES) with optional adaptive escalation.

- **Gated verification**  
  A learned gate scales the confidence of each verification layer per claim, allowing cheap layers to run first and expensive ones only when needed.

- **Curated truth anchors**  
  Admin claims and verified-folder facts form a trusted reference corpus. New information is compared against these standards, not treated as inherently true or false.

- **Hyperbolic embeddings**  
  Document and entity embeddings are computed in the Poincaré ball model (`core/hyperbolic.py`), capturing hierarchical relationships better than Euclidean space. Document embeddings are the hyperbolic Fréchet mean of chunk embeddings, ensuring full coverage without memory blow.

- **Hyperbolic conversation summarization**  
  Older conversation turns are clustered using hyperbolic geometry, and only representative messages are kept in context. This reduces token usage while preserving thematic continuity.

- **Hyperbolic memory and consolidation**  
  Memories are stored and retrieved in hyperbolic space. A maintenance script (`scripts/consolidate_memories.py`) merges related memories into a single hyperbolic centroid memory, reducing memory clutter.

- **Multi-backend support**  
  Works with LM Studio, Ollama, Kobold.cpp, and any OpenAI-compatible API. Supports API-key authentication and per-backend model configuration.

- **Knowledge graphs**  
  - `hypergraph.db` – intra-document nodes and edges.
  - `external-graph.db` – cross-document global nodes, aliases, edges, keyword-topic relations, co-occurrence, and cross-document links.
  - Precomputed entity-fact index for fast multi-hop expansion.

- **Graph-first chat with adaptive reasoning**  
  Expands from initial facts through related keywords, entities, and graph edges before falling back to chunk retrieval. Includes intent detection, conversation history, memory, and logic-module-informed re-ranking.

- **Memory & logic modules**  
  - `memories.db` – long-term chat memory with optional temporal decay.
  - `logic.db` – learned logic, reasoning patterns, skills, and strategies.

- **Deep Research mode**  
  Autonomous research with recursive subtopic exploration, mindmap construction, and multi-page Markdown report generation.

- **Recoll full-text search integration**  
  Optional wrapper for the Recoll Python API or CLI. Use Recoll as an additional evidence source in chat, deep research, and guided learning.

- **Recoll Fast Mode**  
  Lightweight, keyword-focused learning using contextual previews around search hits.

- **OpenAI-compatible server**  
  Exposes `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/models`, `/v1/reasoning`, `/v1/health`, and `/metrics`. All `/v1/*` endpoints (except health/models) accept optional Bearer-token auth via `SERVER_AUTH_TOKEN`, and CORS origins are configurable instead of wide-open.

- **Glassmorphism WebUI**  
  Start with `python main.py --webui` (or double-click `start-webui.bat` / `start-webui.sh` — the browser opens automatically). Nine tabs cover every feature with plain-language controls: Guided Learning (folders, live terminal log, per-document progress bars), Chat (sessions, verification-first reasoning, `remember:` memories), Deep Graph (mindmap canvas with drag, wheel zoom, click-to-expand, and right-click Deep research / Recoll search), Recoll search, Audit & review queue (safe resolve/dismiss, never auto-deletes verified facts), Server health (endpoint latency, database sizes, metrics), Deep Research (report generation with Markdown/PDF export), Logic & Memory browsers, and a full Config tab documenting all 70+ options with guarded editing.

- **Extraction reliability (no silent loss)**  
  Small-model-safe LLM batching, per-chunk solo retry on empty results, second pass for long thin chunks, novelty safety floor (never drops below 30% / 3 chunks), verbatim facts kept with confidence penalty instead of dropped, lenient triage (`MIN_FACT_CONFIDENCE` 0.3) with the verifier deciding truth, and character-offset source-span validation with sentence fallback.

- **Recall-augmented priority extraction**  
  The fast pass scans existing databases for matching topics, dates, references, and events (single batched embedding fetch, hyperbolic distances only) and flags priority chunks for guaranteed extraction plus full verification escalation — must-extract and must-verify, never must-believe. Alias ambiguity is preserved with collective coherence resolution.

- **Embedding alignment guards (1024-dim mxbai contract)**  
  All embedding endpoints and stored vectors must match `EMBEDDING_DIM` (1024). Startup probes exclude mismatched endpoints with loud warnings, foreign-dimension cache/index rows are quarantined rather than mixed, and disk caches are keyed by model and dimension. Switching embedding models requires a full re-embed migration — never silent mixing.

- **Audit & governance**  
  Automatic cleanup, standards comparison, contradiction detection, confidence scoring, provenance tracking, and review-queue handling for unresolved contradictions.

- **Socratic/PSYOP vetting**  
  Document-level scoring of narrative-manipulation indicators, systemic bias signals, source hierarchy, and Data/Model/Policy classification.

- **Performance optimizations**  
  - LLM extraction cache.
  - Thread-local HTTP sessions.
  - SQLite connection pooling.
  - Batch embedding cache reads/writes.
  - External graph state caching.
  - Vectorized chunk similarity.
  - Batch fact/entity inserts.
  - Precompiled regex patterns.
  - Parallel chunk summarization.
  - Parallel file ingestion (configurable).
  - Exact vector store for fast nearest-neighbour search.
  - Incremental graph cache updates.

- **Enhanced retrieval pipeline**  
  - Multi-stage retrieval with Weighted Reciprocal Rank Fusion (WRRF).
  - Feature-based ranking (`retrieval/features.py`, `retrieval/ranking.py`).
  - Hierarchical datapoint retriever with learned ranking potential.
  - Optional Graph Neural Network (GNN) embeddings for structural retrieval.
  - Optional hyperbolic distance for semantic similarity (`USE_HYPERBOLIC_RETRIEVAL`).

- **Contextual topic shift detection**  
  - LSTM-based model (`core/topic_shift_model.py`) learns conversational topic boundaries.
  - Fallback heuristic available.
  - Training script: `scripts/train_topic_shift.py`.

- **Active learning**  
  - Knowledge gap detection with priority scores.
  - Thompson sampling for gap selection (`learning/active_learner.py`).
  - Integration with Recoll-guided learning.
  - Script `scripts/active_learning_review.py` to flag uncertain chunks for review.

- **Observability**  
  - JSON structured logging (`core/logging_config.py`).
  - Prometheus-style metrics (`core/metrics.py`, `/metrics` endpoint).
  - Retrieval evaluation script (`scripts/evaluate.py`).

---

## Installation

### Prerequisites

- **Python 3.11+**
- **Tesseract OCR** (optional, for scanned PDFs)
- At least one backend:
  - **LM Studio** running locally or on a network, or
  - **Ollama** running locally, or
  - **Kobold.cpp**, or
  - Any **OpenAI-compatible API**, including cloud services
- *(Optional)* **ONNX Runtime** and **Hugging Face Hub** for the fast NER extractor
- *(Optional)* **Recoll** with Python API or `recollq` CLI for full-text search
- *(Optional)* **PyTorch** and **PyTorch Geometric** for GNN training (fallback available)
- *(Optional)* **scikit-learn** for training the verification gate

### Install Dependencies

```bash
git clone https://github.com/yourusername/thebrain.git
cd thebrain
pip install -r requirements.txt
```

If `requirements.txt` is not present, create it with:

```text
PyMuPDF
pytesseract
Pillow
openai
requests
numpy
beautifulsoup4
python-docx
ebooklib
striprtf
nbformat
python-magic
markdown-it-py
pyahocorasick
fuzzywuzzy
python-Levenshtein
fastapi
uvicorn
pydantic
onnxruntime
huggingface_hub
transformers
aiohttp
tqdm
scikit-learn
```

### Tesseract OCR Setup

- **Windows**: download from [GitHub UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH.
- **Linux**: `sudo apt install tesseract-ocr`
- **macOS**: `brew install tesseract`

### ONNX NER Model

The fast extractor can automatically download a pre-trained ONNX NER model from Hugging Face.  
The default repository is `optimum/bert-base-NER` or `Xenova/bert-base-NER` (configured in `config.py`).  
Set `FAST_EXTRACTOR_ENABLED = False` to disable.

### Recoll Setup (Optional)

1. Install Recoll.
2. Ensure `recollq` is available in `PATH`, or install the Recoll Python API.
3. Set `USE_RECOLL = true` in `config.py` or `USE_RECOLL=true` in your environment.
4. Build a Recoll index over your documents.

---

## Configuration

All settings live in `config.py`. Many can be overridden via environment variables.

### Backend Providers

TheBrain supports multiple backends through a unified provider abstraction.

Built-in provider types:

| Backend Type | Description |
|--------------|-------------|
| `lmstudio` | LM Studio OpenAI-compatible API |
| `ollama` | Ollama native API |
| `ollama_openai` | Ollama OpenAI-compatible API |
| `koboldcpp` | Kobold.cpp OpenAI-compatible API |
| `openai_compatible` | Generic OpenAI-compatible API with API key |

Backends are defined in `config.BACKENDS`.

Example for Ollama native API:

```python
BACKENDS = [
    {
        "name": "ollama-local",
        "backend": "ollama",
        "url": "http://localhost:11434",
        "model": "llama3.1",
        "embeddings_model": "nomic-embed-text",
        "api_key": "not-needed",
    }
]
```

Example for OpenAI-compatible cloud API:

```python
BACKENDS = [
    {
        "name": "openai",
        "backend": "openai_compatible",
        "url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "embeddings_model": "text-embedding-3-small",
        "api_key": "your-api-key-here",
    }
]
```

You can also provide backend configuration through environment variables:

```text
BACKEND_TYPE=ollama
BACKEND_URL=http://localhost:11434
BACKEND_MODEL=llama3.1
BACKEND_EMBEDDINGS_MODEL=nomic-embed-text
BACKEND_API_KEY=not-needed
```

For multiple backends, set `BACKEND_CONFIG_JSON` to a JSON array of backend objects.

Example config files are provided in `configs/`.

### LM Studio Endpoints

Legacy LM Studio configuration remains supported:

```python
LM_STUDIO_URL = "http://localhost:1234/v1"
MODEL_NAME = "lfm2.5-vl-3b-absolute-heresy-i1"
EMBEDDING_MODEL = "smcleod/text-embedding-mxbai-embed-large-v1"
```

Additional endpoints may be configured via `LM_STUDIO_URL_2`, `MODEL_NAME_2`, etc.

### Optional Model Roles

Separate models can be configured for extraction, verification, chat, and audit:

```python
SMALL_MODEL_URL = ""
SMALL_MODEL_NAME = ""
LARGE_MODEL_URL = ""
LARGE_MODEL_NAME = ""
AUDIT_MODEL_URL = ""
AUDIT_MODEL_NAME = ""
CHAT_MODEL_URL = ""
CHAT_MODEL_NAME = ""
```

If left empty, the main backend model is used.

### Concurrency and Chunking

```python
LLM_ENDPOINT_CAPACITIES = [3, 1, 2]   # balance these to use all endpoints
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
LLM_BATCH_CHUNKS = 4                  # batch chunks per LLM call (2-4 recommended)
CHUNK_EXTRACTION_WORKERS = 4          # should match sum of capacities
```

**Performance Tip:** If you have multiple endpoints but only one seems busy, set `LLM_ENDPOINT_CAPACITIES` to equal values (e.g., `[1, 1, 1]`) or enable `USE_DYNAMIC_ENDPOINT_BALANCING = True`.

### Novelty Gating and Fast Extractor

```python
NOVELTY_ENABLED = True
NOVELTY_SIM_THRESHOLD = 0.92

FAST_EXTRACTOR_ENABLED = True
FAST_EXTRACTOR_MODEL_NAME = "optimum/bert-base-NER"
FAST_EXTRACTOR_MODEL_DIR = "models/ner_onnx"
FAST_EXTRACTOR_CONFIDENCE_THRESHOLD = 0.7
```

### Performance / Quality Flags

```python
LLM_EXTRACTION_CACHE = True
USE_LLM_HTTP_SESSION = True
FTS_ENABLED = True
EMBEDDING_BATCH_SIZE = 128
PARALLEL_PROCESSING_ENABLED = False
PARALLEL_WORKERS = 2
USE_PROGRESS_BARS = True
ADAPTIVE_VERIFICATION = True
AUTO_RESOLVE_CONTRADICTIONS = False
DEEP_RESEARCH_INTERACTIVE = True
DEEP_RESEARCH_AUTO_SUBTOPIC_DEPTH = 2
OCR_BATCH_SIZE = 64                  # pages per OCR batch (memory-safe)
```

### Hyperbolic Embeddings & Prime-Even Gate

```python
# Enable hyperbolic document embeddings and retrieval
USE_HYPERBOLIC_RETRIEVAL = True

# Enable prime-even gated extraction (reduces LLM calls on redundant chunks)
USE_PRIME_EVEN_GATE = True
```

When `USE_PRIME_EVEN_GATE` is enabled, the system will:
- Compute spectral features for each document.
- Use a learned gate to decide which chunks need full LLM extraction.
- Skip redundant chunks while still using the fast ONNX extractor for entities.

**Training the gate:** Process some documents with `USE_PRIME_EVEN_GATE=true` to collect training data, then run:
```bash
python scripts/train_gate.py
```
The trained gate is saved to `models/gate.json`.

### Gated Verification

```python
USE_GATED_VERIFICATION = True
```

When enabled, a learned verification gate scales the confidence of each verification layer per claim. To train:
1. Enable `USE_GATED_VERIFICATION=true` during processing.
2. Process documents to collect training data.
3. Run `python scripts/train_verification_gate.py`.
4. The gate will be saved to `models/verification_gate.json`.

### Hyperbolic Conversation Summarization & Memory

```python
# Use hyperbolic clustering to summarize older conversation history
USE_HYPERBOLIC_CONVERSATION_SUMMARY = True

# Use hyperbolic memory retrieval
USE_HYPERBOLIC_MEMORY = True

# Similarity threshold for merging memories in consolidation (higher = stricter)
MEMORY_CONSOLIDATION_THRESHOLD = 0.5

# Optional clustering utilities (off by default)
USE_HYPERBOLIC_CLUSTERING = False
```

**How it works:**
- Older conversation turns are embedded into the Poincaré ball and clustered. Only one representative per cluster is kept in context, plus the most recent turns verbatim. This reduces token usage while preserving thematic flow.
- Memories are stored with hyperbolic embeddings (`embedding_space='hyperbolic'`) and retrieved using hyperbolic distance, focusing on the most semantically relevant items.
- `scripts/consolidate_memories.py` merges memories that are closer than the threshold into a single memory with a hyperbolic centroid embedding.

### Recoll Settings

```python
USE_RECOLL = os.environ.get("USE_RECOLL", "false").lower() == "true"
RECOLL_CONFDIR = os.environ.get("RECOLL_CONFDIR", "")
RECOLL_EXTRA_DBS = os.environ.get("RECOLL_EXTRA_DBS", "")
RECOLL_DEFAULT_LIMIT = int(os.environ.get("RECOLL_DEFAULT_LIMIT", "20"))
RECOLL_MAX_ROUNDS = int(os.environ.get("RECOLL_MAX_ROUNDS", "10"))
RECOLL_INTERACTIVE = os.environ.get("RECOLL_INTERACTIVE", "false").lower() == "true"
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_TYPE` | `lmstudio` | Backend type: `lmstudio`, `ollama`, `koboldcpp`, `openai_compatible` |
| `BACKEND_URL` | `http://localhost:1234/v1` | Backend base URL |
| `BACKEND_MODEL` | `lfm2.5-vl-3b-absolute-heresy-i1` | Main LLM model |
| `BACKEND_EMBEDDINGS_MODEL` | `smcleod/text-embedding-mxbai-embed-large-v1` | Embedding model |
| `BACKEND_API_KEY` | `not-needed` | API key for OpenAI-compatible backends |
| `BACKEND_CONFIG_JSON` | empty | JSON array for multiple backends |
| `LM_STUDIO_URL` | `http://localhost:1234/v1` | Legacy primary LM Studio URL |
| `MODEL_NAME` | `lfm2.5-vl-3b-absolute-heresy-i1` | Legacy primary LLM model |
| `EMBEDDING_MODEL` | `smcleod/text-embedding-mxbai-embed-large-v1` | Legacy embedding model |
| `CHUNK_SIZE` | `2000` | Document chunk size |
| `CHUNK_OVERLAP` | `200` | Chunk overlap |
| `EMBEDDING_BATCH_SIZE` | `128` | Embedding batch size |
| `LLM_BATCH_CHUNKS` | `4` | Chunks per LLM extraction call |
| `OCR_BATCH_SIZE` | `64` | Pages per OCR batch |
| `USE_PRIME_EVEN_GATE` | `true` | Enable prime-even gated extraction |
| `USE_HYPERBOLIC_RETRIEVAL` | `true` | Use hyperbolic distance in retrieval |
| `USE_GATED_VERIFICATION` | `false` | Enable learned verification gate |
| `USE_HYPERBOLIC_CONVERSATION_SUMMARY` | `true` | Summarize older conversation history with hyperbolic clustering |
| `USE_HYPERBOLIC_MEMORY` | `true` | Store/retrieve memories in hyperbolic space |
| `MEMORY_CONSOLIDATION_THRESHOLD` | `0.5` | Similarity threshold for merging memories |
| `USE_HYPERBOLIC_CLUSTERING` | `false` | Enable hyperbolic clustering utilities |
| `USE_DYNAMIC_ENDPOINT_BALANCING` | `false` | Dynamically balance endpoint capacities |
| `RECOLL_BIN` | `recollq` | Path to Recoll CLI binary |
| `RECOLL_DB` | empty | Recoll config directory |
| `RECOLL_MAX_RESULTS` | `50` | Default max results for Recoll queries |
| `PREVIEW_CHAR_WINDOW` | `1000` | Character window for fast mode previews |
| `RECOLL_AUTO_KEYWORD_LIMIT` | `20` | Seed keyword count for automatic fast mode |
| `SERVER_HOST` | `127.0.0.1` | Bind address (`0.0.0.0` for LAN, with a token set) |
| `SERVER_PORT` | `8000` | API and WebUI port |
| `SERVER_AUTH_TOKEN` | empty | Bearer token for `/v1/*` and `/api/*` (empty = open) |
| `CORS_ORIGINS` | empty | Allowed browser origins, comma-separated |
| `THEBRAIN_ALLOWED_ROOTS` | empty | Restrict `--input` folders (comma-separated, empty = anywhere) |
| `EMBEDDING_DIM` | `1024` | Contract: all endpoints and stored vectors must match (mxbai) |
| `MIN_FACT_CONFIDENCE` | `0.3` | Lenient triage floor (verifier decides truth) |
| `RECALL_PRIORITY_THRESHOLD` | `0.5` | Score at/above which a chunk is priority |
| `OCR_WORKERS` | CPU count (max 8) | Parallel OCR processes |
| `PARALLEL_WORKERS` | `2` | Parallel file workers |
| `PREFETCH_NEXT_DOCUMENT` | `true` | Overlap next-file prep with current LLM calls |

---

## Usage

### Initialize Databases

```bash
python scripts/init_schemas.py
```

Or simply run any mode. `main.py` initializes automatically.

### Guided Learning

Process documents and build the knowledge graph:

```bash
python main.py --guided-learning --input "A:/pdfs"
```

Optional flags:

- `--debug` — detailed logs.
- `--logic` — use learned logic modules during processing.
- `--dry-run` — do not write to databases (simulate processing).
- `--limit N` — process only the first N files.

### Verified Sources and Admin Claims

TheBrain supports a curated truth-anchor corpus in `verification_standards.db`.

#### Verified Folder Ingestion

Treat an entire input folder as verified reference material:

```bash
python main.py --guided-learning --verified --input "A:/verified_standards"
```

Extracted facts from this folder are marked `verified_true` and inserted into the standards corpus. Documents are promoted only once.

#### Bulk Import Facts and Logic Training Data

The `import_data.py` helper imports pre-generated JSON datasets into TheBrain.

Supported modes:

- `--facts facts.json` — imports verified/admin facts into `verification_standards.db`
- `--logic logic.json` — converts logic-training modules into temporary Markdown and runs the existing logic-learning pipeline
- `--facts facts.json --logic logic.json` — imports both

#### Facts JSON format

```json
{
  "facts": [
    {
      "id": "fact-001",
      "statement": "Water freezes at 0C under 1 atm",
      "subject": "Water",
      "predicate": "freezes_at",
      "object": "0C",
      "negation": 0,
      "truth_status": "admin_claim",
      "source_type": "admin_claim",
      "priority": 0,
      "confidence": 1.0,
      "socratic_metadata": {},
      "supporting_evidence": [],
      "provenance": {}
    }
  ]
}
```

#### Logic JSON format

```json
{
  "logic_modules": [
    {
      "name": "Basic Percentage Calculation",
      "category": "reasoning",
      "summary": "A reusable process for calculating percentages accurately.",
      "keywords": ["percentage", "math", "calculation"],
      "content": "Step 1: Identify the base value. Step 2: Convert the percentage to a decimal. Step 3: Multiply.",
      "examples": [
        {
          "input_text": "What is 25 percent of 80?",
          "output_text": "Convert 25% to 0.25 and multiply 80 by 0.25. Result: 20."
        }
      ]
    }
  ]
}
```

#### Admin Claim Ingestion

Add individual indisputable claims directly:

```bash
python main.py --guided-learning --verified --fact "Water freezes at 0C under 1 atm"
```

Multiple claims are allowed:

```bash
python main.py --guided-learning --verified \
  --fact "The Earth is round" \
  --fact "Water freezes at 0C under 1 atm"
```

Admin claims receive the highest priority and are never demoted by audit.

### Audit and Standards Comparison

Compare ordinary extracted facts against the trusted standard corpus:

```bash
python main.py --audit
```

The audit performs graph cleanup and then attempts to align or dispute unverified facts against standards. Facts are never auto-deleted in normal audit mode.

### Review Contradictions

View unresolved contradictions in the review queue without deleting anything:

```bash
python main.py --review-contradictions
```

### Verification Facts JSON

The standard corpus can be imported and exported as JSON.

#### Import

```bash
python scripts/import_verification_facts.py data/verification_facts.json
```

#### Export

```bash
python scripts/export_verification_facts.py data/verification_facts.json
```

### Chat

```bash
python main.py --chat
```

Enable verification-first reasoning:

```bash
python main.py --chat --reasoning
```

Chat supports conversation history, memory, graph-first retrieval, multi-hop expansion, intent-aware answers, Markdown output, optional Recoll search, and verified chat via GIVE pattern.

**Hyperbolic conversation summarization:** Older turns are automatically summarized via hyperbolic clustering (enabled by default). Recent turns are kept verbatim.

**Hyperbolic memory:** Memories are retrieved using hyperbolic distance, focusing on the most relevant items.

### Deep Research Mode

```bash
python main.py --chat --deep-research
```

Generates a comprehensive Markdown report in `reports/`.

### Recoll-Guided Autonomous Learning

```bash
python main.py --guided-learning --recoll
```

Optional: `--recoll-max-rounds 5 --recoll-interactive`.

### Recoll Fast Mode

Search a specific keyword:

```bash
python main.py --recoll-fast --recoll-query "keyword"
```

Automatic seed-keyword mode: `python main.py --recoll-fast`  
Interactive: `python main.py --recoll-fast --interactive`

### Build Recoll Index

```bash
python main.py --build-recoll-index --input "A:/documents"
```

### Server

```bash
python main.py --server
```

Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/models`, `/v1/reasoning`, `/v1/health`, and `/metrics`.

Set `SERVER_AUTH_TOKEN` to require Bearer-token auth on all `/v1/*` endpoints (except models/health) and `/api/*` (except the health summary). `CORS_ORIGINS` replaces the old allow-all policy — leave it empty for local-only use. The server binds `127.0.0.1` by default; set `SERVER_HOST=0.0.0.0` only for LAN access (with a token set).

### WebUI

No terminal skills needed:

```bash
python main.py --webui
```

Or double-click `start-webui.bat` (Windows) / `start-webui.sh` (Mac/Linux) — the browser opens automatically at `http://localhost:8000/ui`. Leave the window open while you work; add `--no-browser` to skip the auto-open.

Nine tabs, each with plain-language controls, a live terminal log, and progress bars: **Guided** (folders, verified/logic toggles, dry run, per-document rows), **Chat** (sessions, verification-first reasoning, deep research, `remember:` memories), **Graph** (mindmap with drag, wheel zoom, click-to-expand, right-click Deep research / Recoll search, facts viewer, scrolling log), **Recoll** (search, fast preview, binary status), **Audit** (one-click audit run, review queue with safe resolve/dismiss), **Server** (endpoint latency, database sizes, metrics tail), **Research** (report generation with Markdown download and print-to-PDF export), **Logic** (module/memory browsers, folder learning runs, consolidation), **Config** (all 70+ options with explanations, validated saving, secrets masked, model switches gated behind explicit confirm).

### Logic Learning

```bash
python main.py --logic --input "A:/logic_examples"
```

### Socratic/PSYOP Scoring

Verified-folder documents are automatically assessed using the Socratic/PSYOP scorer. The result is attached to standards as provenance metadata.

### Training Gate Models

**Prime-even gate (for extraction):**
1. Enable `USE_PRIME_EVEN_GATE=true` in `config.py` or environment.
2. Process some documents to collect training data.
3. Run `python scripts/train_gate.py`.
4. The gate is saved to `models/gate.json` and will be used automatically when the flag remains enabled.

**Verification gate:**
1. Enable `USE_GATED_VERIFICATION=true`.
2. Process documents to collect verification outcomes.
3. Run `python scripts/train_verification_gate.py` (requires scikit-learn).
4. The gate is saved to `models/verification_gate.json`.

### Active Learning Review

After training a prime-even gate, run:

```bash
python scripts/active_learning_review.py
```

This flags chunks where the gate is uncertain (weight near 0.5), so you can review or reprocess them.

### Memory Consolidation

Merge related hyperbolic memories to reduce clutter:

```bash
python scripts/consolidate_memories.py
```

This uses `MEMORY_CONSOLIDATION_THRESHOLD` (similarity) to decide which memories to merge. The resulting consolidated memories have hyperbolic centroid embeddings.

### Data Audit & Extraction Coverage

Audit extracted knowledge against document chunks to find gaps and malformed entries:

```bash
python scripts/audit_extraction_coverage.py --min-items-per-chunk 0.5 --show-samples
```

Flags:
- `--min-items-per-chunk` – minimum total extracted items per chunk to consider adequate.
- `--show-samples` – display examples of invalid source spans.

### Reprocess Deficient Documents

Re‑extract documents with low fact density using full LLM extraction:

```bash
python scripts/reprocess_deficient.py --min-facts-per-chunk 0.2 --limit 50
```

Flags:
- `--min-facts-per-chunk` – threshold for fact density.
- `--limit` – maximum number of documents to reprocess.
- `--dry-run` – list deficient documents without processing.

### Hyperbolic Embedding Migration

Migrate existing embeddings to hyperbolic space and backfill missing fact embeddings:

```bash
python scripts/migrate_hyperbolic.py
```

---

## Training Models

TheBrain includes several trainable models. Each can be trained after appropriate data has been collected.

### Graph Neural Network (GNN)

Trains GraphSAGE on the external knowledge graph using hyperbolic‑derived node features.

```bash
python scripts/train_gnn.py
```

Outputs:
- `models/gnn/gnn_sage.pt` – trained model weights.
- `models/gnn/node_embeddings.npy` – hyperbolic node embeddings.
- `models/gnn/node_ids.npy` – mapping of node IDs.

Requires PyTorch. The trained model is used when `USE_GNN = True` in `config.py`.

### Prime‑Even Gate

Trains the gate that determines which chunks need full LLM extraction.

1. Enable `USE_PRIME_EVEN_GATE=true` in config.
2. Process documents to collect training data.
3. Run:

```bash
python scripts/train_gate.py
```

The gate is saved to `models/gate.json`.

### Verification Gate

Trains the per‑verifier confidence scaling gate.

1. Enable `USE_GATED_VERIFICATION=true`.
2. Process documents.
3. Run:

```bash
python scripts/train_verification_gate.py
```

The gate is saved to `models/verification_gate.json`.

### Distilled Extractor

Trains a smaller seq2seq model for faster extraction. Requires the distilled training data collected during processing.

```bash
python scripts/train_distilled_extractor.py
```

The model is saved to `models/distilled_extractor/`. Set `USE_DISTILLED_EXTRACTOR=true` in config to use it.

### Topic Shift LSTM

Trains the conversational topic shift detector.

```bash
python scripts/train_topic_shift.py
```

The model is saved to `models/topic_shift/topic_shift_lstm.pt`.

---

## Directory Structure

```
TheBrain/
├── main.py                   # CLI (+ --webui one-click start)
├── start-webui.bat / .sh     # Double-click WebUI launchers
├── server.py                 # OpenAI-compatible API + WebUI mount
├── config.py                 # All settings (env-overridable)
├── webui/                    # Glassmorphism WebUI (9 tabs)
│   ├── app.py                # Mount: /ui + /api/* with shared auth
│   ├── schema.py             # Config introspection (73 options + docs)
│   ├── jobs.py               # Guided/audit/research/logic workers + SSE
│   ├── *_api.py              # Per-tab backends reusing CLI functions
│   └── static/index.html     # Single-file UI (no build step)
├── Screenshots/              # UI screenshots (see top of readme)
├── docs/                     # Model card + dataset datasheet templates
├── audit/
├── chat/
│   ├── give_chat.py          # GIVE-pattern verified chat
│   ├── conversation_summarizer.py  # Hyperbolic conversation summarizer
│   └── ...
├── configs/                  # Example backend configs
├── core/
│   ├── backends/             # Backend provider abstraction
│   ├── hyperbolic.py         # Poincaré ball geometry, Fréchet mean
│   ├── hyperbolic_utils.py   # Safe mapping, similarity, interpolation
│   ├── hyperbolic_clustering.py  # Clustering utilities (opt-in)
│   ├── shape_constraints.py  # PAVA / Edgeworth / trapezoid / unimodal / dominance
│   ├── klein.py / dykstra.py # Klein maps + family-block projections
│   ├── regime_audit.py       # Marginal rates, co-occurrence, saturation
│   ├── provenance.py         # Run ledger with hashes + replay checks
│   ├── entity_linking.py     # Collective coherence resolution
│   ├── span_validation.py    # Offset checks with sentence fallback
│   ├── vector_store.py       # Ball-tree index with dim-keyed cache
│   ├── recoll_client.py      # Validated recollq wrapper
│   ├── spectral.py           # Spectral feature extraction
│   ├── metrics.py            # Metrics registry
│   ├── logging_config.py     # JSON logging setup
│   ├── topic_shift_model.py  # LSTM topic shift detector
│   └── ...
├── deep_research/
├── extraction/
│   ├── gate.py               # Prime-even gated extraction
│   ├── recall_index.py       # DB-aware recall preload (disk-cached)
│   ├── recall_augmenter.py   # Priority scoring with batched coherence
│   └── ...
├── extractors/
├── fast_extractor/
├── graph/
│   ├── gnn_sage.py           # GraphSAGE model
│   └── ...
├── ingestion/
├── learning/
│   └── active_learner.py     # Thompson sampling for gaps
├── logic/
├── memory/
│   ├── hyperbolic_memory.py  # Hyperbolic memory storage/retrieval
│   └── ...
├── reasoning/
│   ├── verification_manager.py
│   ├── verification_gate.py  # Learned verification gate
│   ├── semantic_contradiction.py
│   └── ...
├── scripts/
│   ├── train_gate.py
│   ├── tune_gate_weights.py  # 5-fold CV grid + fingerprint stats
│   ├── verify_run.py         # 5-step provenance replay checks
│   ├── audit_regimes.py      # Regime-cube audit runner
│   ├── synthetic_spectral.py # Deterministic regime generator
│   ├── train_verification_gate.py
│   ├── train_topic_shift.py
│   ├── train_gnn.py
│   ├── evaluate.py
│   ├── active_learning_review.py
│   ├── consolidate_memories.py
│   ├── migrate_hyperbolic.py
│   └── ...
├── models/                   # ONNX NER model, GNN, topic shift, gates
├── reports/                  # generated research reports
├── data/                     # SQLite databases and verification standards
└── gazetteers/
```

---

## Database Schemas

TheBrain uses multiple SQLite databases:

| Database | Tables |
|----------|--------|
| `data/index.db` | documents, document_chunks, processing_progress, llm_extraction_cache |
| `data/summaries.db` | doc_summaries, summary_versions |
| `data/key_facts.db` | key_facts, entities, people, locations, dates, events, discoveries, gems, fact_sources, entity_fact_index, key_facts_fts, gate_training_data |
| `data/embeddings.db` | document_embeddings, chunk_embeddings, embedding_cache |
| `data/hypergraph.db` | nodes, edges, doc_entity_nodes |
| `data/external-graph.db` | global_nodes, global_edges, topic_nodes, keyword_topic_edges, keyword_cooccurrence, cross_doc_links, topic_hierarchy, global_nodes_fts |
| `data/ocr_cache.db` | ocr_cache |
| `data/memories.db` | memory_entries (with `embedding_space`), memory_keywords, memory_sessions (with `topic_centroid`), conversation_history |
| `data/logic.db` | logic_modules, logic_examples, logic_keywords, logic_tags |
| `data/reasoning.db` | reasoning_nodes, reasoning_edges, grounding_records, kg_triples, reasoning_paths, reasoning_dependencies, verification_results, contradiction_log, agent_actions, research_nodes, research_edges, implied_triples, verification_gate_training_data |
| `data/recoll_log.db` | recoll_queries, recoll_query_results, recoll_log |
| `data/verification_standards.db` | verified_standards, verified_standard_sources, standard_comparisons, verification_promotions |

All schemas are created automatically on first run.

---

## How It Works

### Document Processing

1. Scan input directory.
2. Hash each file and check processing status.
3. Extract full text using the appropriate extractor.
4. Chunk text with overlap.
5. Generate chunk embeddings and compute document embedding as hyperbolic Fréchet mean.
6. Optional fast extractor pre-pass.
7. Novelty gate determines which chunks need LLM extraction.
8. Prime-even gate (if enabled) further filters chunks for full LLM extraction.
9. LLM extracts facts, entities, relationships, etc.
10. Validate, deduplicate, and verify facts using VerificationManager (SymStep, VeriCoT, R-CoT, ARES).
11. Build hypergraph and external graph.
12. Generate hierarchical summary and mark processed.
13. Files can be processed in parallel (configurable).

### Reasoning and Chat

1. Decompose query into sub-questions.
2. Detect intent.
3. Retrieve initial facts via multi-stage retrieval (graph, vector, lexical, GNN).
4. Fuse results using Weighted Reciprocal Rank Fusion (WRRF).
5. Expand through graph and entity index.
6. Check sufficiency with LLM.
7. Verify claims with SymStep, VeriCoT, FiDeLiS, R-CoT, ARES.
8. Re-rank using memory and logic modules.
9. Synthesize final answer with provenance (GIVE pattern if enabled).
10. Conversation history is summarized using hyperbolic clustering (older turns) to save context.

### Deep Research

1. Start from a main query.
2. Build mindmap.
3. Discover subtopics via LLM and graph analysis.
4. Optionally ask user which subtopics to pursue.
5. Generate detailed reports.
6. Save to `reports/`.

### Recoll-Guided Learning

1. Analyze internal databases for knowledge gaps.
2. Generate Recoll queries from gaps.
3. Search the Recoll index.
4. Process retrieved documents.
5. Log queries and processed documents.
6. Active learning can prioritize gaps using Thompson sampling.

### Recoll Fast Mode

1. Select keyword.
2. Run Recoll query.
3. Extract contextual preview.
4. Store as pseudo-document.
5. Generate embeddings and run extraction.
6. Store extracted facts and source references.

### Standards and Auditing

1. Admin claims and verified-folder facts populate `verification_standards.db`.
2. Socratic/PSYOP scoring attaches provenance metadata to standards.
3. Audit compares unverified facts against standards using exact, embedding, and optional LLM methods.
4. Semantic contradiction detection finds conflicts missed by exact matching.
5. Results are stored in `standard_comparisons`.
6. Admin and verified facts are never auto-deleted or demoted.

---

## Limitations

- Some reasoning components are heuristic rather than formal solvers.
- Contradiction auto-resolution is disabled by default; unresolved conflicts move to review.
- SQLite may struggle with very large graphs.
- JSON parsing may occasionally fail depending on the local LLM.
- No web UI; server API and CLI only.
- Not battle-tested on very large heterogeneous corpora.
- Socratic/PSYOP scoring quality depends on the selected LLM.
- Backend-specific behavior, especially embedding formats, may vary between providers.
- GNN training requires PyTorch and may be resource-intensive for very large graphs.

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## License

[MIT](LICENSE)

---

## Disclaimer

This project is under active development. Some reasoning components may be heuristic and may require tuning for production use. This is a work in progress and is considered highly experimental.

---

## Thank You

A special thank you to **[SocioProphet](https://github.com/SocioProphet)** for the advanced concepts, review report, and technical inspiration that helped shape this project.
