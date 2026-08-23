# TheBrain

TheBrain is a **local-first document intelligence and knowledge extraction engine**.  
It ingests large collections of mixed-format documents (PDF, HTML, Markdown, DOCX, EPUB, RTF, Jupyter Notebook, plain text, source code, and more), extracts structured knowledge using local LLMs and embedding models via **LM Studio**, **Ollama**, **Kobold.cpp**, or any **OpenAI-compatible backend**, and stores it across multiple SQLite databases.

The system includes **graph-based retrieval**, **verification-first reasoning**, **long-term memory**, **learned logic modules**, an **OpenAI-compatible server**, automatic **audit/governance**, optional **deep research mode**, autonomous report generation, **Recoll full-text search integration**, and a **curated verification standard corpus** for truth anchoring and Socratic/PSYOP vetting.

---

## Table of Contents

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
  - [Logic Learning](#logic-learning)
  - [Socratic/PSYOP Scoring](#socraticpsyop-scoring)
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
  PyMuPDF + Tesseract OCR when text extraction fails.

- **Deep LLM knowledge extraction**  
  Extracts atomic facts, typed entities, people, locations, dates, events, discoveries, gems, and relationships with source spans. Includes:
  - Adaptive system prompts based on document type.
  - Stronger source-span enforcement.
  - Relationship type vocabulary.
  - Hierarchical summarization for long documents.
  - Optional fast pre-extraction using ONNX NER, with LLM verification.

- **Verification-first reasoning**  
  Implements multiple verification layers (SymStep, VeriCoT, FiDeLiS, R-CoT, ARES) with optional adaptive escalation.

- **Curated truth anchors**  
  Admin claims and verified-folder facts form a trusted reference corpus. New information is compared against these standards, not treated as inherently true or false.

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
  Exposes `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/models`, `/v1/reasoning`, and `/v1/health`.

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
LLM_ENDPOINT_CAPACITIES = [3, 1, 2]
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
LLM_BATCH_CHUNKS = 1
CHUNK_EXTRACTION_WORKERS = 3
```

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
EMBEDDING_BATCH_SIZE = 32
PARALLEL_PROCESSING_ENABLED = False
PARALLEL_WORKERS = 2
USE_PROGRESS_BARS = True
ADAPTIVE_VERIFICATION = True
AUTO_RESOLVE_CONTRADICTIONS = False
DEEP_RESEARCH_INTERACTIVE = True
DEEP_RESEARCH_AUTO_SUBTOPIC_DEPTH = 2
```

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
| `EMBEDDING_BATCH_SIZE` | `32` | Embedding batch size |
| `LLM_BATCH_CHUNKS` | `1` | Chunks per LLM extraction call |
| `RECOLL_BIN` | `recollq` | Path to Recoll CLI binary |
| `RECOLL_DB` | empty | Recoll config directory |
| `RECOLL_MAX_RESULTS` | `50` | Default max results for Recoll queries |
| `PREVIEW_CHAR_WINDOW` | `1000` | Character window for fast mode previews |
| `RECOLL_AUTO_KEYWORD_LIMIT` | `20` | Seed keyword count for automatic fast mode |

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

### Verified Sources and Admin Claims

TheBrain supports a curated truth-anchor corpus in `verification_standards.db`.

#### Verified Folder Ingestion

Treat an entire input folder as verified reference material:

```bash
python main.py --guided-learning --verified --input "A:/verified_standards"
```

Extracted facts from this folder are marked `verified_true` and inserted into the standards corpus. Documents are promoted only once.

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

Example JSON structure:

```json
{
  "version": 1,
  "facts": [
    {
      "id": "admin-water-freeze",
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

Use `--debug` for detailed logs.

Chat supports:

- Conversation history.
- Memory storage via `remember: <content>`.
- Graph-first retrieval and multi-hop expansion.
- Intent-aware answer generation.
- Markdown output with source citations.
- Optional Recoll full-text search.

### Deep Research Mode

```bash
python main.py --chat --deep-research
```

Generates a comprehensive Markdown report in `reports/`.

### Recoll-Guided Autonomous Learning

```bash
python main.py --guided-learning --recoll
```

Optional:

```bash
python main.py --guided-learning --recoll --recoll-max-rounds 5 --recoll-interactive
```

### Recoll Fast Mode

Search a specific keyword:

```bash
python main.py --recoll-fast --recoll-query "keyword"
```

Automatic seed-keyword mode:

```bash
python main.py --recoll-fast
```

Interactive mode:

```bash
python main.py --recoll-fast --interactive
```

Options:

- `--recoll-query "keyword"`
- `--recoll-limit 50`
- `--preview-chars 1000`

### Build Recoll Index

```bash
python main.py --build-recoll-index --input "A:/documents"
```

### Server

```bash
python main.py --server
```

Endpoints:

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/responses`
- `POST /v1/embeddings`
- `GET /v1/models`
- `POST /v1/reasoning`
- `GET /v1/health`

### Logic Learning

```bash
python main.py --logic --input "A:/logic_examples"
```

### Socratic/PSYOP Scoring

Verified-folder documents are automatically assessed using the Socratic/PSYOP scorer in `reasoning/socratic_scorer.py`. The scorer evaluates:

- 20 PSYOP narrative-manipulation criteria
- Enforcement-vector detection
- Triad of Intentionality
- Data/Model/Policy classification
- Funding and gatekeeping flags
- Lived-experience cluster detection
- Source hierarchy level

The result is attached to standards as provenance metadata. It does not override admin or verified truth anchors.

---

## Directory Structure

```
TheBrain/
├── main.py
├── server.py
├── config.py
├── audit/
├── chat/
├── configs/              # Example backend configs
├── core/
│   └── backends/         # Backend provider abstraction
├── deep_research/
├── extraction/
├── extractors/
├── fast_extractor/
├── graph/
├── ingestion/
├── logic/
├── memory/
├── reasoning/
├── scripts/
├── models/               # ONNX NER model
├── reports/              # generated research reports
├── data/                 # SQLite databases and verification standards
└── gazetteers/
```

---

## Database Schemas

TheBrain uses multiple SQLite databases:

| Database | Tables |
|----------|--------|
| `data/index.db` | documents, document_chunks, processing_progress, llm_extraction_cache |
| `data/summaries.db` | doc_summaries, summary_versions |
| `data/key_facts.db` | key_facts, entities, people, locations, dates, events, discoveries, gems, fact_sources, entity_fact_index, key_facts_fts |
| `data/embeddings.db` | document_embeddings, chunk_embeddings, embedding_cache |
| `data/hypergraph.db` | nodes, edges, doc_entity_nodes |
| `data/external-graph.db` | global_nodes, global_edges, topic_nodes, keyword_topic_edges, keyword_cooccurrence, cross_doc_links, topic_hierarchy, global_nodes_fts |
| `data/ocr_cache.db` | ocr_cache |
| `data/memories.db` | memory_entries, memory_keywords, memory_sessions, conversation_history |
| `data/logic.db` | logic_modules, logic_examples, logic_keywords, logic_tags |
| `data/reasoning.db` | reasoning_nodes, reasoning_edges, grounding_records, kg_triples, reasoning_paths, reasoning_dependencies, verification_results, contradiction_log, agent_actions, research_nodes, research_edges, implied_triples |
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
5. Generate document and chunk embeddings.
6. Optional fast extractor pre-pass.
7. Novelty gate determines which chunks need LLM extraction.
8. LLM extracts facts, entities, relationships, etc.
9. Validate, deduplicate, and store facts.
10. Build hypergraph and external graph.
11. Generate hierarchical summary and mark processed.

### Reasoning and Chat

1. Decompose query into sub-questions.
2. Detect intent.
3. Retrieve initial facts.
4. Expand through graph and entity index.
5. Check sufficiency with LLM.
6. Fall back to chunk similarity and optional Recoll search.
7. Verify claims with SymStep, VeriCoT, FiDeLiS, R-CoT, ARES.
8. Re-rank using memory and logic modules.
9. Synthesize final answer with provenance.

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
4. Results are stored in `standard_comparisons`.
5. Admin and verified facts are never auto-deleted or demoted.

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

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## License

[MIT](LICENSE)

---

## Disclaimer

This project is under active development. Some reasoning components may be heuristic and may require tuning for production use. This is a work in progress and is considered highly experimental.
