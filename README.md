# TheBrain

TheBrain is a **local‑first document intelligence and knowledge extraction engine**.  
It ingests large collections of mixed‑format documents (PDF, HTML, Markdown, DOCX, EPUB, RTF, Jupyter Notebook, plain text, source code, and more), extracts structured knowledge using local LLMs and embedding models via **LM Studio**, and stores it across multiple SQLite databases.

The system includes **graph‑based retrieval**, **verification‑first reasoning**, **long‑term memory**, **learned logic modules**, an **OpenAI‑compatible server**, automatic **audit/governance**, optional **deep research mode** with autonomous report generation, and **full‑text search integration via Recoll**.

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
  - [Chat](#chat)
  - [Deep Research Mode](#deep-research-mode)
  - [Recoll‑Guided Autonomous Learning](#recoll-guided-autonomous-learning)
  - [Recoll Fast Mode](#recoll-fast-mode)
  - [Build Recoll Index](#build-recoll-index)
  - [Server](#server)
  - [Logic Learning](#logic-learning)
  - [Audit](#audit)
  - [Review Contradictions](#review-contradictions)
- [Directory Structure](#directory-structure)
- [Database Schemas](#database-schemas)
- [How It Works](#how-it-works)
  - [Document Processing](#document-processing)
  - [Reasoning and Chat](#reasoning-and-chat)
  - [Deep Research](#deep-research)
  - [Recoll‑Guided Learning](#recoll-guided-learning)
  - [How Recoll Fast Mode Works](#how-recoll-fast-mode-works)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

---

## Features

- **Multi‑format ingestion**  
  PDF, HTML, Markdown, DOCX, EPUB, RTF, Jupyter Notebook, plain text, source code, and more.

- **Full‑document processing**  
  Every page and every chunk is processed—no truncation or arbitrary limits.

- **OCR fallback for scanned PDFs**  
  PyMuPDF + Tesseract OCR when text extraction fails.

- **Deep LLM knowledge extraction**  
  Extracts atomic facts, typed entities, people, locations, dates, events, discoveries, gems, and relationships with source spans. Includes:
  - Adaptive system prompts based on document type (academic, technical, code, transcript, etc.).
  - Stronger source‑span enforcement to reduce verbatim copying.
  - Relationship type vocabulary for precise relation extraction.
  - Hierarchical summarization for long documents.
  - Optional fast pre‑extraction using an ONNX NER model to accelerate entity/date/location extraction, with LLM verification.

- **Verification‑first reasoning**  
  Implements multiple verification layers (SymStep, VeriCoT, FiDeLiS, R‑CoT, ARES) with optional adaptive escalation (cheap checks first, heavy LLM checks only when needed).

- **Knowledge graphs**  
  - `hypergraph.db` – intra‑document nodes and edges.  
  - `external-graph.db` – cross‑document global nodes, aliases, edges, keyword‑topic relations, co‑occurrence, and cross‑document links.  
  - Precomputed entity‑fact index for fast multi‑hop expansion.

- **Graph‑first chat with adaptive reasoning**  
  Expands from initial facts through related keywords, entities, and graph edges (including hypergraph) before falling back to chunk retrieval. Includes:
  - Intent detection (factual, comparative, causal, temporal, summary).
  - Conversation history and session context.
  - Memory and logic module‑informed re‑ranking.
  - Enhanced answer prompt with source prioritization and Markdown formatting.

- **Memory & logic modules**  
  - `memories.db` – long‑term chat memory with optional temporal decay.  
  - `logic.db` – learned logic, reasoning patterns, skills, strategies, etc. Logic modules can now influence retrieval and response generation.

- **Deep Research mode**  
  Autonomous research with recursive subtopic exploration, mindmap construction, and multi‑page Markdown report generation.

- **Recoll full‑text search integration**  
  - Optional wrapper for the Recoll Python API.  
  - Build and maintain a separate full‑text index over your documents.  
  - Use Recoll as an additional evidence source in chat and deep research.  
  - Autonomous mode: `--guided-learning --recoll` that uses knowledge gaps to generate Recoll queries, process retrieved documents, and expand the knowledge graph automatically.  
  - Dedicated `recoll_log.db` to avoid duplicate queries and track processed documents.

- **Recoll Fast Mode (new)**  
  Lightweight, keyword‑focused learning using `--recoll-fast`. Processes contextual previews around search hits instead of full documents, enabling rapid learning from many results.

- **OpenAI‑compatible server**  
  Exposes `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/models`, `/v1/reasoning`, and `/v1/health`.

- **Audit & governance**  
  Automatic cleanup, contradiction detection, confidence scoring, provenance tracking, and a review queue for unresolved contradictions.

- **Performance optimizations**  
  - LLM extraction cache.  
  - Optional HTTP session reuse.  
  - Embedding batch size configurable (default 32).  
  - FTS‑backed keyword search with caching.  
  - External graph embedding matrix for fast fuzzy matching.  
  - Configurable parallel file processing.  
  - Progress bars (if `tqdm` installed).  
  - Optional async LLM calls (requires `aiohttp`).  
  - Optional small/large/audit model roles.  
  - Multi‑endpoint embedding generation with local cache.

---

## Installation

### Prerequisites

- **Python 3.11+**
- **Tesseract OCR** (optional, for scanned PDFs)
- **LM Studio** running locally (or on a network) with at least one LLM and one embedding model loaded
- *(Optional)* **ONNX Runtime** and **Hugging Face Hub** for the fast NER extractor
- *(Optional)* **Recoll** with Python API or `recollq` CLI for full‑text search

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
aiohttp   # optional, for async LLM
tqdm      # optional, for progress bars
```

### Tesseract OCR Setup

- **Windows**: download from [GitHub UB‑Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH.  
- **Linux**: `sudo apt install tesseract-ocr`  
- **macOS**: `brew install tesseract`

### ONNX NER Model

The fast extractor can automatically download a pre‑trained ONNX NER model from Hugging Face.  
The default repository is `optimum/bert-base-NER` or `Xenova/bert-base-NER` (configured in `config.py`).  
You can also manually download and place the model files in `models/ner_onnx` (or your configured directory).  
Set `FAST_EXTRACTOR_ENABLED = False` to disable.

### Recoll Setup (Optional)

1. Install Recoll on your system.
2. Ensure the `recollq` CLI is available in your `PATH` **or** install the Python API:

   **Windows (Python API)**  
   The wheel is usually located in `C:\Program Files\Recoll\Share\dist`.

   ```powershell
   pip install "C:\Program Files\Recoll\Share\dist\Recoll-<version>-cp3xx-cp3xx-win_amd64.whl"
   ```

   **Linux/macOS**  
   Use your package manager or compile from source, then make sure `recollq` is in `PATH`.

3. Verify:

   ```python
   # If using Python API
   from recoll import recoll
   ```

   ```bash
   # If using CLI
   recollq -h
   ```

4. Set `USE_RECOLL = true` in `config.py` or `USE_RECOLL=true` in your environment.

5. Build a Recoll index over your documents (see [Build Recoll Index](#build-recoll-index)).

---

## Configuration

All settings live in `config.py`. Many can be overridden via environment variables.

### LM Studio Endpoints

```python
LM_STUDIO_URL = "http://localhost:1234/v1"
MODEL_NAME = "lfm2.5-vl-3b-absolute-heresy-i1"
EMBEDDING_MODEL = "smcleod/text-embedding-mxbai-embed-large-v1"

LM_STUDIO_URL_2 = ""
MODEL_NAME_2 = ""
EMBEDDING_MODEL_2 = ""

LM_STUDIO_URL_3 = ""
MODEL_NAME_3 = ""
EMBEDDING_MODEL_3 = ""
```

Leave `_2` and `_3` blank to disable extra endpoints. If configured, requests are distributed concurrently.

### Optional Model Roles

You can specify separate models for different tasks:

```python
SMALL_MODEL_URL = ""   # if set, used for initial extraction
SMALL_MODEL_NAME = ""
LARGE_MODEL_URL = ""   # if set, used for verification/gap‑filling
LARGE_MODEL_NAME = ""
AUDIT_MODEL_URL = ""   # if set, used for audit tasks
AUDIT_MODEL_NAME = ""
```

If left empty, the main model is used for all tasks.

### Concurrency and Chunking

```python
LLM_ENDPOINT_CAPACITIES = [3, 1, 1]  # must match number of endpoints

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
LLM_BATCH_CHUNKS = 4
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
LLM_EXTRACTION_CACHE = True          # reuse previous extractions
USE_LLM_HTTP_SESSION = True          # reuse HTTP connections
FTS_ENABLED = True                   # use FTS5 for keyword search
EMBEDDING_BATCH_SIZE = 32            # batch size for embedding calls
PARALLEL_PROCESSING_ENABLED = False  # enable parallel file processing
PARALLEL_WORKERS = 2
USE_PROGRESS_BARS = True             # show progress bars if tqdm available
ADAPTIVE_VERIFICATION = True         # escalate verification only when needed
AUTO_RESOLVE_CONTRADICTIONS = False  # false = move to review queue
DEEP_RESEARCH_INTERACTIVE = True     # ask user before exploring subtopics
DEEP_RESEARCH_AUTO_SUBTOPIC_DEPTH = 2
```

### Recoll Settings

```python
USE_RECOLL = os.environ.get("USE_RECOLL", "false").lower() == "true"
RECOLL_CONFDIR = os.environ.get("RECOLL_CONFDIR", "")
RECOLL_EXTRA_DBS = os.environ.get("RECOLL_EXTRA_DBS", "")  # space‑separated list
RECOLL_DEFAULT_LIMIT = int(os.environ.get("RECOLL_DEFAULT_LIMIT", "20"))
RECOLL_MAX_ROUNDS = int(os.environ.get("RECOLL_MAX_ROUNDS", "10"))
RECOLL_INTERACTIVE = os.environ.get("RECOLL_INTERACTIVE", "false").lower() == "true"
RECOLL_LOG_DB_FILE = str(DATA_DIR / "recoll_log.db")
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LM_STUDIO_URL` | `http://localhost:1234/v1` | Primary LLM endpoint |
| `MODEL_NAME` | `lfm2.5-vl-3b-absolute-heresy-i1` | Primary LLM model |
| `EMBEDDING_MODEL` | `smcleod/text-embedding-mxbai-embed-large-v1` | Embedding model |
| `CHUNK_SIZE` | `2000` | Document chunk size (characters) |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `EMBEDDING_BATCH_SIZE` | `16` | Batch size for embedding requests |
| `LLM_BATCH_CHUNKS` | `2` | Number of chunks per LLM extraction call |
| `NOVELTY_ENABLED` | `true` | Skip redundant chunks |
| `RECOLL_BIN` | `recollq` | Path to Recoll CLI binary |
| `RECOLL_DB` | `""` (empty) | Recoll config directory (empty uses default) |
| `RECOLL_MAX_RESULTS` | `50` | Default max results for Recoll queries |
| `PREVIEW_CHAR_WINDOW` | `1000` | Character window for fast mode previews |
| `RECOLL_AUTO_KEYWORD_LIMIT` | `20` | Number of seed keywords for automatic fast mode |

---

## Usage

### Initialize Databases

```bash
python scripts/init_schemas.py
```

Or simply run any mode; `main.py` will initialize automatically.

### Guided Learning

Process documents and build the knowledge graph:

```bash
python main.py --guided-learning --input "A:/pdfs"
```

Optional flags:

- `--debug` – show detailed logs.
- `--logic` – also use learned logic modules during processing (requires previously learned modules).

### Chat

Start an interactive chat:

```bash
python main.py --chat
```

Use `--reasoning` to enable verification‑first reasoning:

```bash
python main.py --chat --reasoning
```

Add `--debug` to show LLM/embedding endpoint logs:

```bash
python main.py --chat --reasoning --debug
```

Chat now supports:

- Conversation history (last 10 turns).
- Memory storage (`remember: <content>`).
- Graph‑first retrieval with multi‑hop expansion.
- Intent‑aware answer generation.
- Markdown output with source citations.
- Optional Recoll full‑text search when `USE_RECOLL=true`.

### Deep Research Mode

To enable autonomous deep research on a topic, combine `--chat` with `--deep-research`:

```bash
python main.py --chat --deep-research
```

When you enter a query, the system will:

1. Retrieve initial facts and expand through the knowledge graph.
2. Build a mindmap of related concepts.
3. Suggest subtopics (and optionally ask if you want to explore each).
4. Generate a comprehensive Markdown report in the `reports/` folder.
5. Recursively explore subtopics up to the configured depth.

### Recoll‑Guided Autonomous Learning

Use Recoll to fill knowledge gaps automatically:

```bash
python main.py --guided-learning --recoll
```

Optional flags:

- `--recoll-max-rounds 5` – maximum autonomous iterations.
- `--recoll-interactive` – ask before processing each retrieved document.
- `--debug` – extra logging.

How it works:

1. Analyzes internal databases for gaps (low confidence facts, sparse entities, low‑coverage keywords, unconnected topics).
2. Generates Recoll search queries using the LLM.
3. Queries the Recoll index to find relevant documents.
4. Processes new documents through TheBrain’s extraction pipeline.
5. Logs all queries and processed documents in `recoll_log.db` to avoid duplicates.

### Recoll Fast Mode

`--recoll-fast` is designed for **lightweight, keyword‑focused learning**. Instead of pulling entire documents, it:

1. Runs a Recoll query for the given keyword (or automatic seed keywords).
2. Extracts a contextual preview around the keyword (default ±1000 chars).
3. Stores the preview as a pseudo‑document.
4. Generates embeddings.
5. Runs LLM extraction on the preview chunk(s).
6. Stores extracted facts/entities/relationships into the knowledge base.

**Examples**

Search for a specific keyword:

```bash
python main.py --recoll-fast --recoll-query "keyword"
```

Automatic mode (uses seed keywords from the knowledge base):

```bash
python main.py --recoll-fast
```

Interactive mode:

```bash
python main.py --recoll-fast --interactive
```

Options:

- `--recoll-query "keyword"` – specify the query.
- `--recoll-limit 50` – maximum number of Recoll results to process.
- `--preview-chars 1000` – character window around the keyword.

### Build Recoll Index

To build/update a separate full‑text index for your documents:

```bash
python main.py --build-recoll-index --input "A:/documents"
```

This will scan the folder, extract text, and add each document to the Recoll index using a writable connection.

### Server

Launch the OpenAI‑compatible API:

```bash
python main.py --server
```

Endpoints:

- `POST /v1/chat/completions` – chat with optional `reasoning: true` or `deep_research: true`.
- `POST /v1/completions` – text completion.
- `POST /v1/responses` – responses API format.
- `POST /v1/embeddings` – generate embeddings.
- `GET /v1/models` – list available models.
- `POST /v1/reasoning` – reasoning endpoint.
- `GET /v1/health` – health check with endpoint status.

### Logic Learning

Learn logic modules from examples:

```bash
python main.py --logic --input "A:/logic_examples"
```

### Audit

Run automatic cleanup and contradiction detection:

```bash
python main.py --audit
```

To review unresolved contradictions without deleting anything:

```bash
python main.py --review-contradictions
```

---

## Directory Structure

```
TheBrain/
├── main.py
├── server.py
├── config.py
├── core/
├── extractors/
├── ingestion/
├── extraction/
├── graph/
├── chat/
├── memory/
├── logic/
├── reasoning/
├── audit/
├── deep_research/
├── fast_extractor/
├── scripts/
├── models/               # ONNX NER model
├── reports/              # generated research reports
└── data/                 # SQLite databases (including recoll_log.db) 
```

---

## Database Schemas

TheBrain uses multiple SQLite databases:

| Database | Tables |
|----------|--------|
| `data/index.db` | documents, document_chunks, processing_progress, llm_extraction_cache |
| `data/summaries.db` | doc_summaries, summary_versions |
| `data/key_facts.db` | key_facts, entities, people, locations, dates, events, discoveries, gems, fact_sources, key_facts_fts |
| `data/embeddings.db` | document_embeddings, chunk_embeddings, embedding_cache |
| `data/hypergraph.db` | nodes, edges, doc_entity_nodes |
| `data/external-graph.db` | global_nodes, global_edges, topic_nodes, keyword_topic_edges, keyword_cooccurrence, cross_doc_links, topic_hierarchy, global_nodes_fts |
| `data/ocr_cache.db` | ocr_cache |
| `data/memories.db` | memory_entries, memory_keywords, memory_sessions |
| `data/logic.db` | logic_modules, logic_examples, logic_keywords, logic_tags |
| `data/reasoning.db` | reasoning_nodes, reasoning_edges, grounding_records, kg_triples, reasoning_paths, reasoning_dependencies, verification_results, contradiction_log, agent_actions |
| `data/recoll_log.db` | recoll_queries, processed_documents |

All schemas are created automatically on first run.

---

## How It Works

### Document Processing

1. Scan input directory.
2. Hash each file and check processing status.
3. Extract full text using the appropriate extractor.
4. Chunk text with overlap.
5. Generate document and chunk embeddings.
6. (Optional) Run fast extractor pre‑pass (ONNX NER + rules) to pre‑extract entities, dates, locations.
7. Novelty gate determines which chunks need LLM extraction.
8. LLM extracts atomic claims, entities, relationships, etc. (verifying pre‑extractions if provided).
9. Validate, deduplicate, and store facts in `key_facts.db`, populate entity‑fact index, build hypergraph and external graph.
10. Generate hierarchical summary and mark processed.

### Reasoning and Chat

1. Decompose query into sub‑questions.
2. Detect intent (factual, comparative, causal, temporal, summary).
3. Retrieve initial facts from the external graph, optionally using FTS.
4. Expand facts through multi‑hop graph traversal (keyword co‑occurrence, global edges, hypergraph, entity‑fact index).
5. Check sufficiency with LLM.
6. If insufficient, add chunks via embedding similarity (and optionally Recoll full‑text results).
7. Verify claims with SymStep, VeriCoT, FiDeLiS, R‑CoT, ARES (adaptive if enabled).
8. Re‑rank facts using memory and logic modules.
9. Synthesize final answer with provenance and Markdown formatting.

### Deep Research

1. Start from the main query.
2. Build a mindmap with hierarchical relationships between concepts.
3. Discover subtopics via LLM and graph analysis.
4. Optionally ask user which subtopics to pursue.
5. Generate a detailed report for each subtopic, including a coherence pass.
6. Save reports to `reports/`.

### Recoll‑Guided Learning

1. Query internal SQLite for knowledge gaps.
2. Generate Recoll search queries from those gaps using LLM.
3. Search the Recoll full‑text index.
4. Process retrieved documents through the standard pipeline.
5. Log all queries and processed documents.
6. Repeat until no new documents are processed or max rounds reached.

### How Recoll Fast Mode Works

1. Select a keyword (explicit or from automatic seed keywords derived from knowledge gaps).
2. Run `recollq` (or Recoll Python API) for that keyword.
3. For each result, extract a contextual preview around the keyword (default ±1000 chars).
4. Treat each preview as a pseudo‑document.
5. Generate embeddings for the preview.
6. Run LLM extraction on the preview chunk(s).
7. Store extracted facts, entities, relationships, and source references in the knowledge base.
8. Log the query and processed results in `recoll_log.db` to prevent duplicates.

---

## Limitations

- Some reasoning components (VeriCoT, R‑CoT, ARES) are heuristic rather than full formal solvers.
- Contradiction auto‑resolution deletes lower‑confidence entries; may need human review in production.
- SQLite may struggle with millions of edges; no vector database integration yet.
- JSON parsing can occasionally fail depending on the local LLM; retry mechanism helps but not perfect.
- No web UI; server API and CLI only.
- Not battle‑tested on very large heterogeneous corpora.

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## License

[MIT](LICENSE)

---

## Disclaimer

This project is under active development. Some reasoning components may be heuristic and may require tuning for production use. This is a work in progress and is considered highly experimental.
