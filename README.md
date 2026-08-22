# TheBrain

TheBrain is a local-first document intelligence and knowledge extraction engine. It ingests large collections of mixed-format documents (PDF, HTML, Markdown, DOCX, EPUB, RTF, Jupyter Notebook, plain text, source code, and more), extracts structured knowledge using local LLMs and embedding models via LM Studio, and stores it across multiple SQLite databases. The system includes graph-based retrieval, verification-first reasoning, long-term memory, learned logic modules, an OpenAI-compatible server, automatic audit/governance, optional deep research mode with autonomous report generation, and full-text search integration via Recoll.

---

## Features

- **Multi-format ingestion**  
  PDF, HTML, Markdown, DOCX, EPUB, RTF, Jupyter Notebook, plain text, source code, and more.

- **Full-document processing**  
  Every page and every chunk is processed—no truncation or arbitrary limits.

- **Deep LLM knowledge extraction**  
  Extracts atomic facts, typed entities, people, locations, dates, events, discoveries, gems, and relationships with source spans. Now includes:
  - Adaptive system prompts based on document type (e.g., academic, technical, code, transcript).
  - Stronger source‑span enforcement to reduce verbatim copying.
  - Relationship type vocabulary for precise relation extraction.
  - Hierarchical summarization for long documents.
  - Optional fast pre‑extraction using an ONNX NER model to accelerate entity/date/location extraction, with LLM verification.

- **Verification-first reasoning**  
  Implements multiple verification layers (SymStep, VeriCoT, FiDeLiS, R‑CoT, ARES) with optional adaptive escalation (cheap checks first, heavy LLM checks only when needed).

- **Knowledge graphs**  
  - `hypergraph.db` – intra-document nodes and edges.  
  - `external-graph.db` – cross-document global nodes, aliases, edges, keyword-topic relations, co-occurrence, and cross-document links.  
  - Precomputed entity‑fact index for fast multi‑hop expansion.

- **Graph-first chat with adaptive reasoning**  
  Expands from initial facts through related keywords, entities, and graph edges (including hypergraph) before falling back to chunk retrieval. Includes:
  - Intent detection (factual, comparative, causal, temporal, summary).
  - Conversation history and session context.
  - Memory and logic module‑informed re‑ranking.
  - Enhanced answer prompt with source prioritization and Markdown formatting.

- **Memory & logic modules**  
  - `memories.db` – long-term chat memory with optional temporal decay.  
  - `logic.db` – learned logic, reasoning patterns, skills, strategies, etc. Logic modules can now influence retrieval and response generation.

- **Deep Research mode**  
  Autonomous research with recursive subtopic exploration, mindmap construction, and multi‑page Markdown report generation.

- **Recoll full‑text search integration**  
  - Optional wrapper for the Recoll Python API.  
  - Build and maintain a separate full‑text index over your documents.  
  - Use Recoll as an additional evidence source in chat and deep research.  
  - New autonomous mode: `--guided-learning --recoll` that uses knowledge gaps to generate Recoll queries, process retrieved documents, and expand the knowledge graph automatically.  
  - Dedicated `recoll_log.db` to avoid duplicate queries and track processed documents.

- **OpenAI-compatible server**  
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

---

## Installation

### Prerequisites

- Python 3.11+
- Tesseract OCR (optional, for scanned PDFs)
- LM Studio running locally (or on a network) with at least one LLM and embedding model loaded.
- (Optional) ONNX Runtime and Hugging Face Hub for the fast NER extractor.
- (Optional) Recoll with Python API installed (for full‑text search).

### Install Dependencies

```bash
pip install -r requirements.txt
```

If requirements.txt is not present, create it with:

```
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

Tesseract Setup

· Windows: download from [GitHub UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH.
· Linux: sudo apt install tesseract-ocr
· macOS: brew install tesseract

ONNX NER Model

The fast extractor can automatically download a pre‑trained ONNX NER model from Hugging Face. The default repository is optimum/bert-base-NER or Xenova/bert-base-NER (configured in config.py). You can also manually download and place the model files in models/ner_onnx (or your configured directory). Set FAST_EXTRACTOR_ENABLED = False to disable.

Recoll Setup (Optional)

1. Install Recoll on your system. For Windows, ensure the Python API wheel matches your Python version (usually found in C:\Program Files\Recoll\Share\dist).
2. Install the Python API:
   ```powershell
   pip install "C:\Program Files\Recoll\Share\dist\Recoll-<version>-cp3xx-cp3xx-win_amd64.whl"
   ```
3. Verify:
   ```python
   from recoll import recoll
   ```
4. Set USE_RECOLL = true in config.py or environment variable USE_RECOLL=true.

---

Configuration

All settings are in config.py. Important options:

LM Studio Endpoints

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

Leave _2/_3 blank to disable extra endpoints. If configured, the system will distribute requests concurrently.

Optional Model Roles (Small / Large / Audit)

You can specify separate models for different tasks via environment variables or by editing config.py:

```python
SMALL_MODEL_URL = ""   # if set, used for initial extraction
SMALL_MODEL_NAME = ""
LARGE_MODEL_URL = ""   # if set, used for verification/gap-filling
LARGE_MODEL_NAME = ""
AUDIT_MODEL_URL = ""   # if set, used for audit tasks
AUDIT_MODEL_NAME = ""
```

If left empty, the main model is used for all tasks.

Concurrency Capacities

```python
LLM_ENDPOINT_CAPACITIES = [3, 1, 1]  # must match number of endpoints
```

Chunking

```python
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
LLM_BATCH_CHUNKS = 4   # number of chunks per LLM call (configurable)
CHUNK_EXTRACTION_WORKERS = 3
```

Novelty Gating

```python
NOVELTY_ENABLED = True
NOVELTY_SIM_THRESHOLD = 0.92
```

Fast Extractor

```python
FAST_EXTRACTOR_ENABLED = True
FAST_EXTRACTOR_MODEL_NAME = "optimum/bert-base-NER"
FAST_EXTRACTOR_MODEL_DIR = "models/ner_onnx"
FAST_EXTRACTOR_CONFIDENCE_THRESHOLD = 0.7
```

Performance / Quality Flags

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

Recoll Settings

```python
USE_RECOLL = os.environ.get("USE_RECOLL", "false").lower() == "true"
RECOLL_CONFDIR = os.environ.get("RECOLL_CONFDIR", "")
RECOLL_EXTRA_DBS = os.environ.get("RECOLL_EXTRA_DBS", "")  # space-separated list
RECOLL_DEFAULT_LIMIT = int(os.environ.get("RECOLL_DEFAULT_LIMIT", "20"))
RECOLL_MAX_ROUNDS = int(os.environ.get("RECOLL_MAX_ROUNDS", "10"))
RECOLL_INTERACTIVE = os.environ.get("RECOLL_INTERACTIVE", "false").lower() == "true"
RECOLL_LOG_DB_FILE = str(DATA_DIR / "recoll_log.db")
```

---

Usage

Initialize Databases

```bash
python scripts/init_schemas.py
```

Or simply run any mode; main.py will initialize automatically.

Guided Learning

Process documents and build the knowledge graph:

```bash
python main.py --guided-learning --input "A:/pdfs"
```

Optional flags:

· --debug – show detailed logs.
· --logic – also use learned logic modules during processing (requires previously learned modules).

Chat

Start an interactive chat:

```bash
python main.py --chat
```

Use --reasoning to enable verification-first reasoning:

```bash
python main.py --chat --reasoning
```

Add --debug to show LLM/embedding endpoint logs:

```bash
python main.py --chat --reasoning --debug
```

Chat now supports:

· Conversation history (last 10 turns).
· Memory storage (remember: <content>).
· Graph-first retrieval with multi-hop expansion.
· Intent-aware answer generation.
· Markdown output with source citations.
· Optional Recoll full-text search when USE_RECOLL=true.

Deep Research Mode

To enable autonomous deep research on a topic, combine --chat with --deep-research:

```bash
python main.py --chat --deep-research
```

When you enter a query, the system will:

1. Retrieve initial facts and expand through the knowledge graph.
2. Build a mindmap of related concepts.
3. Suggest subtopics (and optionally ask if you want to explore each).
4. Generate a comprehensive Markdown report in the reports/ folder.
5. Recursively explore subtopics up to the configured depth.

Recoll-Guided Autonomous Learning

Use Recoll to fill knowledge gaps automatically:

```bash
python main.py --guided-learning --recoll
```

Optional flags:

· --recoll-max-rounds 5 – maximum autonomous iterations.
· --recoll-interactive – ask before processing each retrieved document.
· --debug – extra logging.

How it works:

1. Analyzes internal databases for gaps (low confidence facts, sparse entities, low-coverage keywords, unconnected topics).
2. Generates Recoll search queries using the LLM.
3. Queries the Recoll index to find relevant documents.
4. Processes new documents through TheBrain’s extraction pipeline.
5. Logs all queries and processed documents in recoll_log.db to avoid duplicates.

Build Recoll Index

To build/update a separate full-text index for your documents:

```bash
python main.py --build-recoll-index --input "A:/documents"
```

This will scan the folder, extract text, and add each document to the Recoll index using a writable connection.

Server

Launch the OpenAI-compatible API:

```bash
python main.py --server
```

Endpoints:

· POST /v1/chat/completions – chat with optional reasoning: true or deep_research: true.
· POST /v1/completions – text completion.
· POST /v1/responses – responses API format.
· POST /v1/embeddings – generate embeddings.
· GET /v1/models – list available models.
· POST /v1/reasoning – reasoning endpoint.
· GET /v1/health – health check with endpoint status.

Logic Learning

Learn logic modules from examples:

```bash
python main.py --logic --input "A:/logic_examples"
```

Audit

Run automatic cleanup and contradiction detection:

```bash
python main.py --audit
```

To review unresolved contradictions without deleting anything:

```bash
python main.py --review-contradictions
```

---

Directory Structure

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
├── data/                 # SQLite databases (including recoll_log.db)
└── upload-to-github/     # prepared repository folder (if created)
```

---

How It Works

Document Processing

1. Scan input directory.
2. Hash each file and check processing status.
3. Extract full text using the appropriate extractor.
4. Chunk text with overlap.
5. Generate document and chunk embeddings.
6. (Optional) Run fast extractor pre‑pass (ONNX NER + rules) to pre‑extract entities, dates, locations.
7. Novelty gate determines which chunks need LLM extraction.
8. LLM extracts atomic claims, entities, relationships, etc. (verifying pre‑extractions if provided).
9. Validate, deduplicate, and store facts in key_facts.db, populate entity‑fact index, build hypergraph and external graph.
10. Generate hierarchical summary and mark processed.

Reasoning and Chat

1. Decompose query into sub-questions.
2. Detect intent (factual, comparative, causal, temporal, summary).
3. Retrieve initial facts from the external graph, optionally using FTS.
4. Expand facts through multi‑hop graph traversal (keyword co‑occurrence, global edges, hypergraph, entity‑fact index).
5. Check sufficiency with LLM.
6. If insufficient, add chunks via embedding similarity (and optionally Recoll full‑text results).
7. Verify claims with SymStep, VeriCoT, FiDeLiS, R‑CoT, ARES (adaptive if enabled).
8. Re‑rank facts using memory and logic modules.
9. Synthesize final answer with provenance and Markdown formatting.

Deep Research

1. Start from the main query.
2. Build a mindmap with hierarchical relationships between concepts.
3. Discover subtopics via LLM and graph analysis.
4. Optionally ask user which subtopics to pursue.
5. Generate a detailed report for each subtopic, including a coherence pass.
6. Save reports to reports/.

Recoll-Guided Learning

1. Query internal SQLite for knowledge gaps.
2. Generate Recoll search queries from those gaps using LLM.
3. Search the Recoll full-text index.
4. Process retrieved documents through the standard pipeline.
5. Log all queries and processed documents.
6. Repeat until no new documents are processed or max rounds reached.

---

License

MIT

---

Disclaimer

This project is under active development. Some reasoning components may be heuristic and may require tuning for production use. This is a work in progress and is considered highly experimental.
