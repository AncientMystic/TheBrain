
# TheBrain

TheBrain is a local-first document analysis and knowledge extraction engine that transforms large collections of files (PDFs, HTML, Markdown, DOCX, EPUB, plain text, and more) into a structured, queryable knowledge graph with verification-first reasoning.

It uses local LLMs and embedding models via LM Studio to extract facts, entities, relationships, dates, locations, events, discoveries, and notable insights from entire documents. The extracted knowledge is stored in SQLite databases and can be queried through a built-in chat interface or an OpenAI-compatible API server.

---

## Features

- **Multi-format ingestion**  
  PDF, HTML, Markdown, DOCX, EPUB, RTF, Jupyter Notebook, plain text, source code, and more.

- **Full-document processing**  
  Every page and every chunk is processed—no truncation or arbitrary limits.

- **Deep LLM knowledge extraction**  
  Extracts atomic facts, typed entities, people, locations, dates, events, discoveries, gems, and relationships with source spans.

- **Verification-first reasoning**  
  Implements multiple verification layers (SymStep, VeriCoT, FiDeLiS, R-CoT, ARES) to ensure answers are grounded and reliable.

- **Knowledge graphs**  
  - `hypergraph.db` – intra-document nodes and edges.  
  - `external-graph.db` – cross-document global nodes, aliases, edges, keyword-topic relations, co-occurrence, and cross-document links.

- **Novelty-gated chunk processing**  
  Skips chunks that are both semantically similar to already-processed chunks and introduce no new named entities/dates/locations.

- **Multi-stage validation**  
  Source span verification, confidence filtering, deduplication, and fact-source tracking.

- **Embeddings**  
  Document- and chunk-level embeddings stored in SQLite, used for semantic retrieval and fallback.

- **Graph-first chat with adaptive reasoning**  
  Expands from initial facts through related keywords, entities, and graph edges before falling back to chunk retrieval.

- **Memory & logic modules**  
  - `memories.db` – long-term chat memory.  
  - `logic.db` – learned logic, reasoning patterns, skills, strategies, etc.

- **OpenAI-compatible server**  
  Exposes `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/models`, and `/v1/reasoning`.

- **Audit & governance**  
  Automatic cleanup, contradiction detection, confidence scoring, and provenance tracking.

- **Parallel processing**  
  Supports up to three separate LM Studio endpoints with per-endpoint concurrency capacities.

---

## Architecture

### Databases

| File | Contents |
|------|----------|
| `index.db` | Documents, chunks, processing progress, LLM extraction cache |
| `summaries.db` | Document summaries and key points |
| `key_facts.db` | Facts, entities, people, locations, dates, events, discoveries, gems, fact sources |
| `embeddings.db` | Document embeddings, chunk embeddings, embedding cache |
| `hypergraph.db` | Intra-document nodes and edges |
| `external-graph.db` | Global nodes, global edges, keyword-topic edges, co-occurrence, cross-document links |
| `ocr_cache.db` | OCR text cache |
| `memories.db` | Memory entries, memory keywords, sessions |
| `logic.db` | Logic modules, examples, keywords, tags |
| `reasoning.db` | Reasoning nodes, edges, grounding records, KG triples, verification results, contradictions, agent actions |

### Reasoning Pipeline

1. **Query decomposition** – Break the query into atomic sub-questions with verification methods.
2. **Graph-first retrieval** – Retrieve facts and entities from the external graph.
3. **Adaptive expansion** – Expand through related keywords, entities, and graph edges until the LLM decides the context is sufficient.
4. **Chunk fallback** – Only if the graph facts are insufficient, retrieve relevant chunks via embedding similarity.
5. **Verification** – Each atomic claim is checked against multiple layers (SymStep, VeriCoT, FiDeLiS, R-CoT, ARES).
6. **Synthesis** – Generate a final answer using only verified facts and relevant excerpts.

---

## Installation

### Prerequisites

- Python 3.11+
- Tesseract OCR (optional, for scanned PDFs)
- LM Studio running locally (or on a network) with at least one LLM and embedding model loaded.

### Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, create it with:

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
```

### Tesseract Setup

- Windows: download from [GitHub UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH.
- Linux: `sudo apt install tesseract-ocr`
- macOS: `brew install tesseract`

---

## Configuration

All settings are in `config.py`. Important options:

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

Leave `_2`/`_3` blank to disable extra endpoints. If configured, the system will distribute requests concurrently.

### Concurrency Capacities

```python
LLM_ENDPOINT_CAPACITIES = [3, 1, 1]  # must match number of endpoints
```

### Chunking

```python
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
LLM_BATCH_CHUNKS = 2
CHUNK_EXTRACTION_WORKERS = 3
```

### Novelty Gating

```python
NOVELTY_ENABLED = True
NOVELTY_SIM_THRESHOLD = 0.92
```

### Server Settings

```python
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
SERVER_AUTH_TOKEN = ""
```

---

## Usage

### Initialize Databases

```bash
python scripts/init_schemas.py
```

Or run any mode; `main.py` will initialize automatically.

### Guided Learning

Process documents and build the knowledge graph:

```bash
python main.py --guided-learning --input "A:/pdfs"
```

This will:
- Scan recursively for supported files.
- Extract text, chunk, embed, extract knowledge, build graphs, and store everything.

### Chat

Start an interactive chat:

```bash
python main.py --chat
```

Use `--reasoning` to enable verification-first reasoning:

```bash
python main.py --chat --reasoning
```

Add `--debug` to show LLM/embedding endpoint logs:

```bash
python main.py --chat --reasoning --debug
```

### Server

Launch the OpenAI-compatible API:

```bash
python main.py --server
```

Then use:

- `POST /v1/chat/completions` with JSON body:

```json
{
  "model": "lfm2.5-vl-3b-absolute-heresy-i1",
  "messages": [{"role": "user", "content": "What is Zealandia?"}],
  "reasoning": true
}
```

- `GET /v1/models`
- `POST /v1/embeddings`
- `POST /v1/reasoning`

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

---

## Directory Structure

```
TheBrain/
├── main.py
├── server.py
├── config.py
├── apply_changes.py
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
├── scripts/
└── data/              # SQLite databases
```

---

## How It Works

### Document Processing

1. Scan input directory.
2. Hash each file and check processing status.
3. Extract full text using the appropriate extractor.
4. Chunk text with overlap.
5. Generate document and chunk embeddings.
6. Novelty gate determines which chunks need LLM extraction.
7. LLM extracts atomic claims, entities, etc.
8. Validate and deduplicate.
9. Store facts in `key_facts.db`, build hypergraph and external graph.
10. Generate summary and mark processed.

### Reasoning and Chat

1. Decompose query into sub-questions.
2. Retrieve initial facts from the external graph.
3. Expand facts by following related keywords and entity edges.
4. Check sufficiency with LLM.
5. If insufficient, add chunks via embedding similarity.
6. Verify claims with SymStep, VeriCoT, FiDeLiS, R-CoT, ARES.
7. Synthesize final answer with provenance.

---

## License

MIT

---

## Disclaimer

This project is under active development. Some reasoning components may be heuristic and may require tuning for production use. This is a work in progress. it is very much so, thrown together. it is to be considered **highly experimental**. 
```
