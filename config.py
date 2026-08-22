from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
GAZETTEERS_DIR = BASE_DIR / "gazetteers"

for folder in [DATA_DIR, GAZETTEERS_DIR]:
    folder.mkdir(exist_ok=True, parents=True)

INDEX_DB_FILE = str(DATA_DIR / "index.db")
SUMMARIES_DB_FILE = str(DATA_DIR / "summaries.db")
KEY_FACTS_DB_FILE = str(DATA_DIR / "key_facts.db")
EMBEDDINGS_DB_FILE = str(DATA_DIR / "embeddings.db")
HYPERGRAPH_DB_FILE = str(DATA_DIR / "hypergraph.db")
EXTERNAL_GRAPH_DB_FILE = str(DATA_DIR / "external-graph.db")
OCR_CACHE_DB_FILE = str(DATA_DIR / "ocr_cache.db")
MEMORIES_DB_FILE = str(DATA_DIR / "memories.db")
LOGIC_DB_FILE = str(DATA_DIR / "logic.db")
REASONING_DB_FILE = str(DATA_DIR / "reasoning.db")

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
SERVER_AUTH_TOKEN = ""

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "lfm2.5-vl-3b-absolute-heresy-i1")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "smcleod/text-embedding-mxbai-embed-large-v1")

LM_STUDIO_URL_2 = os.environ.get("LM_STUDIO_URL_2", "http://10.0.0.33:1234/v1")
MODEL_NAME_2 = os.environ.get("MODEL_NAME_2", "lfm2.5-vl-3b-absolute-heresy-i1:2")
EMBEDDING_MODEL_2 = os.environ.get("EMBEDDING_MODEL_2", "smcleod/text-embedding-mxbai-embed-large-v1:2")

LM_STUDIO_URL_3 = os.environ.get("LM_STUDIO_URL_3", "")
MODEL_NAME_3 = os.environ.get("MODEL_NAME_3", "")
EMBEDDING_MODEL_3 = os.environ.get("EMBEDDING_MODEL_3", "")

LLM_ENDPOINTS = []
for url, model in [(LM_STUDIO_URL, MODEL_NAME), (LM_STUDIO_URL_2, MODEL_NAME_2), (LM_STUDIO_URL_3, MODEL_NAME_3)]:
    if url and model:
        LLM_ENDPOINTS.append({"url": url, "model": model, "api_key": "not-needed"})

EMBEDDING_ENDPOINTS = []
for url, model in [(LM_STUDIO_URL, EMBEDDING_MODEL), (LM_STUDIO_URL_2, EMBEDDING_MODEL_2), (LM_STUDIO_URL_3, EMBEDDING_MODEL_3)]:
    if url and model:
        EMBEDDING_ENDPOINTS.append({"url": url, "model": model, "api_key": "not-needed"})

if not LLM_ENDPOINTS:
    LLM_ENDPOINTS = [{"url": LM_STUDIO_URL, "model": MODEL_NAME, "api_key": "not-needed"}]
if not EMBEDDING_ENDPOINTS:
    EMBEDDING_ENDPOINTS = [{"url": LM_STUDIO_URL, "model": EMBEDDING_MODEL, "api_key": "not-needed"}]

LLM_ENDPOINT_CAPACITIES = [4, 2, 1]
while len(LLM_ENDPOINT_CAPACITIES) < len(LLM_ENDPOINTS):
    LLM_ENDPOINT_CAPACITIES.append(1)
LLM_ENDPOINT_CAPACITIES = LLM_ENDPOINT_CAPACITIES[:len(LLM_ENDPOINTS)]

API_RETRY_ATTEMPTS = 3
API_RETRY_BACKOFF = 2.0
API_TIMEOUT = 480
EMBEDDING_TIMEOUT = 240

USE_JSON_MODE = False

LLM_EXTRACTION_CACHE = False
LLM_CACHE_DB = "index"

LLM_BATCH_CHUNKS = 2
CHUNK_EXTRACTION_WORKERS = 3

NOVELTY_ENABLED = True
NOVELTY_SIM_THRESHOLD = 0.92
NOVELTY_NEW_ENTITY_RATIO = 0.0

FULL_DOC_PAGES = None
OCR_DPI = 80
TITLE_PAGE_DPI = 120
TITLE_PAGE_COUNT = 3
MIN_TEXT_CHARS_FOR_OCR_SKIP = 200

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
MAX_CHUNKS_PER_LLM_CALL = LLM_BATCH_CHUNKS

EMBEDDING_BATCH_SIZE = 16

MAX_TOPICS_PER_KEYWORD = 8
WEAK_DOC_COUNT = 3
WEAK_AVG_WEIGHT = 0.25
SEMANTIC_SIM_THRESHOLD = 0.35

GUIDED_LEARNING_STRENGTH = 2.0
GUIDED_LEARNING_MIN_CONFIDENCE = 0.5
GUIDED_LEARNING_PAGES_TO_OCR = None
GUIDED_LEARNING_OCR_DPI = 100

CHAT_MAX_CONTEXT_TOKENS = 8000
CHAT_MIN_FACTS_BEFORE_FALLBACK = 50
CHAT_TOP_K_CHUNKS = 8

# Debug verbosity (set by --debug flag)
DEBUG_VERBOSE = False

TEXT_EXTENSIONS = {
    '.txt', '.text', '.md', '.markdown', '.html', '.htm',
    '.pdf', '.docx', '.epub', '.rtf', '.ipynb',
    '.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.hpp',
    '.cs', '.go', '.rb', '.php', '.swift', '.sh', '.bat', '.ps1',
    '.json', '.xml', '.csv', '.yaml', '.yml', '.toml', '.ini',
    '.log', '.rst', '.tex', '.adoc'
}

IGNORE_DIRS = {
    '.git', 'node_modules', 'venv', '.venv', '__pycache__',
    'dist', 'build', 'vendor', 'assets', 'images', 'img',
    'font', 'fonts', 'static', 'public', 'target', 'bin', 'obj'
}

IGNORE_FILES = {
    '.DS_Store', 'Thumbs.db'
}
