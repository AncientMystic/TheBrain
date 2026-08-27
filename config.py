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
RECOLL_LOG_DB_FILE = str(DATA_DIR / "recoll_log.db")
MEMORIES_DB_FILE = str(DATA_DIR / "memories.db")
LOGIC_DB_FILE = str(DATA_DIR / "logic.db")
REASONING_DB_FILE = str(DATA_DIR / "reasoning.db")
VERIFICATION_STANDARDS_DB_FILE = str(DATA_DIR / "verification_standards.db")
VERIFICATION_FACTS_JSON_FILE = str(DATA_DIR / "verification_facts.json")

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
SERVER_AUTH_TOKEN = ""

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "lfm2.5-vl-3b-absolute-heresy-i1")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "smcleod/text-embedding-mxbai-embed-large-v1")

# Unified backend configuration
BACKENDS = []
# Backend config can be set via environment variable JSON or individual vars
BACKEND_CONFIG_JSON = os.environ.get("BACKEND_CONFIG_JSON", "")
if BACKEND_CONFIG_JSON:
    try:
        import json as _json
        BACKENDS = _json.loads(BACKEND_CONFIG_JSON)
    except Exception:
        print("Warning: BACKEND_CONFIG_JSON is not valid JSON; ignoring.")
else:
    # Build default backends from legacy LM Studio and new env vars
    _backend_url = os.environ.get("BACKEND_URL", LM_STUDIO_URL)
    _backend_model = os.environ.get("BACKEND_MODEL", MODEL_NAME)
    _backend_emb = os.environ.get("BACKEND_EMBEDDINGS_MODEL", EMBEDDING_MODEL)
    _backend_type = os.environ.get("BACKEND_TYPE", "lmstudio")
    _backend_key = os.environ.get("BACKEND_API_KEY", "not-needed")
    BACKENDS.append({
        "name": "default",
        "backend": _backend_type,
        "url": _backend_url,
        "model": _backend_model,
        "embeddings_model": _backend_emb,
        "api_key": _backend_key,
    })


LM_STUDIO_URL_2 = os.environ.get("LM_STUDIO_URL_2", "")
MODEL_NAME_2 = os.environ.get("MODEL_NAME_2", "")
EMBEDDING_MODEL_2 = os.environ.get("EMBEDDING_MODEL_2", "")

LM_STUDIO_URL_3 = os.environ.get("LM_STUDIO_URL_3", "")
MODEL_NAME_3 = os.environ.get("MODEL_NAME_3", "lfm2.5-vl-3b-absolute-heresy-i1")
EMBEDDING_MODEL_3 = os.environ.get("EMBEDDING_MODEL_3", "text-embedding-mxbai-embed-large-v1")

LLM_ENDPOINTS = []
if BACKENDS:
    # Build LLM endpoints from BACKENDS list
    for be in BACKENDS:
        if be.get("model") and be.get("url"):
            LLM_ENDPOINTS.append({
                "url": be.get("url"),
                "model": be.get("model"),
                "api_key": be.get("api_key", "not-needed"),
                "backend": be.get("backend", "lmstudio"),
                "embeddings_model": be.get("embeddings_model", be.get("model")),
                "capacity": be.get("capacity", 3 if len(LLM_ENDPOINTS) == 0 else 1),
            })
else:
    for url, model in [(LM_STUDIO_URL, MODEL_NAME), (LM_STUDIO_URL_2, MODEL_NAME_2), (LM_STUDIO_URL_3, MODEL_NAME_3)]:
        if url and model:
            LLM_ENDPOINTS.append({"url": url, "model": model, "api_key": "not-needed", "backend": "lmstudio"})

EMBEDDING_ENDPOINTS = []
if BACKENDS:
    for be in BACKENDS:
        if be.get("embeddings_model") and be.get("url"):
            EMBEDDING_ENDPOINTS.append({
                "url": be.get("url"),
                "model": be.get("embeddings_model", be.get("model")),
                "api_key": be.get("api_key", "not-needed"),
                "backend": be.get("backend", "lmstudio"),
            })
else:
    for url, model in [(LM_STUDIO_URL, EMBEDDING_MODEL), (LM_STUDIO_URL_2, EMBEDDING_MODEL_2), (LM_STUDIO_URL_3, EMBEDDING_MODEL_3)]:
        if url and model:
            EMBEDDING_ENDPOINTS.append({"url": url, "model": model, "api_key": "not-needed", "backend": "lmstudio"})

if not LLM_ENDPOINTS:
    LLM_ENDPOINTS = [{"url": LM_STUDIO_URL, "model": MODEL_NAME, "api_key": "not-needed"}]
if not EMBEDDING_ENDPOINTS:
    EMBEDDING_ENDPOINTS = [{"url": LM_STUDIO_URL, "model": EMBEDDING_MODEL, "api_key": "not-needed"}]

if not LLM_ENDPOINTS:
    raise RuntimeError("No LLM endpoints configured. Check LM_STUDIO_URL and MODEL_NAME.")
if not EMBEDDING_ENDPOINTS:
    raise RuntimeError("No embedding endpoints configured. Check LM_STUDIO_URL and EMBEDDING_MODEL.")

# Derive capacities from endpoint definitions
LLM_ENDPOINT_CAPACITIES = []
for _ep in LLM_ENDPOINTS:
    _cap = _ep.get("capacity")
    if _cap is None:
        _cap = 3 if len(LLM_ENDPOINT_CAPACITIES) == 0 else 1
    LLM_ENDPOINT_CAPACITIES.append(int(_cap))

API_RETRY_ATTEMPTS = 3
API_RETRY_BACKOFF = 2.0
API_TIMEOUT = 480
EMBEDDING_TIMEOUT = 240

USE_JSON_MODE = False

LLM_EXTRACTION_CACHE = True
LLM_CACHE_DB = "index"

LLM_BATCH_CHUNKS = 1
CHUNK_EXTRACTION_WORKERS = 3

NOVELTY_ENABLED = True
NOVELTY_SIM_THRESHOLD = 0.92
NOVELTY_NEW_ENTITY_RATIO = 0.0

FULL_DOC_PAGES = None
OCR_DPI = 80
OCR_LANG = "eng"  # OCR language(s), e.g., "eng+spa"
TITLE_PAGE_DPI = 100
TITLE_PAGE_COUNT = 3
MIN_TEXT_CHARS_FOR_OCR_SKIP = 200

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
MAX_CHUNKS_PER_LLM_CALL = LLM_BATCH_CHUNKS

EMBEDDING_BATCH_SIZE = int(os.environ.get('EMBEDDING_BATCH_SIZE', '32'))

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
CHAT_TOP_K_CHUNKS = int(os.environ.get("CHAT_TOP_K_CHUNKS", "8"))

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


# --- New settings for enhancements ---
# Conversation
CONVERSATION_MAX_TURNS = 15          # number of recent turns to include
CONVERSATION_SUMMARY_THRESHOLD = 30  # when to summarize older turns

# Topic extraction for chat retrieval
USE_ONNX_TOPIC_EXTRACTION = True    # use ONNX NER + rules to extract topic terms
USE_LLM_TOPIC_EXTRACTION = False    # optionally use a small LLM to refine topic terms

# Deep research
DEEP_RESEARCH_MAX_DEPTH = 7          # maximum graph expansion depth
DEEP_RESEARCH_MAX_SUBTOPICS = 8      # max subtopics per level
DEEP_RESEARCH_REPORT_DIR = str(DATA_DIR / "reports")
DEEP_RESEARCH_MIN_FACTS = 20         # minimum facts before report generation
DEEP_RESEARCH_MIN_CONFIDENCE = 0.6   # confidence threshold for inclusion

# Fast extractor (ONNX NER)
FAST_EXTRACTOR_ENABLED = True
FAST_EXTRACTOR_MODEL_NAME = "optimum/bert-base-NER"
FAST_EXTRACTOR_MODEL_DIR = str(BASE_DIR / "models" / "ner_onnx")
FAST_EXTRACTOR_CONFIDENCE_THRESHOLD = 0.7  # below this, LLM verification used
FAST_EXTRACTOR_LOW_CONFIDENCE_RATIO = 0.2   # max ratio of items to send to LLM
ONNX_DEVICE = os.environ.get("ONNX_DEVICE", "directml")  # "auto", "cpu", "cuda", "directml"

# Performance
RETRIEVAL_CACHE_ENABLED = True
RETRIEVAL_CACHE_TTL = 300             # seconds
EMBEDDING_BATCH_SIZE = 128             # increase for better throughput


# --- Optional model roles (leave empty to use default endpoints) ---
SMALL_MODEL_URL = os.environ.get("SMALL_MODEL_URL", "http://localhost:1234/v1")
SMALL_MODEL_NAME = os.environ.get("SMALL_MODEL_NAME", "liquidai/lfm2.5-1.2b-instruct")
SMALL_MODEL_URL_2 = os.environ.get("SMALL_MODEL_URL_2", "")
SMALL_MODEL_NAME_2 = os.environ.get("SMALL_MODEL_NAME_2", "liquidai/lfm2.5-1.2b-instruct")

SMALL_MODEL_ENDPOINT = None  # filled dynamically


CHAT_MODEL_URL = os.environ.get("CHAT_MODEL_URL", "")
CHAT_MODEL_NAME = os.environ.get("CHAT_MODEL_NAME", "")
USE_CHAT_MODEL = os.environ.get("USE_CHAT_MODEL", "false").lower() == "true"
LARGE_MODEL_URL = os.environ.get("LARGE_MODEL_URL", "")
LARGE_MODEL_NAME = os.environ.get("LARGE_MODEL_NAME", "")
LARGE_MODEL_ENDPOINT = None

AUDIT_MODEL_URL = os.environ.get("AUDIT_MODEL_URL", "")
AUDIT_MODEL_NAME = os.environ.get("AUDIT_MODEL_NAME", "")
AUDIT_MODEL_ENDPOINT = None

# --- Parallel processing ---
PARALLEL_PROCESSING_ENABLED = False
PARALLEL_WORKERS = 2

# --- Graph optimization ---
BATCH_GLOBAL_NODE_LOOKUPS = True  # use in-memory map for exact matches
EXTERNAL_GRAPH_CACHE_MAX_NODES = int(os.environ.get("EXTERNAL_GRAPH_CACHE_MAX_NODES", "100000"))

# Neural model configuration
RERANKER_ENABLED = True
RERANKER_MODEL_REPO = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_MODEL_DIR = str(BASE_DIR / "models" / "reranker")

LOCAL_EMBEDDER_ENABLED = True
LOCAL_EMBEDDER_MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_EMBEDDER_MODEL_DIR = str(BASE_DIR / "models" / "sentence_embedder")

INTENT_CLASSIFIER_ENABLED = True
INTENT_CLASSIFIER_MODEL_REPO = "valhalla/distilbart-mnli-12-3"
INTENT_CLASSIFIER_MODEL_DIR = str(BASE_DIR / "models" / "intent_classifier")

GNN_ENABLED = True
ENABLE_GNN_TRAINING = True
GNN_MODEL_DIR = str(BASE_DIR / "models" / "gnn")


# --- Enhanced Retrieval Ranking ---
RETRIEVAL_RANKING_WEIGHTS = {
    'query_overlap': 0.25,
    'rare_term_boost': 0.1,
    'semantic_similarity': 0.2,
    'graph_proximity': 0.1,
    'entity_salience': 0.05,
    'doc_relevance': 0.1,
    'type_weight': 0.1,
    'confidence': 0.1,
}
USE_MULTI_STAGE_RETRIEVAL = True

# --- Lazy Model Loading ---
LAZY_LOAD_MODELS = True  # If True, neural models load on first use.

# Retrieval fusion weights
RETRIEVAL_STAGE_WEIGHTS = {
    "graph": 0.5,
    "vector": 0.3,
    "lexical": 0.2,
}

# Hierarchical datapoint map retrieval
DATAPOINT_SCORE_WEIGHTS = {
    "query_overlap": 0.35,
    "graph_proximity": 0.25,
    "semantic_similarity": 0.25,
    "confidence": 0.15,
}
MAX_MAP_NODES = 200
MAX_SELECTED_NODES = 15
EXPANSION_DEPTH = 2
ENABLE_QUOTE_STORAGE = True

# Batch verification (optional, disabled by default)
ENABLE_BATCH_VERIFICATION = os.environ.get("ENABLE_BATCH_VERIFICATION", "false").lower() == "true"

# Statistical keyword extraction
ENABLE_STATISTICAL_KEYWORDS = True

# Optional advanced features (disabled by default)
ENABLE_BATCH_VERIFICATION = False
ENABLE_CALIBRATED_ARES = False
ENABLE_GRAPH_KEYWORD_PAGERANK = False
ENABLE_TEMPORAL_KEYWORDS = False
ENABLE_LOGIC_LEARNING_FROM_PATHS = False
ENABLE_COMMUNITY_DETECTION = False

# --- Adaptive verification ---
ADAPTIVE_VERIFICATION = True
VERIFICATION_ESCALATION_THRESHOLD = 0.6  # if initial confidence below this, escalate

# --- Contradiction handling ---
AUTO_RESOLVE_CONTRADICTIONS = False  # if False, move to review queue instead of deleting

# --- Progress bars ---
USE_PROGRESS_BARS = True
TQDM_AVAILABLE = False  # will be set at runtime

# --- Retry fallback ---
LLM_BATCH_RETRY_SINGLE = True  # on batch failure, retry each chunk individually

# --- Deep research enhancements ---
DEEP_RESEARCH_INTERACTIVE = True  # ask user to continue on subtopics
DEEP_RESEARCH_AUTO_SUBTOPIC_DEPTH = 2

# --- Audit model selection ---
AUDIT_MODEL = None  # will be set from AUDIT_MODEL_ENDPOINT if exists


# Fill small/large/audit endpoints if specified
for _endpoint in LLM_ENDPOINTS:
    if SMALL_MODEL_URL and SMALL_MODEL_NAME:
        SMALL_MODEL_ENDPOINT = {"url": SMALL_MODEL_URL, "model": SMALL_MODEL_NAME, "api_key": "not-needed"}
    if LARGE_MODEL_URL and LARGE_MODEL_NAME:
        LARGE_MODEL_ENDPOINT = {"url": LARGE_MODEL_URL, "model": LARGE_MODEL_NAME, "api_key": "not-needed"}
    if AUDIT_MODEL_URL and AUDIT_MODEL_NAME:
        AUDIT_MODEL_ENDPOINT = {"url": AUDIT_MODEL_URL, "model": AUDIT_MODEL_NAME, "api_key": "not-needed"}
if SMALL_MODEL_ENDPOINT and SMALL_MODEL_ENDPOINT not in LLM_ENDPOINTS:
    LLM_ENDPOINTS.append(SMALL_MODEL_ENDPOINT)
if LARGE_MODEL_ENDPOINT and LARGE_MODEL_ENDPOINT not in LLM_ENDPOINTS:
    LLM_ENDPOINTS.append(LARGE_MODEL_ENDPOINT)
if AUDIT_MODEL_ENDPOINT and AUDIT_MODEL_ENDPOINT not in LLM_ENDPOINTS:
    LLM_ENDPOINTS.append(AUDIT_MODEL_ENDPOINT)


# Build second small model endpoint
if SMALL_MODEL_URL_2 and SMALL_MODEL_NAME_2:
    SMALL_MODEL_ENDPOINT_2 = {"url": SMALL_MODEL_URL_2, "model": SMALL_MODEL_NAME_2, "api_key": "not-needed"}
    if SMALL_MODEL_ENDPOINT_2 not in LLM_ENDPOINTS:
        LLM_ENDPOINTS.append(SMALL_MODEL_ENDPOINT_2)

# Build chat model endpoint
if USE_CHAT_MODEL and CHAT_MODEL_URL and CHAT_MODEL_NAME:
    CHAT_MODEL_ENDPOINT = {"url": CHAT_MODEL_URL, "model": CHAT_MODEL_NAME, "api_key": "not-needed"}
    if CHAT_MODEL_ENDPOINT not in LLM_ENDPOINTS:
        LLM_ENDPOINTS.append(CHAT_MODEL_ENDPOINT)

USE_LLM_HTTP_SESSION = os.environ.get('USE_LLM_HTTP_SESSION', 'true').lower() == 'true'

FTS_ENABLED = os.environ.get('FTS_ENABLED', 'true').lower() == 'true'


# Comprehensive upgrade settings
USE_ASYNC_LLM = False  # currently synchronous; async not implemented
INCLUDE_HYPERGRAPH_IN_EXPANSION = True
MEMORY_DECAY_ENABLED = True
MEMORY_DECAY_FACTOR = 0.0001  # decay per second (approx half-life ~2h)
FTS_ENABLED = True
USE_LLM_HTTP_SESSION = True
PARALLEL_PROCESSING_ENABLED = False
PARALLEL_WORKERS = 2
ADAPTIVE_VERIFICATION = True
VERIFICATION_ESCALATION_THRESHOLD = 0.6
AUTO_RESOLVE_CONTRADICTIONS = False
USE_PROGRESS_BARS = True
TQDM_AVAILABLE = True


# Advanced settings
USE_ASYNC_LLM = True   # Set True if aiohttp installed and LM Studio supports async
INCREMENTAL_EXTRACTION = False  # Track per-chunk extraction state to allow resume
LOGIC_EXECUTOR_ENABLED = True  # Use logic modules to influence query processing
REPORT_COHERENCE_PASS = True   # Run final coherence check on deep research reports


# --- Recoll integration ---
USE_RECOLL = os.environ.get("USE_RECOLL", "false").lower() == "true"
RECOLL_CONFDIR = os.environ.get("RECOLL_CONFDIR", "")
RECOLL_EXTRA_DBS = os.environ.get("RECOLL_EXTRA_DBS", "")  # space-separated list
RECOLL_DEFAULT_LIMIT = int(os.environ.get("RECOLL_DEFAULT_LIMIT", "20"))
RECOLL_MAX_ROUNDS = int(os.environ.get("RECOLL_MAX_ROUNDS", "10"))
RECOLL_INTERACTIVE = os.environ.get("RECOLL_INTERACTIVE", "false").lower() == "true"

# Recoll Fast Mode
RECOLL_BIN = os.environ.get("RECOLL_BIN", "recollq")
RECOLL_DB = os.environ.get("RECOLL_DB", "")  # empty = use Recoll default
RECOLL_MAX_RESULTS = int(os.environ.get("RECOLL_MAX_RESULTS", "50"))
PREVIEW_CHAR_WINDOW = int(os.environ.get("PREVIEW_CHAR_WINDOW", "1000"))
PREVIEW_PAGE_WINDOW = int(os.environ.get("PREVIEW_PAGE_WINDOW", "1"))
RECOLL_FAST_LLM_BATCH_CHUNKS = int(os.environ.get("RECOLL_FAST_LLM_BATCH_CHUNKS", "4"))

RECOLL_AUTO_KEYWORD_LIMIT = int(os.environ.get("RECOLL_AUTO_KEYWORD_LIMIT", "20"))

# Additional small model (second worker)
SMALL_MODEL_ENDPOINT_2 = None

# Dedicated chat model (optional)
CHAT_MODEL_ENDPOINT = None

# Async validation queue (small models extract, large models validate)
ENABLE_ASYNC_VALIDATION = os.environ.get("ENABLE_ASYNC_VALIDATION", "true").lower() == "true"
VALIDATION_QUEUE_SIZE = int(os.environ.get("VALIDATION_QUEUE_SIZE", "200"))
VALIDATION_BATCH_SIZE = int(os.environ.get("VALIDATION_BATCH_SIZE", "5"))
VALIDATION_WORKERS = int(os.environ.get("VALIDATION_WORKERS", "2"))
VALIDATION_MODEL_GROUP = os.environ.get("VALIDATION_MODEL_GROUP", "large")  # or "main"
EXTRACTION_MODEL_GROUP = os.environ.get("EXTRACTION_MODEL_GROUP", "small")  # or "main"

# --- Phase 3 Settings ---
USE_GNN = True
GNN_MODEL_DIR = str(BASE_DIR / "models" / "gnn")
GNN_EMBEDDING_DIM = 64
USE_VERIFIED_CHAT = True
PARALLEL_INGESTION = True
PARALLEL_INGESTION_WORKERS = 4


# --- Phase 4 Settings ---
ENABLE_SEMANTIC_CONTRADICTIONS = True
ACTIVE_LEARNING_ENABLED = True
ACTIVE_LEARNING_ROUNDS = 3


# --- Phase 5 Settings ---
ENABLE_JSON_LOGGING = True
METRICS_ENABLED = True
