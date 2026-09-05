"""
Config schema introspection for WebUI (generated, not hand-maintained).

Exposes name, value, env var, type, group + one-line docs so users know what
each toggle does. Safe to extend: add entries to DOCS for new keys.
"""
import config as _cfg

GROUPS = {
    "server": ["SERVER_HOST", "SERVER_PORT", "SERVER_AUTH_TOKEN", "CORS_ORIGINS"],
    "backends": ["BACKEND_TYPE", "BACKEND_URL", "BACKEND_MODEL", "BACKEND_EMBEDDINGS_MODEL", "BACKEND_API_KEY", "BACKEND_CONFIG_JSON"],
    "chunking": ["CHUNK_SIZE", "CHUNK_OVERLAP", "LLM_BATCH_CHUNKS", "LLM_BATCH_SMALL", "LLM_BATCH_LARGE", "LLM_BATCH_MAX", "CHUNK_EXTRACTION_WORKERS"],
    "embeddings": ["EMBEDDING_MODEL", "EMBEDDING_DIM", "EMBEDDING_BATCH_SIZE", "EMBEDDING_CACHE_TTL", "EMBEDDING_CACHE_MAXSIZE", "EMBEDDING_QUANT", "LOCAL_EMBEDDER_MODEL_REPO"],
    "novelty": ["NOVELTY_ENABLED", "NOVELTY_SIM_THRESHOLD", "NOVELTY_MIN_KEEP_RATIO", "NOVELTY_MIN_KEEP_COUNT"],
    "recall": ["RECALL_AUGMENT_ENABLED", "RECALL_PRIORITY_THRESHOLD", "RECALL_PRIORITY_WEIGHTS", "RECALL_MAX_QUERY_CHARS"],
    "validation": ["MIN_FACT_CONFIDENCE", "MIN_PRIORITY_CONFIDENCE", "VALIDATION_WORKERS", "VALIDATION_BATCH_SIZE", "VALIDATION_MAX_PER_CALL", "VALIDATION_TIMEOUT"],
    "retrieval": ["CHAT_TOP_K_CHUNKS", "CHAT_MAX_CONTEXT_TOKENS", "RERANKER_ENABLED", "USE_HYPERBOLIC_RETRIEVAL"],
    "recoll": ["USE_RECOLL", "RECOLL_BIN", "RECOLL_DB", "RECOLL_MAX_RESULTS", "RECOLL_HARD_MAX_RESULTS", "RECOLL_TIMEOUT", "RECOLL_MAX_QUERY_CHARS"],
    "ocr": ["OCR_DPI", "OCR_LANG", "OCR_BATCH_SIZE", "OCR_WORKERS", "MIN_TEXT_CHARS_FOR_OCR_SKIP"],
    "gates": ["USE_PRIME_EVEN_GATE", "GATE_STRUCTURED_INIT", "GATE_BETA_PRIME", "GATE_GAMMA_EVEN", "GATE_DELTA_PRIME", "GATE_DELTA_ANCHOR", "GATE_LAM1", "GATE_LAM2", "GATE_LAM3", "GATE_LAM4", "GATE_LR", "GATE_BATCH_SIZE"],
    "ingestion": ["PARALLEL_PROCESSING_ENABLED", "PARALLEL_WORKERS", "PARALLEL_INGESTION_WORKERS", "PREFETCH_NEXT_DOCUMENT", "PREFETCH_DEPTH", "GC_EVERY_N_FILES", "GC_MEM_MB"],
}

DOCS = {
    "SERVER_HOST": "Bind address (127.0.0.1 local only, 0.0.0.0 LAN).",
    "SERVER_PORT": "API port.",
    "SERVER_AUTH_TOKEN": "Bearer token required for /v1/* and /api/* (empty = open, warning on LAN).",
    "CORS_ORIGINS": "Allowed browser origins, comma-separated (empty = none).",
    "BACKEND_TYPE": "Provider: lmstudio, ollama, koboldcpp, openai_compatible.",
    "BACKEND_URL": "Base URL of main backend.",
    "BACKEND_MODEL": "Main LLM model name.",
    "BACKEND_EMBEDDINGS_MODEL": "Embedding model (must stay 1024-dim mxbai or re-embed all).",
    "BACKEND_API_KEY": "API key for cloud backends (kept in memory, never logged).",
    "BACKEND_CONFIG_JSON": "JSON array for multiple backends.",
    "CHUNK_SIZE": "Chars per document chunk (full-doc, all chunks processed).",
    "CHUNK_OVERLAP": "Overlap chars to avoid boundary loss.",
    "LLM_BATCH_CHUNKS": "Chunks per LLM call (small-safe default; larger overflows small models).",
    "LLM_BATCH_SMALL": "Batch for small local models.",
    "LLM_BATCH_LARGE": "Batch for large models.",
    "LLM_BATCH_MAX": "Hard cap for any batch.",
    "CHUNK_EXTRACTION_WORKERS": "Parallel chunk batches (respects endpoint capacities).",
    "EMBEDDING_MODEL": "Embedding model id (changing dims poisons cache — see warning).",
    "EMBEDDING_DIM": "Contract: all endpoints + stored blobs must match (1024 mxbai). Mismatch quarantined, never mixed.",
    "EMBEDDING_BATCH_SIZE": "Texts per embedding HTTP batch.",
    "EMBEDDING_CACHE_TTL": "In-memory cache seconds.",
    "EMBEDDING_CACHE_MAXSIZE": "Max cached batches (LRU).",
    "EMBEDDING_QUANT": "float32 exact or float16 halves RAM (DB stays float32).",
    "LOCAL_EMBEDDER_MODEL_REPO": "Local ONNX mirror id (must match 1024).",
    "NOVELTY_ENABLED": "Skip near-duplicate chunks (saves calls, floor guarantees coverage).",
    "NOVELTY_SIM_THRESHOLD": "Similarity above which chunk is redundant (needs new entities to survive).",
    "NOVELTY_MIN_KEEP_RATIO": "Safety floor: keep at least this fraction (most-distant first).",
    "NOVELTY_MIN_KEEP_COUNT": "Safety floor: keep at least this many chunks.",
    "RECALL_AUGMENT_ENABLED": "Scan DBs in fast pass for topics/dates/refs/events (no LLM).",
    "RECALL_PRIORITY_THRESHOLD": "Score at/above which chunk is priority (must-extract + must-verify).",
    "RECALL_PRIORITY_WEIGHTS": "Dict of signal weights (entity/topic/date/event/standards/contradiction).",
    "RECALL_MAX_QUERY_CHARS": "Max chars for generated recall queries.",
    "MIN_FACT_CONFIDENCE": "Lenient triage floor (verifier decides truth, not this).",
    "MIN_PRIORITY_CONFIDENCE": "Lower floor for priority anchors.",
    "VALIDATION_WORKERS": "Parallel validation batches (capped by endpoint capacities).",
    "VALIDATION_BATCH_SIZE": "Items per validation batch.",
    "VALIDATION_MAX_PER_CALL": "LLM items per call (small for JSON reliability).",
    "VALIDATION_TIMEOUT": "Seconds per validation call.",
    "CHAT_TOP_K_CHUNKS": "Chunks per answer (pagination, not truncation of corpus).",
    "CHAT_MAX_CONTEXT_TOKENS": "Max assembled context tokens.",
    "RERANKER_ENABLED": "Cross-encoder rerank on/off.",
    "USE_HYPERBOLIC_RETRIEVAL": "Distance in Poincare ball (on) vs cosine fallback (off, not recommended).",
    "USE_RECOLL": "Enable Recoll full-text source.",
    "RECOLL_BIN": "recollq binary (resolved via PATH, shell=False).",
    "RECOLL_DB": "Recoll confdir (absolute).",
    "RECOLL_MAX_RESULTS": "Default result cap.",
    "RECOLL_HARD_MAX_RESULTS": "Hard cap for any query.",
    "RECOLL_TIMEOUT": "Seconds per recollq call.",
    "OCR_DPI": "Scan resolution for OCR pages.",
    "OCR_LANG": "Tesseract languages.",
    "OCR_BATCH_SIZE": "Pages per OCR batch.",
    "OCR_WORKERS": "Parallel OCR processes (order-preserving).",
    "MIN_TEXT_CHARS_FOR_OCR_SKIP": "Skip OCR when extracted text longer than this.",
    "USE_PRIME_EVEN_GATE": "Triage gate on/off (off = all chunks to LLM, slower).",
    "GATE_STRUCTURED_INIT": "Prime-supported init (on) vs diffuse random (off, slower).",
    "GATE_BETA_PRIME": "Init mass on prime indices.",
    "GATE_GAMMA_EVEN": "Init mass on even gaps.",
    "GATE_DELTA_PRIME": "Init mass on prime unitary loadings.",
    "GATE_DELTA_ANCHOR": "Init mass on index-2 anchor.",
    "GATE_LAM1": "Sparsity weight.",
    "GATE_LAM2": "Prime-pull weight.",
    "GATE_LAM3": "Even-pull weight.",
    "GATE_LAM4": "Anchor weight (default lam2/8).",
    "GATE_LR": "Proximal step size.",
    "GATE_BATCH_SIZE": "Rows per gate training batch.",
    "PARALLEL_PROCESSING_ENABLED": "Process files in parallel (2 workers, independent + deterministic).",
    "PARALLEL_WORKERS": "File workers (2 safe with WAL + pools).",
    "PARALLEL_INGESTION_WORKERS": "Ingestion pool size (overrides workers when set).",
    "PREFETCH_NEXT_DOCUMENT": "Overlap next-file extract/embed with current LLM (depth 1, bounded RAM).",
    "PREFETCH_DEPTH": "How many files ahead to prepare (1 safe).",
    "GC_EVERY_N_FILES": "Full GC cadence (no per-file stall).",
    "GC_MEM_MB": "GC threshold hint for future memory-pressure mode.",
}


def _kind(name, value):
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, (list, dict)):
        return "json"
    return "str"


def get_schema():
    items = []
    for group, keys in GROUPS.items():
        for k in keys:
            v = getattr(_cfg, k, None)
            # Hide secret values (show presence only)
            if "TOKEN" in k or "API_KEY" in k:
                shown = "****" if v else ""
            else:
                shown = v
            items.append({"key": k, "group": group, "value": shown, "kind": _kind(k, v),
                          "doc": DOCS.get(k, "")})
    return {"groups": list(GROUPS.keys()), "items": items}
