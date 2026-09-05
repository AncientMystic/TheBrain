"""
Model context detection + prompt budgeting (generic, no model-specific tables).

Answers must fit the model that will actually serve them. Routing is
round-robin across pools, so no single call site can know its endpoint ahead
of time — the safe bound is the MINIMUM context across the pool that serves
answers (typically one shared local model, so the bound costs nothing).
Detection order per endpoint: explicit config override, live backend probe,
conservative fallback. Never raises, never blocks ingestion on failure.
"""
import logging
import threading
import time

logger = logging.getLogger(__name__)

_CACHE = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL = 3600

# Candidate metadata fields that may carry a context length, any backend shape.
_CONTEXT_KEYS = ("max_context_length", "context_length", "max_model_len",
                 "context_window", "num_ctx", "n_ctx", "max_tokens")


def _plausible(value):
    try:
        v = int(value)
    except Exception:
        return None
    return v if 1024 <= v <= 10_000_000 else None


def _search_mapping(mapping):
    """Depth-first scan (max 3 levels) for plausible context-length fields."""
    try:
        stack = [(mapping, 0)]
        while stack:
            obj, depth = stack.pop()
            if not isinstance(obj, dict) or depth > 3:
                continue
            for k, v in obj.items():
                try:
                    if isinstance(k, str) and k.lower() in _CONTEXT_KEYS:
                        hit = _plausible(v)
                        if hit:
                            return hit
                except Exception:
                    continue
                if isinstance(v, dict):
                    stack.append((v, depth + 1))
                elif isinstance(v, list):
                    for item in v[:8]:
                        if isinstance(item, dict):
                            stack.append((item, depth + 1))
    except Exception:
        pass
    return None


def _probe_openai_compatible(base_url, model):
    """GET {base}/models and match model id. Works for LM Studio, Ollama
    OpenAI mode, KoboldCpp and generic OpenAI-compatible servers."""
    import requests
    base = (base_url or "").rstrip("/")
    if not base:
        return None
    resp = requests.get(f"{base}/models", timeout=3)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("data", []) if isinstance(data, dict) else []
    want = str(model or "").lower()
    pool = [it for it in items if isinstance(it, dict)]
    ranked = sorted(pool, key=lambda it: 0 if str(it.get("id", "")).lower() in (want, want.split("/")[-1]) else 1)
    for item in ranked[:4]:
        hit = _search_mapping(item)
        if hit:
            return hit
    return None


def _probe_ollama_native(base_url, model):
    """POST /api/show for native Ollama servers (num_ctx / model_info)."""
    import requests
    base = (base_url or "").rstrip("/")
    if not base or not model:
        return None
    resp = requests.post(f"{base}/api/show", json={"name": model}, timeout=3)
    resp.raise_for_status()
    return _search_mapping(resp.json())


def detect_endpoint_context(endpoint):
    """Return detected context tokens for one endpoint dict, or None."""
    try:
        import config as _cfg
        overrides = getattr(_cfg, "MODEL_CONTEXT_JSON", None) or {}
        model = str((endpoint or {}).get("model", ""))
        if model in overrides:
            hit = _plausible(overrides[model])
            if hit:
                return hit
        single = int(getattr(_cfg, "MODEL_MAX_CONTEXT", 0) or 0)
        if single > 0:
            return single
    except Exception:
        pass
    base = str((endpoint or {}).get("url", ""))
    model = str((endpoint or {}).get("model", ""))
    key = (base, model)
    now = time.time()
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit and now - hit[1] < _CACHE_TTL:
            return hit[0]
    detected = None
    for probe in (_probe_openai_compatible, _probe_ollama_native):
        try:
            detected = probe(base, model)
            if detected:
                break
        except Exception:
            continue
    with _CACHE_LOCK:
        _CACHE[key] = (detected, now)
    return detected


def pool_min_context(endpoints, fallback=None):
    """Minimum detected context across endpoints (the safe generation bound).

    Endpoints that fail probing fall back individually so one offline server
    cannot drag the bound to zero; a total detection failure returns the
    configured fallback instead of blocking the request.
    """
    try:
        import config as _cfg
        fb = int(fallback if fallback is not None else getattr(_cfg, "MODEL_FALLBACK_CONTEXT", 8192))
    except Exception:
        fb = 8192
    eps = list(endpoints or [])
    if not eps:
        return fb
    # Parallel probes so many-endpoint pools don't stall startup/requests.
    try:
        from concurrent.futures import ThreadPoolExecutor as _TPE
        with _TPE(max_workers=min(len(eps), 4)) as _ex:
            detected = list(_ex.map(detect_endpoint_context, eps))
    except Exception:
        detected = [detect_endpoint_context(ep) for ep in eps]
    good = [d for d in detected if d]
    if not good:
        return fb
    # Any endpoint that failed probing is assumed to share the pool's minimum
    # rather than zero — it was healthy enough to be configured.
    floor = min(good)
    return floor


def answer_pool_min_context():
    """Bound for answer generation: minimum over the chat pool."""
    try:
        import config as _cfg
        from core.model_router import get_endpoint_for_group  # noqa: F401 (warms pools)
        eps = list(getattr(_cfg, "LLM_ENDPOINTS", []))
        # Chat pool falls back to main endpoints when no dedicated chat model.
        chat = []
        if getattr(_cfg, "USE_CHAT_MODEL", False) and getattr(_cfg, "CHAT_MODEL_URL", "") and getattr(_cfg, "CHAT_MODEL_NAME", ""):
            chat = [{"url": _cfg.CHAT_MODEL_URL, "model": _cfg.CHAT_MODEL_NAME}]
        return pool_min_context(chat or eps)
    except Exception:
        return 8192


def answer_budget(answer_reserve_tokens=None, prompt_overhead_tokens=None):
    """(budget_chars, label) pair for the current answer pool (single probe set)."""
    try:
        window = int(answer_pool_min_context())
    except Exception:
        window = 8192
    return answer_budget_chars(answer_reserve_tokens, prompt_overhead_tokens), f"~{window}-token window"


def answer_budget_chars(answer_reserve_tokens=None, prompt_overhead_tokens=None):
    """Usable context characters for retrieved material under the pool bound.

    Reserves answer + overhead tokens first, converts the remainder at ~4
    chars/token, floored so tiny windows still function. Callers fit facts
    first (highest signal), then excerpts, and note anything omitted.
    """
    try:
        import config as _cfg
        window = int(answer_pool_min_context())
        reserve = int(answer_reserve_tokens if answer_reserve_tokens is not None
                      else getattr(_cfg, "ANSWER_RESERVE_TOKENS", 2048))
        overhead = int(prompt_overhead_tokens if prompt_overhead_tokens is not None
                       else getattr(_cfg, "PROMPT_OVERHEAD_TOKENS", 1000))
        per_token = float(getattr(_cfg, "CHARS_PER_TOKEN", 4.0))
    except Exception:
        window, reserve, overhead, per_token = 8192, 2048, 1000, 4.0
    usable = max(0, window - reserve - overhead)
    return max(2000, int(usable * per_token))
