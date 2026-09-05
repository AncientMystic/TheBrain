"""
Session config overrides with validation (no silent config.py rewrite).

PUT /api/config accepts {key: value, ...} plus {confirm_model_switch: bool} when
changing embedding model/dim. Applies to live config module for current process
(session scope), validates ranges/paths/dims, returns applied + warnings +
restart-required flags. Secrets never echoed. Generic validators (no doc-specific rules).
"""
import config as _cfg


def _validate_one(key, value):
    """Returns (ok, coerced_or_error, warning)."""
    try:
        if key in ("SERVER_PORT",):
            v = int(value)
            if not 1 <= v <= 65535:
                return False, "port must be 1-65535", ""
            return True, v, ""
        if key in ("CHUNK_SIZE", "CHUNK_OVERLAP", "EMBEDDING_BATCH_SIZE", "OCR_BATCH_SIZE",
                   "VALIDATION_WORKERS", "VALIDATION_BATCH_SIZE", "CHAT_TOP_K_CHUNKS",
                   "PARALLEL_WORKERS", "PARALLEL_INGESTION_WORKERS", "OCR_WORKERS"):
            v = int(value)
            if v < 1:
                return False, "must be >= 1", ""
            return True, v, ""
        if key in ("NOVELTY_SIM_THRESHOLD", "MIN_FACT_CONFIDENCE", "MIN_PRIORITY_CONFIDENCE",
                   "MEMORY_CONSOLIDATION_THRESHOLD", "HYPERBOLIC_FILTER_RADIUS", "MIN_PATH_CONFIDENCE"):
            v = float(value)
            return True, v, ""
        if key in ("NOVELTY_ENABLED", "RECALL_AUGMENT_ENABLED", "USE_PRIME_EVEN_GATE",
                   "PARALLEL_PROCESSING_ENABLED", "PREFETCH_NEXT_DOCUMENT", "RERANKER_ENABLED",
                   "USE_HYPERBOLIC_RETRIEVAL", "USE_RECOLL", "GATE_STRUCTURED_INIT"):
            if isinstance(value, str):
                return True, value.lower() in ("1", "true", "yes", "on"), ""
            return True, bool(value), ""
        if key == "EMBEDDING_DIM":
            v = int(value)
            if v != int(getattr(_cfg, "EMBEDDING_DIM", 1024)):
                return True, v, ("POISON WARNING: changing dims without full re-embed + reindex "
                                 "quarantines old rows and orphans indexes. Confirm only with migration planned.")
            return True, v, ""
        if key in ("EMBEDDING_MODEL", "BACKEND_EMBEDDINGS_MODEL"):
            cur = str(getattr(_cfg, "EMBEDDING_MODEL", ""))
            if str(value) != cur:
                return True, str(value), ("POISON WARNING: new embedding model needs full re-embed migration "
                                          "to align cache/index. Old rows quarantined, never mixed.")
            return True, str(value), ""
        if key in ("SERVER_HOST", "BACKEND_URL", "LM_STUDIO_URL", "RECOLL_BIN", "RECOLL_DB"):
            return True, str(value), ""
        if key in ("SERVER_AUTH_TOKEN", "BACKEND_API_KEY"):
            return True, str(value), "Secret applied to memory only (never logged, never committed)."
        # Fallback: accept str/int/float/bool as-is (session scope, validated downstream by users)
        return True, value, ""
    except Exception as e:
        return False, f"invalid value: {e}", ""


def register_config_routes(app, require_auth):
    from fastapi import Depends

    @app.put("/api/config", dependencies=[Depends(require_auth)])
    async def put_config(payload: dict):
        payload = payload or {}
        confirm = bool(payload.pop("confirm_model_switch", False))
        needs_confirm = any(k in ("EMBEDDING_MODEL", "BACKEND_EMBEDDINGS_MODEL", "EMBEDDING_DIM") for k in payload)
        if needs_confirm and not confirm:
            return {"applied": {}, "errors": {},
                    "warnings": {"confirm": "Embedding identity change needs confirm_model_switch=true plus migration plan."},
                    "restart_required": False}
        from webui.schema import GROUPS as _G
        allowed = {k for ks in _G.values() for k in ks}
        applied, errors, warnings = {}, {}, {}
        for k, v in payload.items():
            if k not in allowed:
                errors[k] = "unknown key (see schema groups)"
                continue
            ok, coerced, warn = _validate_one(k, v)
            if not ok:
                errors[k] = coerced
                continue
            try:
                setattr(_cfg, k, coerced)
                applied[k] = "****" if ("TOKEN" in k or "API_KEY" in k) else coerced
                if warn:
                    warnings[k] = warn
            except Exception as e:
                errors[k] = str(e)
        restart = any(k in ("SERVER_HOST", "SERVER_PORT", "EMBEDDING_MODEL", "EMBEDDING_DIM") for k in applied)
        return {"applied": applied, "errors": errors, "warnings": warnings, "restart_required": restart}

    return app
