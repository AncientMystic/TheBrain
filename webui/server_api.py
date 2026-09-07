"""
Server and health tab backend reusing existing checks (no duplicated logic).

- endpoints: per-LLM-endpoint latency via provider health checks (bounded, timed)
- metrics: tail of Prometheus-style metrics endpoint (capped lines)
- dbs: SQLite file sizes + index freshness (no content reads)
All read-only except health ping, shared auth, generic.
"""
import time


def register_server_routes(app, require_auth):
    from fastapi import Depends

    @app.get("/api/server/endpoints", dependencies=[Depends(require_auth)])
    async def server_endpoints():
        import config as _cfg
        out = []
        for ep in getattr(_cfg, "LLM_ENDPOINTS", []):
            t0 = time.time()
            loaded = None
            try:
                from core.backends import create_backend
                provider = create_backend(ep)
                ok = bool(provider.health_check())
                try:
                    resident = provider.loaded_models() if hasattr(provider, "loaded_models") else None
                    if resident is not None:
                        want = str(ep.get("model", "")).lower()
                        loaded = any(want and (want in str(r).lower() or str(r).lower().startswith(want.split(":")[0] + ":")) for r in resident)
                except Exception:
                    pass
            except Exception as e:
                ok = False
                err = str(e)[:150]
            else:
                err = ""
            out.append({"url": ep.get("url", ""), "model": ep.get("model", ""),
                        "backend": ep.get("backend", ""), "ok": ok, "loaded": loaded,
                        "ms": int((time.time() - t0) * 1000), "error": err})
        return {"endpoints": out}

    @app.get("/api/server/metrics", dependencies=[Depends(require_auth)])
    async def server_metrics(lines: int = 60):
        lines = max(1, min(int(lines or 60), 200))
        try:
            from core.metrics import get_all_metrics
            text = get_all_metrics() or ""
            tail = text.splitlines()[-lines:]
            return {"metrics": "\n".join(tail)}
        except Exception as e:
            return {"metrics": "", "error": str(e)[:200]}

    @app.get("/api/server/overview", dependencies=[Depends(require_auth)])
    async def server_overview():
        """Single-call dashboard payload: health, endpoints, DB sizes,
        corpus counts (cheap COUNTs only) + parsed metric stats.

        Powers the futuristic dashboard tab with one round-trip; every
        source below reuses existing helpers (no duplicated logic).
        """
        import config as _cfg
        out = {"ok": True}
        try:
            out["llm_endpoints"] = len(getattr(_cfg, "LLM_ENDPOINTS", []))
            out["embedding_model"] = str(getattr(_cfg, "EMBEDDING_MODEL", ""))
            out["embedding_dim"] = int(getattr(_cfg, "EMBEDDING_DIM", 1024))
        except Exception:
            out["llm_endpoints"] = 0
        # Corpus counts (indexed COUNTs, no content reads)
        counts = {}
        try:
            from core import db as _db
            for _alias, _sql in (
                ("facts", "SELECT COUNT(*) AS n FROM key_facts"),
                ("chunks", "SELECT COUNT(*) AS n FROM document_chunks"),
                ("docs", "SELECT COUNT(*) AS n FROM documents"),
                ("memories", "SELECT COUNT(*) AS n FROM memory_entries"),
                ("logic_modules", "SELECT COUNT(*) AS n FROM logic_modules"),
            ):
                try:
                    _c = _db.db_connect("key_facts" if _alias == "facts" else
                                        "index" if _alias in ("chunks", "docs") else
                                        "memories" if _alias == "memories" else "logic")
                    _row = _c.execute(_sql).fetchone()
                    counts[_alias] = int(_row["n"]) if _row else 0
                    _c.close()
                except Exception:
                    counts[_alias] = 0
        except Exception:
            pass
        out["counts"] = counts
        # Verification distribution (trust at a glance, warrant colors).
        try:
            _c = _db.db_connect("key_facts")
            _dist, _act = {}, []
            try:
                for _row in _c.execute(
                        "SELECT verification_status, COUNT(*) AS n FROM key_facts GROUP BY verification_status"):
                    _dist[str(_row[0] or "unverified")] = int(_row[1] or 0)
            except Exception:
                pass
            try:
                _ci = _db.db_connect("index")
                _docs = _ci.execute(
                    "SELECT file_hash, filename, updated_at FROM documents ORDER BY updated_at DESC LIMIT 8"
                ).fetchall()
                _ci.close()
                for _d in _docs:
                    try:
                        _n = _c.execute("SELECT COUNT(*) AS n FROM key_facts WHERE doc_hash=?",
                                        (_d[0],)).fetchone()
                        _nn = int(_n["n"]) if _n else 0
                    except Exception:
                        _nn = 0
                    _act.append({"name": str(_d[1] or "?")[:80],
                                 "at": str(_d[2] or ""),
                                 "facts": _nn})
            except Exception:
                pass
            _c.close()
            out["verification"] = _dist
            out["activity"] = _act
        except Exception:
            out["verification"] = {}
            out["activity"] = []
        # Open breaker circuits (reliability pills).
        try:
            from core.breaker import open_circuits as _brk_open
            out["breakers"] = _brk_open()
        except Exception:
            out["breakers"] = []
        # Metric stats: counters raw + histogram count/avg/p95 (capped)
        stats = {"counters": {}, "timings": {}}
        try:
            from core.metrics import get_all_metrics
            _cur = None
            for _line in (get_all_metrics() or "").splitlines():
                _line = _line.strip()
                if not _line or _line.startswith("#"):
                    continue
                _parts = _line.split()
                if len(_parts) != 2:
                    continue
                _name, _val = _parts
                try:
                    _f = float(_val)
                except Exception:
                    continue
                if _name.endswith("_bucket") or _name.endswith("_sum"):
                    continue
                stats["counters"][_name] = _f
            # Derive timing summaries from known histogram families
            try:
                from core.metrics import get_histogram
                import math as _math
                for _h in ("extraction_duration_seconds", "chat_duration_seconds",
                           "retrieval_duration_seconds", "embedding_duration_seconds"):
                    try:
                        _vals = sorted(get_histogram(_h))
                    except Exception:
                        continue
                    if not _vals:
                        continue
                    _vals = _vals[-500:]
                    _n = len(_vals)
                    stats["timings"][_h] = {
                        "n": _n, "avg": round(sum(_vals) / _n, 2),
                        "p95": round(_vals[min(_n - 1, int(_n * 0.95))], 2),
                        "max": round(_vals[-1], 2),
                    }
            except Exception:
                pass
        except Exception as e:
            stats["error"] = str(e)[:200]
        out["metrics"] = stats
        return out

    @app.get("/api/server/dbs", dependencies=[Depends(require_auth)])
    async def server_dbs():
        import os as _os
        try:
            from core import db as _db
            out = []
            for name, path in _db.DB_FILES.items():
                try:
                    st = _os.stat(path) if path and _os.path.exists(path) else None
                    out.append({"name": name, "mb": round(st.st_size / 1048576, 1) if st else 0.0,
                                "mtime": int(st.st_mtime) if st else 0})
                except Exception:
                    out.append({"name": name, "mb": 0.0, "mtime": 0})
            return {"dbs": out}
        except Exception as e:
            return {"dbs": [], "error": str(e)[:200]}

    return app
