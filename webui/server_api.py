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
            try:
                from core.backends import create_backend
                ok = bool(create_backend(ep).health_check())
            except Exception as e:
                ok = False
                err = str(e)[:150]
            else:
                err = ""
            out.append({"url": ep.get("url", ""), "model": ep.get("model", ""),
                        "backend": ep.get("backend", ""), "ok": ok,
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
