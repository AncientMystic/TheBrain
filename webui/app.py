"""
WebUI mount for TheBrain (skeleton batch 1: shell + config + guided mock).

Mounts under main server app: /ui (static) + /api/* (config schema, jobs).
All /api/* except health summary reuse require_auth Bearer (same as /v1/*).
"""
from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import config


def mount_webui(app):
    from webui.schema import get_schema
    from webui import jobs as _jobs
    try:
        from server import require_auth as _auth
    except Exception:
        async def _auth():
            return True
    try:
        from webui.graph_api import register_graph_routes
        register_graph_routes(app, _auth)
    except Exception as e:
        print(f"  (Graph routes skipped: {e})")
    try:
        from webui.chat_api import register_chat_routes
        register_chat_routes(app, _auth)
    except Exception as e:
        print(f"  (Chat routes skipped: {e})")
    try:
        from webui.config_api import register_config_routes
        register_config_routes(app, _auth)
    except Exception as e:
        print(f"  (Config routes skipped: {e})")
    try:
        from webui.recoll_api import register_recoll_routes
        register_recoll_routes(app, _auth)
    except Exception as e:
        print(f"  (Recoll routes skipped: {e})")
    try:
        from webui.audit_api import register_audit_routes
        register_audit_routes(app, _auth)
    except Exception as e:
        print(f"  (Audit routes skipped: {e})")
    try:
        from webui.server_api import register_server_routes
        register_server_routes(app, _auth)
    except Exception as e:
        print(f"  (Server routes skipped: {e})")

    @app.get("/api/health/summary")
    async def health_summary():
        try:
            import config as _cfg
            eps = len(getattr(_cfg, "LLM_ENDPOINTS", []))
            edim = int(getattr(_cfg, "EMBEDDING_DIM", 1024))
            emodel = str(getattr(_cfg, "EMBEDDING_MODEL", ""))
        except Exception:
            eps, edim, emodel = 0, 1024, ""
        return {"llm_endpoints": eps, "embedding_dim": edim, "embedding_model": emodel}

    @app.get("/api/config/schema", dependencies=[Depends(_auth)])
    async def config_schema():
        return JSONResponse(get_schema())

    @app.post("/api/jobs/guided", dependencies=[Depends(_auth)])
    async def start_guided(payload: dict = None):
        jid = _jobs.create_job("guided-learning", payload or {})
        return {"job_id": jid}

    @app.get("/api/jobs/{jid}/stream")
    async def job_stream(jid: str):
        # SSE with mocked buffered events (skeleton: drain current queue, then done hint)
        from fastapi.responses import StreamingResponse
        import asyncio
        import json as _json

        async def gen():
            job = _jobs.get_job(jid)
            if not job:
                yield f"data: {_json.dumps({'type': 'error', 'msg': 'unknown job'})}\n\n"
                return
            # Drain with timeout until done (skeleton: mock finishes in ~2s)
            while True:
                try:
                    import queue as _q
                    ev = job["queue"].get_nowait()
                    yield f"data: {_json.dumps(ev, default=str)}\n\n"
                    if ev.get("type") == "done":
                        break
                except Exception:
                    if job.get("done"):
                        break
                    await asyncio.sleep(0.15)
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/api/jobs/{jid}/cancel", dependencies=[Depends(_auth)])
    async def job_cancel(jid: str):
        ok = _jobs.cancel_job(jid)
        if not ok:
            raise HTTPException(404, "unknown job")
        return {"cancelled": True}

    # Static shell (glass UI)
    static_dir = str(Path(__file__).parent / "static")
    try:
        app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="webui")
    except Exception:
        pass
    return app
