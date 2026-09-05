"""
Deep-research tab backend reusing the coordinator (no duplicated logic).

- start: runs coordinator as a job (same function as CLI/chat deep path)
- reports: lists generated Markdown reports (capped, newest first)
- report: previews one report (capped chars, no path traversal)
All bounded, shared auth, generic.
"""
from pathlib import Path
import config as _cfg


def _reports_dir():
    try:
        d = Path(_cfg.BASE_DIR) / "reports"
    except Exception:
        d = Path("reports")
    return d


def register_research_routes(app, require_auth):
    from fastapi import Depends
    from pydantic import BaseModel
    from typing import Optional

    class StartBody(BaseModel):
        query: str
        session_id: Optional[str] = None

    @app.post("/api/jobs/research", dependencies=[Depends(require_auth)])
    async def start_research(body: StartBody):
        q = (body.query or "").strip()[:500]
        if not q:
            return {"job_id": "", "error": "empty query"}
        from webui import jobs as _jobs
        jid = _jobs.create_job("research", {"query": q, "session_id": body.session_id})
        return {"job_id": jid}

    @app.get("/api/research/reports", dependencies=[Depends(require_auth)])
    async def list_reports(limit: int = 20):
        limit = max(1, min(int(limit or 20), 50))
        d = _reports_dir()
        try:
            files = sorted(d.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit] if d.is_dir() else []
        except Exception:
            files = []
        return {"reports": [{"name": f.name, "kb": round(f.stat().st_size / 1024, 1)} for f in files]}

    @app.get("/api/research/report", dependencies=[Depends(require_auth)])
    async def get_report(name: str):
        # No traversal: basename only, must stay inside reports dir
        safe = Path(str(name or "")).name
        if not safe or not safe.endswith(".md"):
            return {"text": "", "error": "invalid name"}
        p = _reports_dir() / safe
        try:
            if not str(p.resolve()).startswith(str(_reports_dir().resolve())):
                return {"text": "", "error": "invalid path"}
            text = p.read_text(encoding="utf-8", errors="replace")[:20000]
            return {"text": text}
        except Exception as e:
            return {"text": "", "error": str(e)[:200]}

    return app
