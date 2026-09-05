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
        # Strict allow-list: report names the coordinator writes are
        # [A-Za-z0-9._-]+.md. Anything else (slashes, .., null bytes,
        # absolute paths, other extensions) is rejected before touching disk.
        import re as _re
        import os as _os
        raw = str(name or "")
        if not _re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}\.md", raw):
            return {"text": "", "error": "invalid name"}
        root = _reports_dir().resolve()
        p = (root / raw).resolve()
        try:
            # is_relative_to is immune to the sibling-prefix bypass that
            # plain startswith allows (e.g. /reports-evil/x vs /reports).
            inside = p.is_relative_to(root)
        except Exception:
            try:
                inside = _os.path.commonpath([str(p), str(root)]) == str(root)
            except Exception:
                inside = False
        if not inside or not p.is_file():
            return {"text": "", "error": "invalid path"}
        try:
            text = p.read_text(encoding="utf-8", errors="replace")[:20000]
            return {"text": text}
        except Exception as e:
            return {"text": "", "error": str(e)[:200]}

    return app
