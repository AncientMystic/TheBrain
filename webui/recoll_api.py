"""
Recoll tab backend reusing existing search pipelines (no duplicated logic).

- search: direct RecollClient query (read-only, capped, allow-listed binary via existing guards)
- fast: keyword fast-mode status (full pipeline runs as job in later batch; this batch returns preview count)
- index: reports builder availability (full build runs via CLI/main pipeline; UI triggers job in later batch)
All bounded inputs, shared auth, generic (no doc-specific rules).
"""
import time


def register_recoll_routes(app, require_auth):
    from fastapi import Depends
    from pydantic import BaseModel
    from typing import Optional

    class SearchBody(BaseModel):
        query: str
        limit: int = 20

    @app.post("/api/recoll/search", dependencies=[Depends(require_auth)])
    async def recoll_search(body: SearchBody):
        q = (body.query or "").strip()[:500]
        if not q:
            return {"results": [], "ms": 0}
        try:
            lim = max(1, min(int(body.limit or 20), 200))
        except Exception:
            lim = 20
        t0 = time.time()
        try:
            from core.recoll_client import RecollClient
            client = RecollClient()
            results, _ = client.search(q, limit=lim)
            out = []
            for r in (results or [])[:lim]:
                out.append({"path": str(r.get("path", ""))[:300], "title": str(r.get("title", ""))[:200],
                            "snippet": str(r.get("snippet", ""))[:500]})
            return {"results": out, "ms": int((time.time() - t0) * 1000)}
        except Exception as e:
            return {"results": [], "ms": int((time.time() - t0) * 1000), "error": str(e)[:300]}

    class FastBody(BaseModel):
        keyword: Optional[str] = ""
        limit: int = 20

    @app.post("/api/recoll/fast", dependencies=[Depends(require_auth)])
    async def recoll_fast(body: FastBody):
        # Skeleton: preview via same search path (full fast-mode pipeline job lands next batch)
        return await recoll_search(SearchBody(query=body.keyword or "", limit=body.limit))

    @app.get("/api/recoll/status", dependencies=[Depends(require_auth)])
    async def recoll_status():
        import shutil
        import config as _cfg
        binary = getattr(_cfg, "RECOLL_BIN", "recollq")
        return {"binary": binary, "available": bool(shutil.which(binary)),
                "db": getattr(_cfg, "RECOLL_DB", ""), "max_results": getattr(_cfg, "RECOLL_MAX_RESULTS", 50)}

    return app
