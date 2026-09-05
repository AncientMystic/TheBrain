"""
Audit + review queue APIs reusing CLI functions (no duplicated logic).

- audit run executes as a job (same audit_all as --audit, never auto-deletes admin/verified)
- review table reads contradiction_log review_needed (capped, read-only)
- resolve updates status/note only (no DELETEs anywhere)
All bounded, shared auth, generic.
"""
import time


def register_audit_routes(app, require_auth):
    from fastapi import Depends
    from pydantic import BaseModel
    from typing import Optional

    @app.post("/api/jobs/audit", dependencies=[Depends(require_auth)])
    async def start_audit():
        from webui import jobs as _jobs
        jid = _jobs.create_job("audit", {})
        return {"job_id": jid}

    @app.get("/api/review", dependencies=[Depends(require_auth)])
    async def list_review(limit: int = 100):
        limit = max(1, min(int(limit or 100), 200))
        try:
            from core import db
            conn = db.db_connect("reasoning")
            try:
                cur = conn.cursor()
                cur.execute("""SELECT id, status, triple_a_id, triple_b_id, details
                               FROM contradiction_log WHERE status='review_needed'
                               ORDER BY id DESC LIMIT ?""", (limit,))
                rows = cur.fetchall()
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            return {"items": [], "error": str(e)[:200]}
        return {"items": [{"id": r["id"], "triple_a": r["triple_a_id"], "triple_b": r["triple_b_id"],
                           "details": str(r["details"] or "")[:500]} for r in rows]}

    class ResolveBody(BaseModel):
        decision: str = "resolved"
        note: Optional[str] = ""

    @app.post("/api/review/{item_id}/resolve", dependencies=[Depends(require_auth)])
    async def resolve_item(item_id: int, body: ResolveBody):
        decision = (body.decision or "resolved").strip().lower()
        if decision not in ("resolved", "dismissed"):
            decision = "resolved"
        note = (body.note or "").strip()[:500]
        try:
            from core import db
            conn = db.db_connect("reasoning")
            try:
                cur = conn.cursor()
                cur.execute("SELECT details FROM contradiction_log WHERE id=?", (item_id,))
                row = cur.fetchone()
                if not row:
                    return {"ok": False, "error": "unknown id"}
                prev = row["details"] or ""
                stamped = f"[webui {decision} {time.strftime('%Y-%m-%d')}] {note}\n{prev}"[:2000] if note else prev
                cur.execute("""UPDATE contradiction_log SET status=?, resolved_by='webui',
                               resolved_at=CURRENT_TIMESTAMP, details=? WHERE id=?""",
                            (decision, stamped, item_id))
                conn.commit()
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}
        return {"ok": True}

    return app
