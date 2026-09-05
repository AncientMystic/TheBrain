"""
Logic and memory browsers reusing existing tables (read-only, no writes).

- modules: logic_modules with keyword aggregates, search + cap
- memory: distinct sessions + entries per session with cap
All bounded, shared auth, generic.
"""


def register_logic_routes(app, require_auth):
    from fastapi import Depends

    @app.get("/api/logic/modules", dependencies=[Depends(require_auth)])
    async def logic_modules(search: str = "", limit: int = 50):
        limit = max(1, min(int(limit or 50), 100))
        q = (search or "").strip()[:200]
        try:
            from core import db
            conn = db.db_connect("logic")
            try:
                cur = conn.cursor()
                if q:
                    like = f"%{q}%"
                    cur.execute("""SELECT logic_id, name, category, summary FROM logic_modules
                                   WHERE name LIKE ? OR category LIKE ? OR summary LIKE ?
                                   ORDER BY logic_id DESC LIMIT ?""", (like, like, like, limit))
                else:
                    cur.execute("""SELECT logic_id, name, category, summary FROM logic_modules
                                   ORDER BY logic_id DESC LIMIT ?""", (limit,))
                rows = cur.fetchall()
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            return {"items": [], "error": str(e)[:200]}
        return {"items": [{"id": r["logic_id"], "name": r["name"], "category": r["category"],
                           "summary": str(r["summary"] or "")[:300]} for r in rows]}

    @app.get("/api/memory/sessions", dependencies=[Depends(require_auth)])
    async def memory_sessions(limit: int = 50):
        limit = max(1, min(int(limit or 50), 100))
        try:
            from core import db
            conn = db.db_connect("memories")
            try:
                cur = conn.cursor()
                cur.execute("""SELECT session_id, MAX(timestamp) AS last_active, COUNT(*) AS n
                               FROM memory_entries GROUP BY session_id
                               ORDER BY last_active DESC LIMIT ?""", (limit,))
                rows = cur.fetchall()
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            return {"items": [], "error": str(e)[:200]}
        return {"items": [{"session": r["session_id"], "entries": r["n"],
                           "last_active": r["last_active"]} for r in rows]}

    @app.post("/api/jobs/logic-learn", dependencies=[Depends(require_auth)])
    async def start_logic_learn(payload: dict = None):
        from webui import jobs as _jobs
        jid = _jobs.create_job("logic-learn", payload or {})
        return {"job_id": jid}

    @app.post("/api/jobs/consolidate", dependencies=[Depends(require_auth)])
    async def start_consolidate():
        from webui import jobs as _jobs
        jid = _jobs.create_job("consolidate", {})
        return {"job_id": jid}

    @app.get("/api/memory/entries", dependencies=[Depends(require_auth)])
    async def memory_entries(session: str = "", limit: int = 30):
        limit = max(1, min(int(limit or 30), 100))
        try:
            from core import db
            conn = db.db_connect("memories")
            try:
                cur = conn.cursor()
                if session:
                    cur.execute("""SELECT memory_id, session_id, timestamp, memory_type, content, importance
                                   FROM memory_entries WHERE session_id=? ORDER BY timestamp DESC LIMIT ?""",
                                (session, limit))
                else:
                    cur.execute("""SELECT memory_id, session_id, timestamp, memory_type, content, importance
                                   FROM memory_entries ORDER BY timestamp DESC LIMIT ?""", (limit,))
                rows = cur.fetchall()
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            return {"items": [], "error": str(e)[:200]}
        return {"items": [{"id": r["memory_id"], "session": r["session_id"], "time": r["timestamp"],
                           "type": r["memory_type"], "content": str(r["content"] or "")[:300],
                           "importance": r["importance"]} for r in rows]}

    return app
