"""
Chat API reusing the exact command-line pipeline (no duplicated logic).

Normal, reasoning-first and deep-research paths dispatch to the same functions
the server and CLI use, so answers match everywhere. Session memory persists
via existing conversation helpers. Bounded inputs, no artificial limits on corpus.
"""
import time


def register_chat_routes(app, require_auth):
    from fastapi import Depends
    from pydantic import BaseModel
    from typing import Optional

    class ChatBody(BaseModel):
        query: str
        session_id: Optional[str] = None
        reasoning: bool = False
        deep_research: bool = False

    @app.post("/api/chat", dependencies=[Depends(require_auth)])
    async def chat(body: ChatBody):
        q = (body.query or "").strip()
        if not q:
            return {"answer": "", "facts": [], "ms": 0}
        # remember: prefix stores memory via existing path (same as CLI)
        if q.lower().startswith("remember:"):
            try:
                from memory import store_memory as _store
                _store(body.session_id or f"webui_{int(time.time())}", q[len("remember:"):].strip(), memory_type="user_note")
                return {"answer": "Memory stored.", "facts": [], "ms": 0}
            except Exception as e:
                return {"answer": f"Memory store failed: {e}", "facts": [], "ms": 0}
        t0 = time.time()
        try:
            # Reuse server pipeline (normal + reasoning + deep-research + logic + memories)
            from server import ChatMessage as _CM, _process_chat as _pc
            msgs = [_CM(role="user", content=q)]
            answer, facts = _process_chat(msgs, body.session_id, body.reasoning, body.deep_research)
            clean = []
            for f in (facts or [])[:20]:
                if isinstance(f, dict):
                    clean.append({"fact_text": str(f.get("fact_text", ""))[:300],
                                  "confidence": f.get("confidence", 0),
                                  "doc": f.get("doc_name", f.get("doc_hash", ""))})
            return {"answer": answer, "facts": clean, "ms": int((time.time() - t0) * 1000)}
        except Exception as e:
            return {"answer": f"Chat failed: {e}", "facts": [], "ms": int((time.time() - t0) * 1000)}

    return app
