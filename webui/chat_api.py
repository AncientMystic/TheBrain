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
                from memory.bus import store as _store
                _store(body.session_id or f"webui_{int(time.time())}", q[len("remember:"):].strip(), memory_type="user_note")
                return {"answer": "Memory stored.", "facts": [], "ms": 0}
            except Exception as e:
                return {"answer": f"Memory store failed: {e}", "facts": [], "ms": 0}
        t0 = time.time()
        try:
            # Reuse server pipeline (normal + reasoning + deep-research + logic + memories)
            from server import ChatMessage as _CM, _process_chat as _pc
            msgs = [_CM(role="user", content=q)]
            try:
                from core.llm import _select_endpoint_by_type as _sel
                _ep = _sel("chat")
                _served = {"model": _ep.get("model", ""), "url": _ep.get("url", "")} if _ep else {}
            except Exception:
                _served = {}
            answer, facts = _pc(msgs, body.session_id, body.reasoning, body.deep_research)
            clean = []
            # 25 matches the context fact cap so every cited [S#] tag resolves to a footnote.
            for f in (facts or [])[:25]:
                if isinstance(f, dict):
                    clean.append({"fact_text": str(f.get("fact_text", ""))[:300],
                                  "confidence": f.get("confidence", 0),
                                  "doc": f.get("doc_name", f.get("doc_hash", "")),
                                  "truth_class": f.get("truth_class", ""),
                                  "verification_status": f.get("verification_status", "")})
            return {"answer": answer, "facts": clean, "ms": int((time.time() - t0) * 1000), "served": _served}
        except Exception as e:
            return {"answer": f"Chat failed: {e}", "facts": [], "ms": int((time.time() - t0) * 1000)}

    def _clean_feet(facts):
        clean = []
        for f in (facts or [])[:25]:
            if isinstance(f, dict):
                clean.append({"fact_text": str(f.get("fact_text", ""))[:300],
                              "confidence": f.get("confidence", 0),
                              "doc": f.get("doc_name", f.get("doc_hash", "")),
                              "truth_class": f.get("truth_class", ""),
                              "verification_status": f.get("verification_status", "")})
        return clean

    @app.post("/api/chat/stream", dependencies=[Depends(require_auth)])
    async def chat_stream(body: ChatBody):
        """Token-streaming twin of /api/chat (SSE over POST so auth headers work).

        Normal path: same retrieval stages as server._process_chat, then live
        synthesis tokens, then a done event with the authoritative answer +
        facts. Reasoning/deep paths run blocking (multi-call pipelines) and
        emit a single done event — documented, not silent.
        """
        from fastapi.responses import StreamingResponse
        import json as _json

        q = (body.query or "").strip()
        if q.lower().startswith("remember:"):
            def _memgen():
                try:
                    from memory.bus import store as _store
                    _store(body.session_id or f"webui_{int(time.time())}",
                           q[len("remember:"):].strip(), memory_type="user_note")
                    yield "data: " + _json.dumps({"t": "done", "answer": "Memory stored.",
                                                                  "facts": [], "ms": 0}) + "\n\n"
                except Exception as e:
                    yield "data: " + _json.dumps({"t": "done", "answer": f"Memory store failed: {e}",
                                                                  "facts": [], "ms": 0}) + "\n\n"
            return StreamingResponse(_memgen(), media_type="text/event-stream")

        t0 = time.time()

        def _gen():
            try:
                if body.reasoning or body.deep_research:
                    from server import _process_chat as _pc
                    from server import ChatMessage as _CM
                    answer, facts = _pc([_CM(role="user", content=q)],
                                        body.session_id, body.reasoning, body.deep_research)
                    yield "data: " + _json.dumps({"t": "done", "answer": answer,
                                                                  "facts": _clean_feet(facts),
                                                                  "ms": int((time.time() - t0) * 1000)}) + "\n\n"
                    return
                # Same stages as server._process_chat (normal path), verbatim.
                from chat import analyze_query, retrieve_from_graph, fallback_to_chunks
                from chat.context_builder import build_tagged_context
                from chat.synthesize import synthesize_answer_stream
                from logic.decision import decide_logic_modules
                from memory.bus import retrieve as retrieve_memories
                from core import db
                analysis = analyze_query(q)
                logic_context = ""
                try:
                    _ids = decide_logic_modules(q, context=q[:1000])
                    if _ids:
                        _conn = db.db_connect("logic")
                        _cur = _conn.cursor()
                        for _lid in _ids:
                            _cur.execute("SELECT name, category, summary, content FROM logic_modules WHERE logic_id=?", (_lid,))
                            _row = _cur.fetchone()
                            if _row:
                                logic_context += f"[Logic: {_row[0]} ({_row[1]})]\n{_row[2]}\n{_row[3]}\n\n"
                        _conn.close()
                except Exception:
                    pass
                try:
                    memories = retrieve_memories(q, top_k=5, session_id=body.session_id)
                    memory_text = "\n".join([f"[Memory] {m[2]}" for m in memories])
                except Exception:
                    memory_text = ""
                facts = retrieve_from_graph(analysis, top_k=50)
                chunks = fallback_to_chunks(q, top_k=3)
                try:
                    from core.model_context import answer_budget
                    _budget, _blabel = answer_budget()
                except Exception:
                    _budget, _blabel = None, ""
                context, facts, _tagmap = build_tagged_context(
                    facts, chunks=chunks, budget_chars=_budget, model_label=_blabel)
                if logic_context:
                    context = logic_context + "\n\n" + context
                if memory_text:
                    context = memory_text + "\n\n" + context
                try:
                    import config as _cfgt2
                    if getattr(_cfgt2, "TRIVIUM_CONTEXT", False):
                        from core.trivium import trivium_context_block as _tcb2
                        _tri2 = _tcb2(q)
                        if _tri2:
                            context = _tri2 + "\n\n" + context
                except Exception:
                    pass
                _full = []
                for ev in synthesize_answer_stream(q, context):
                    if ev.get("t") == "tok":
                        _full.append(ev.get("x", ""))
                        yield "data: " + _json.dumps({"t": "tok", "x": ev.get("x", "")}) + "\n\n"
                    elif ev.get("t") == "done":
                        _full = [ev.get("answer", "")]
                yield "data: " + _json.dumps({"t": "done", "answer": "".join(_full),
                                                              "facts": _clean_feet(facts),
                                                              "ms": int((time.time() - t0) * 1000)}) + "\n\n"
            except Exception as e:
                yield "data: " + _json.dumps({"t": "done", "answer": f"Chat failed: {e}",
                                                              "facts": [],
                                                              "ms": int((time.time() - t0) * 1000)}) + "\n\n"

        return StreamingResponse(_gen(), media_type="text/event-stream")

    return app
