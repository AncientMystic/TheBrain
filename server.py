import time, uuid, json
from typing import List, Optional, Any
from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from chat import analyze_query, retrieve_from_graph, fallback_to_chunks, build_context, generate_answer
from memory.retrieve import retrieve_memories
from logic.decision import decide_logic_modules
from core.embeddings import get_embeddings_batch
from core import db
from reasoning.orchestrator import orchestrate_reasoning
from deep_research.coordinator import DeepResearchCoordinator
import logging
logger = logging.getLogger(__name__)

app = FastAPI(title="TheBrain OpenAI-Compatible API")

_cors_origins = getattr(config, "CORS_ORIGINS", [])
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins if _cors_origins else [],
    allow_credentials=bool(_cors_origins),
    allow_methods=["*"] if _cors_origins else ["GET", "POST"],
    allow_headers=["*"] if _cors_origins else ["Content-Type", "Authorization"],
)


async def require_auth(authorization: Optional[str] = Header(None, alias="Authorization")):
    expected = getattr(config, "SERVER_AUTH_TOKEN", "")
    if not expected:
        if getattr(config, "DEBUG_VERBOSE", False):
            logger.debug("Auth check skipped: no SERVER_AUTH_TOKEN configured")
        return True
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    if authorization[len("Bearer "):] != expected:
        raise HTTPException(status_code=401, detail="Invalid auth token")
    return True

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = config.MODEL_NAME
    messages: List[ChatMessage]
    max_tokens: int = 1024
    temperature: float = 0.0
    stream: bool = False
    session_id: Optional[str] = None
    reasoning: bool = False
    deep_research: bool = False

class EmbeddingRequest(BaseModel):
    input_text: Any = None
    input: Any = None
    model: str = config.EMBEDDING_MODEL

def _process_chat(messages, session_id=None, reasoning=False, deep_research=False):
    user_msgs = [m.content for m in messages if m.role == "user"]
    query = user_msgs[-1] if user_msgs else ""
    if not query:
        return "", []

    if deep_research:
        coordinator = DeepResearchCoordinator(session_id)
        report_path = coordinator.run(query)
        return f"Deep research report generated: {report_path}", []

    if reasoning:
        answer, facts = orchestrate_reasoning(query)
        return answer, facts

    # Normal chat pipeline
    analysis = analyze_query(query)

    logic_ids = decide_logic_modules(query, context=query[:1000])
    logic_context = ""
    if logic_ids:
        conn = db.db_connect("logic")
        cur = conn.cursor()
        for lid in logic_ids:
            cur.execute("SELECT name, category, summary, content FROM logic_modules WHERE logic_id=?", (lid,))
            row = cur.fetchone()
            if row:
                logic_context += f"[Logic: {row[0]} ({row[1]})]\n{row[2]}\n{row[3]}\n\n"
        conn.close()

    memories = retrieve_memories(query, top_k=5, session_id=session_id)
    memory_text = "\n".join([f"[Memory] {m[2]}" for m in memories])

    facts = retrieve_from_graph(analysis, top_k=50)
    chunks = fallback_to_chunks(query, top_k=3)

    context = build_context(facts, chunks=chunks)
    if logic_context:
        context = logic_context + "\n\n" + context
    if memory_text:
        context = memory_text + "\n\n" + context

    answer = generate_answer(query, context)
    return answer, facts

@app.get("/v1/models")
async def list_models():
    models = []
    for ep in config.LLM_ENDPOINTS:
        models.append({
            "id": ep["model"],
            "object": "model",
            "backend": ep.get("backend", "lmstudio"),
            "url": ep.get("url", ""),
        })
    return {"data": models, "object": "list"}

@app.post("/v1/chat/completions", dependencies=[Depends(require_auth)])
async def chat_completions(req: ChatCompletionRequest):
    answer, facts = _process_chat(req.messages, req.session_id, req.reasoning, req.deep_research)
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

@app.post("/v1/completions", dependencies=[Depends(require_auth)])
async def completions(req: ChatCompletionRequest):
    answer, facts = _process_chat(req.messages, req.session_id, req.reasoning, req.deep_research)
    return {
        "id": f"cmpl-{uuid.uuid4()}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"text": answer, "index": 0, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

@app.post("/v1/responses", dependencies=[Depends(require_auth)])
async def responses(req: ChatCompletionRequest):
    answer, facts = _process_chat(req.messages, req.session_id, req.reasoning, req.deep_research)
    return {
        "id": f"resp-{uuid.uuid4()}",
        "object": "response",
        "created": int(time.time()),
        "model": req.model,
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": answer}]}],
        "usage": {"input_tokens": 0, "output_tokens": 0}
    }

@app.post("/v1/embeddings", dependencies=[Depends(require_auth)])
async def embeddings(req: EmbeddingRequest):
    texts = req.input_text if req.input_text is not None else req.input
    if texts is None:
        return {"data": [], "model": req.model, "object": "list"}
    if isinstance(texts, str):
        texts = [texts]
    embs = get_embeddings_batch(texts, model=req.model)
    data = [{"object": "embedding", "index": i, "embedding": emb} for i, emb in enumerate(embs) if emb is not None]
    return {"data": data, "model": req.model, "object": "list"}

@app.post("/v1/reasoning", dependencies=[Depends(require_auth)])
async def reasoning_endpoint(req: ChatCompletionRequest):
    answer, facts = orchestrate_reasoning(req.messages[-1].content)
    return {"answer": answer, "facts": facts}

@app.get("/v1/health")
async def health():
    statuses = []
    for ep in config.LLM_ENDPOINTS:
        try:
            from core.backends import create_backend
            provider = create_backend(ep)
            ok = provider.health_check()
            statuses.append({
                "url": ep.get("url", ""),
                "model": ep.get("model", ""),
                "backend": ep.get("backend", "lmstudio"),
                "ok": ok,
            })
        except Exception as e:
            logger.error(f"Backend health check failed for {ep.get('url', '')}: {e}", exc_info=True)
            statuses.append({"url": ep.get("url", ""), "ok": False, "error": "backend unavailable"})
    return {"status": "ok", "endpoints": len(config.LLM_ENDPOINTS), "endpoint_status": statuses}

@app.get("/metrics", dependencies=[Depends(require_auth)])
async def metrics():
    from core.metrics import get_all_metrics
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(get_all_metrics())
