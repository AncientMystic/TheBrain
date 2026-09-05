
"""
GIVE-pattern verified chat.
Observe -> Reflect -> Speak
"""
import config
from typing import List, Dict, Any
from core.llm import call_model
from retrieval.orchestrator import RetrievalOrchestrator
from reasoning.verification_manager import VerificationManager
from chat.query_analyzer import analyze_query
from chat.context_builder import build_context
from core.model_router import get_chat_endpoint
from core.metrics import inc_counter, Timer
from chat.responder import clean_answer


def generate_answer_verified(query: str, conversation_history: str = "", active_entities=None) -> str:
    inc_counter("chat_verified_requests_total")
    """
    Generate answer using GIVE pattern:
    Observe: retrieve candidate datapoints.
    Reflect: verify facts using VerificationManager.
    Speak: synthesize answer with only verified facts.
    """
    # Observe
    analysis = analyze_query(query)
    orchestrator = RetrievalOrchestrator()
    datapoints = orchestrator.retrieve(query, analysis, top_k=50)

    facts = [dp for dp in datapoints if dp.get("type") == "fact"]
    chunks = []
    for dp in datapoints:
        if dp.get("type") == "chunk_ref":
            chunks.append((0, dp.get("chunk_id"), dp.get("doc_hash"), dp.get("text", "")))

    # Reflect: verify facts in ONE batch (single embedding/triple fan-out).
    # Pre-cap by pre-verification confidence so batch cost stays bounded;
    # the cap is prompt-size hygiene on candidates, never on the corpus.
    import time as _t
    _t0 = _t.time()
    try:
        _pre = sorted(facts, key=lambda f: float(f.get("confidence", 0) or 0), reverse=True)[:25]
    except Exception:
        _pre = facts[:25]
    vm = VerificationManager()
    try:
        _batch = vm.verify_batch(_pre)
    except Exception:
        _batch = []
        for fact in _pre:
            try:
                _batch.append(vm.verify_single_online(fact))
            except Exception:
                continue
    verified_facts = [v for v in _batch
                      if v.get("verification_status") in ("verified", "partially_verified")
                      and float(v.get("confidence_final", 0) or 0) >= 0.6]
    if getattr(config, "DEBUG_VERBOSE", False):
        print(f"    (Verify: {len(facts)} candidates -> {len(_pre)} batched -> "
              f"{len(verified_facts)} verified in {_t.time() - _t0:.1f}s)")

    if not verified_facts:
        context = build_context([], chunks=chunks, conversation_history=conversation_history)
        prompt = f"""The user asked: {query}

We could not find enough verified information. Answer in at most 3 sentences from the snippets below, clearly stating uncertainty. One citation total, short form `(Document: filename)`.

Context:
{context}

Answer:"""
        raw_answer = call_model(prompt, max_tokens=min(getattr(config, "CHAT_ANSWER_MAX_TOKENS", 4096), 4096))
        return clean_answer(raw_answer)

    # Filter chunks: keep only those whose doc_hash appears in verified_facts
    verified_doc_hashes = {f.get("doc_hash") for f in verified_facts if f.get("doc_hash")}
    if verified_doc_hashes:
        chunks = [c for c in chunks if c[2] in verified_doc_hashes]
    # Speak: use verified facts and chunks, with graph context organization if enabled
    if getattr(config, "USE_CONTEXT_ORGANIZER", True):
        try:
            from chat.context_organizer import organize_facts
            organized = organize_facts(verified_facts, active_entities=active_entities)
            if organized:
                # Keep chunks as a separate section if available
                if chunks:
                    chunk_lines = ["[Raw excerpts]"]
                    for _, _, doc_hash, text in chunks[:10]:
                        chunk_lines.append(f"- {text[:300]}")
                    organized = organized + "\n\n" + "\n".join(chunk_lines)
                context = organized
            else:
                context = build_context(verified_facts, chunks=chunks, conversation_history=conversation_history)
        except Exception as e:
            if config.DEBUG_VERBOSE:
                print(f"    (Context organizer error: {e})")
            context = build_context(verified_facts, chunks=chunks, conversation_history=conversation_history)
    else:
        context = build_context(verified_facts, chunks=chunks, conversation_history=conversation_history)

    prompt = f"""The user asked: {query}

Use only the verified facts and graph paths below (pre-deduped, confidence-scored, highest first). If a graph path is present, explain the connection using that path. Do not combine facts that are not explicitly linked.

Structure: 1-2 sentence direct answer first, then short sections or bullets for distinct parts, then a 1-sentence verdict. Aim for ~400 words.
Citations: at most ONCE per paragraph or bullet, short form `(Document: filename)`. Never cite the same document twice in a row. Every figure, date, and proper name MUST carry a citation. Never hedge ("debated", "moderate") without a cited fact behind it.
If information is insufficient, say so in one sentence and stop.

Context:
{context}

Answer:"""
    raw_answer = call_model(prompt, max_tokens=getattr(config, "CHAT_ANSWER_MAX_TOKENS", 4096))
    return clean_answer(raw_answer)
