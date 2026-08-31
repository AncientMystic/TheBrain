
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

    # Reflect: verify facts
    vm = VerificationManager()
    verified_facts = []
    for fact in facts:
        vfact = vm.verify_single_online(fact)
        if vfact["verification_status"] in ("verified", "partially_verified"):
            if vfact.get("confidence_final", 0) >= 0.6:
                verified_facts.append(vfact)

    if not verified_facts:
        context = build_context([], chunks=chunks, conversation_history=conversation_history)
        prompt = f"""The user asked: {query}

We could not find enough verified information. Provide a cautious answer based on the available snippets, clearly stating uncertainty.

Context:
{context}

Answer:"""
        raw_answer = call_model(prompt, max_tokens=getattr(config, "CHAT_ANSWER_MAX_TOKENS", 32768))
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

Use only the verified facts and graph paths below to answer accurately. If a graph path is present, explain the connection using that path. Do not combine facts that are not explicitly linked. Cite sources as [doc: filename] when possible.
If information is insufficient, say so.

Context:
{context}

Answer:"""
    raw_answer = call_model(prompt, max_tokens=getattr(config, "CHAT_ANSWER_MAX_TOKENS", 32768))
    return clean_answer(raw_answer)
