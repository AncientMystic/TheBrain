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


def generate_answer_verified(query: str, conversation_history: str = "") -> str:
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

    facts = [dp for dp in datapoints if dp.get('type') == 'fact']
    chunks = []
    for dp in datapoints:
        if dp.get('type') == 'chunk_ref':
            chunks.append((0, dp.get('doc_hash'), dp.get('text', '')))

    # Reflect: verify facts
    vm = VerificationManager()
    verified_facts = []
    for fact in facts:
        vfact = vm.verify_single_online(fact)
        if vfact['verification_status'] in ('verified', 'partially_verified'):
            # Keep only high confidence verified facts
            if vfact.get('confidence_final', 0) >= 0.6:
                verified_facts.append(vfact)

    if not verified_facts:
        # Fall back to chunks only
        context = build_context([], chunks=chunks, conversation_history=conversation_history)
        prompt = f"""The user asked: {query}

We could not find enough verified information. Provide a cautious answer based on the available snippets, clearly stating uncertainty.

Context:
{context}

Answer:"""
        return call_model(prompt, max_tokens=1024)

    # Speak: use verified facts and chunks
    context = build_context(verified_facts, chunks=chunks, conversation_history=conversation_history)
    prompt = f"""The user asked: {query}

Use only the verified facts below to answer accurately. Cite sources as [doc: filename] when possible.
If information is insufficient, say so.

Context:
{context}

Answer:"""
    return call_model(prompt, max_tokens=1024)
