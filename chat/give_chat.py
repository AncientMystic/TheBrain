
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

    try:
        from core.model_context import answer_budget
        _budget, _blabel = answer_budget()
    except Exception:
        _budget, _blabel = None, ""
    # Organized path is free-form text: fit the fact list to budget first so the
    # groups + excerpts stay inside the serving model's window (same omitted note).
    if verified_facts and _budget is not None:
        try:
            room, fitted = int(_budget), []
            for fact in verified_facts:
                est = len(str(fact.get("fact_text", ""))) + len(str(fact.get("source_span", ""))) + 120
                if fitted and room - est < 0:
                    continue
                fitted.append(fact)
                room -= est
            if not fitted:
                fitted = verified_facts[:1]
            _omitted = len(verified_facts) - len(fitted)
            verified_facts = fitted
        except Exception:
            _omitted = 0
    else:
        _omitted = 0

    if not verified_facts:
        context = build_context([], chunks=chunks, conversation_history=conversation_history,
                                budget_chars=_budget, model_label=_blabel)
        from chat.synthesize import synthesize_cautious
        return synthesize_cautious(query, context)

    # Cross-chunk dedupe (order-preserving): identical claims extracted from
    # adjacent chunks must not appear twice downstream.
    try:
        from chat.context_builder import _dedup_facts
        verified_facts = _dedup_facts(verified_facts, limit=max(len(verified_facts), 1))
    except Exception:
        pass
    # Number tags in final order so the ledger below aligns with citations.
    for _ti, _tf in enumerate(verified_facts):
        try:
            _tf["citation_tag"] = f"S{_ti + 1}"
        except Exception:
            pass
    # Filter chunks: keep only those whose doc_hash appears in verified_facts
    verified_doc_hashes = {f.get("doc_hash") for f in verified_facts if f.get("doc_hash")}
    if verified_doc_hashes:
        chunks = [c for c in chunks if c[2] in verified_doc_hashes]
    # Speak: use verified facts and chunks, with graph context organization if enabled
    if getattr(config, "USE_CONTEXT_ORGANIZER", True):
        try:
            from chat.context_organizer import organize_facts
            from chat.context_builder import build_tag_ledger
            organized = organize_facts(verified_facts, active_entities=active_entities)
            if organized:
                # Keep chunks as a separate section if available
                if chunks:
                    chunk_lines = ["[Raw excerpts]"]
                    for _, _, doc_hash, text in chunks[:10]:
                        chunk_lines.append(f"- {text[:300]}")
                    organized = organized + "\n\n" + "\n".join(chunk_lines)
                _ledger = build_tag_ledger(verified_facts)
                context = organized + ("\n\n" + _ledger if _ledger else "")
            else:
                context, verified_facts, _ = build_tagged_context(
                    verified_facts, chunks=chunks, conversation_history=conversation_history,
                    budget_chars=None, model_label=_blabel)
        except Exception as e:
            if config.DEBUG_VERBOSE:
                print(f"    (Context organizer error: {e})")
            from chat.context_builder import build_tagged_context as _btc
            context, verified_facts, _ = _btc(
                verified_facts, chunks=chunks, conversation_history=conversation_history,
                budget_chars=_budget, model_label=_blabel)
    else:
        from chat.context_builder import build_tagged_context as _btc2
        context, verified_facts, _ = _btc2(
            verified_facts, chunks=chunks, conversation_history=conversation_history,
            budget_chars=_budget, model_label=_blabel)
    if _omitted:
        context += (f"\n\n[Context fit to {_blabel or 'current model'}: {_omitted} lower-ranked "
                    f"verified facts omitted; ranked highest-first.]")

    from chat.synthesize import synthesize_answer
    return synthesize_answer(query, context)
