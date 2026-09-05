"""
Reasoning beam search and verification.
"""
import hashlib
import config
from reasoning.decompose import decompose_query
from reasoning.verify import verify_claim, verify_claim_adaptive
from reasoning.agents import KGQueryAgent
from core.query_analyzer import analyze_query
from chat.retriever import retrieve_from_graph, fallback_to_chunks
from chat.context_builder import build_context
from core.llm import call_model_json
from core import db


def generate_candidate_claims(sub_question, context, kg, beam_size=3):
    if not context.strip():
        return []
    prompt = f"""Given the sub-question and context below, generate up to {beam_size} distinct atomic claims that could answer the sub-question.
Each claim must be a JSON object with keys:
- subject: string
- predicate: string (use "is_a", "has", "located_in", "works_for", "part_of", "caused_by", "produced", "discovered", "related_to")
- object: string
- source_span: string (2-4 word pointer from context)
- confidence: float (0-1)
- text: string (concise sentence stating the claim)

Sub-question: {sub_question}

Context:
{context}

Return JSON with key "claims" as a list of claim objects.
"""
    data = call_model_json(prompt, max_tokens=1024)
    if not data or "claims" not in data:
        return []
    claims = []
    for c in data["claims"]:
        if isinstance(c, dict):
            c.setdefault("subject", "")
            c.setdefault("predicate", "related_to")
            c.setdefault("object", "")
            c.setdefault("source_span", "")
            c.setdefault("confidence", 0.5)
            c.setdefault("text", "")
            c.setdefault("conclusion", c.get("object", ""))
            claims.append(c)
    return claims


def _norm(text):
    import re
    try:
        return re.sub(r"\s+", " ", str(text or "").lower()).strip()
    except Exception:
        return ""


def canonical_claim_key(claim):
    """Normalized (subject, predicate, object) triple key for a claim dict."""
    if not isinstance(claim, dict):
        return ("", "", "")
    return (_norm(claim.get("subject")), _norm(claim.get("predicate")), _norm(claim.get("object")))


def canonical_path_id(beam):
    """Order-insensitive ID: renumbered/reordered duplicates share one ID.

    Exact canonicalization over normalized triple keys (sorted multiset hash),
    so it never merges distinct shapes — only true duplicates collapse.
    """
    try:
        keys = sorted(canonical_claim_key(c) for c in (beam or []))
        return hashlib.sha256(repr(keys).encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""


def dedupe_beams(beams):
    """Keep best-scoring beam per canonical ID, stable order. Never raises."""
    try:
        scored = []
        for beam in beams or []:
            try:
                score = sum(float(c.get("final_confidence", 0) or 0) for c in beam)
            except Exception:
                score = 0.0
            scored.append((score, beam))
        scored.sort(key=lambda x: x[0], reverse=True)
        seen, out = set(), []
        for _, beam in scored:
            pid = canonical_path_id(beam)
            if pid in seen:
                continue
            seen.add(pid)
            out.append(beam)
        return out
    except Exception:
        return list(beams or [])


def dedupe_claims(claims):
    """Drop exact-duplicate claims (normalized triple), keep first occurrence.

    Empty claims (no triple at all) are always kept — there is nothing to
    compare, and dropping them would silently lose content.
    """
    seen, out = set(), []
    for c in claims or []:
        try:
            key = canonical_claim_key(c)
            if any(key):
                if key in seen:
                    continue
                seen.add(key)
            out.append(c)
        except Exception:
            out.append(c)
    return out


def facts_to_claims(facts, top_k=5):
    claims = []
    for f in facts[:top_k]:
        claims.append({
            "subject": f.get("canonical_value") or f.get("fact_type") or "fact",
            "predicate": "has_fact",
            "object": f.get("fact_text", ""),
            "source_span": f.get("source_span", ""),
            "confidence": f.get("confidence", 0.8),
            "text": f.get("fact_text", ""),
            "conclusion": f.get("fact_text", ""),
            "final_confidence": f.get("confidence", 0.8),
            "fact_id": f.get("fact_id"),
            "verified": True
        })
    return claims


def beam_search(sub_questions, kg, beam_size=3, max_steps=10, diagnostics=None):
    """Beam search with canonical dedup and honest outage accounting.

    diagnostics (optional dict, filled in place): counts of candidate-empty
    fallbacks, accepted-empty fallbacks, and empty-beam outcomes, plus dupes
    collapsed. An outage is always recorded loudly here, never a quiet fewer.
    """
    def _note(key):
        try:
            if diagnostics is not None:
                diagnostics[key] = int(diagnostics.get(key, 0)) + 1
        except Exception:
            pass

    beams = [[]]
    all_verified_claims = []
    for sq in sub_questions:
        analysis = analyze_query(sq["question"])
        facts = retrieve_from_graph(analysis, top_k=10)
        chunks = fallback_to_chunks(sq["question"], top_k=3)
        context = build_context(facts, chunks=chunks)

        new_beams = []
        for beam in beams:
            candidates = generate_candidate_claims(sq["question"], context, kg, beam_size)

            if not candidates:
                _note("candidate_empty_fallbacks")
                candidates = facts_to_claims(facts, top_k=beam_size)

            scored = []
            for c in candidates:
                if config.ADAPTIVE_VERIFICATION:
                    results = verify_claim_adaptive(c, source_text=context, kg=kg)
                else:
                    results = verify_claim(c, source_text=context, kg=kg)
                confidence = sum(v["confidence"] for v in results if v["verified"])
                if c.get("verified") and c.get("final_confidence", 0) > confidence:
                    confidence = c["final_confidence"]
                c["final_confidence"] = confidence
                scored.append((confidence, c))

            scored.sort(key=lambda x: x[0], reverse=True)
            accepted = []
            for conf, claim in scored:
                if conf >= 0.5 and claim.get("subject"):
                    accepted.append(claim)
                    all_verified_claims.append(claim)
                if len(accepted) >= beam_size:
                    break

            if not accepted:
                _note("accepted_empty_fallbacks")
                accepted = facts_to_claims(facts, top_k=beam_size)
                for claim in accepted:
                    all_verified_claims.append(claim)

            for claim in accepted:
                new_beams.append(beam + [claim])

        if new_beams:
            new_beams.sort(key=lambda b: sum(c.get("final_confidence",0) for c in b), reverse=True)
            before = len(new_beams)
            new_beams = dedupe_beams(new_beams)[:beam_size]
            if diagnostics is not None:
                try:
                    diagnostics["beams_deduped"] = int(diagnostics.get("beams_deduped", 0)) + (before - len(new_beams))
                except Exception:
                    pass
            beams = new_beams

    if beams:
        beams.sort(key=lambda b: sum(c.get("final_confidence",0) for c in b), reverse=True)
        return beams[0], dedupe_claims(all_verified_claims)
    _note("empty_beam_outcomes")
    return [], dedupe_claims(all_verified_claims)


def reason_with_verification(query, kg=None, max_steps=10, beam_size=3, diagnostics=None):
    """Run reasoning with verification."""
    if kg is None:
        kg = KGQueryAgent()
    sub_questions = decompose_query(query)
    if not sub_questions:
        if diagnostics is not None:
            try:
                diagnostics["no_sub_questions"] = 1
            except Exception:
                pass
        return [], []
    verified_claims, all_claims = beam_search(sub_questions, kg, beam_size, max_steps, diagnostics)
    return verified_claims, all_claims
