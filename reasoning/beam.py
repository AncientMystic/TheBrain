from reasoning.decompose import decompose_query
from reasoning.verify import verify_claim
from reasoning.agents import KGQueryAgent
from chat.query_analyzer import analyze_query
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
            "verified": True  # already validated during ingestion
        })
    return claims


def beam_search(sub_questions, kg, beam_size=3, max_steps=10):
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
                candidates = facts_to_claims(facts, top_k=beam_size)

            scored = []
            for c in candidates:
                results = verify_claim(c, source_text=context, kg=kg)
                confidence = sum(v["confidence"] for v in results if v["verified"])
                # Fact claims are already validated; give them at least their fact confidence
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
                accepted = facts_to_claims(facts, top_k=beam_size)
                for claim in accepted:
                    all_verified_claims.append(claim)

            for claim in accepted:
                new_beams.append(beam + [claim])

        if new_beams:
            new_beams.sort(key=lambda b: sum(c.get("final_confidence",0) for c in b), reverse=True)
            beams = new_beams[:beam_size]

    if beams:
        beams.sort(key=lambda b: sum(c.get("final_confidence",0) for c in b), reverse=True)
        return beams[0], all_verified_claims
    return [], all_verified_claims


def reason_with_verification(query, kg=None, max_steps=10, beam_size=3):
    if kg is None:
        kg = KGQueryAgent()
    sub_questions = decompose_query(query)
    if not sub_questions:
        return [], []
    verified_claims, all_claims = beam_search(sub_questions, kg, beam_size, max_steps)
    return verified_claims, all_claims
