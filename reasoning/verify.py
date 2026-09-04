import re
import json
from typing import List, Dict, Optional
from core import db
from reasoning.graph import query_kg_triples
from core.llm import call_model_json
import config
import logging
logger = logging.getLogger(__name__)


def _safe_str(value):
    """Convert any value to string, or return empty string for None/dict/list."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        logger.warning("Unexpected exception occurred", exc_info=True)
        return ""


def triple_to_key(subject, predicate, obj):
    return (_safe_str(subject).lower().strip(), _safe_str(predicate).lower().strip(), _safe_str(obj).lower().strip())


def claims_to_triples(claims):
    triples = set()
    for c in claims:
        triples.add(triple_to_key(_safe_str(c.get("subject")), _safe_str(c.get("predicate")), _safe_str(c.get("object"))))
    return triples


def derive_implied_triples(triples: set) -> set:
    """Compute transitive closure for selected predicates."""
    new_triples = set(triples)
    changed = True
    while changed:
        changed = False
        for subj, pred, obj in list(new_triples):
            if pred in ("is_a", "part_of", "located_in", "belongs_to"):
                for subj2, pred2, obj2 in list(new_triples):
                    if subj2 == obj and pred2 == pred:
                        new_triple = (subj, pred, obj2)
                        if new_triple not in new_triples:
                            new_triples.add(new_triple)
                            changed = True
    return new_triples


def verify_symstep(claim: Dict, prior_claims: List[Dict]) -> bool:
    """
    Check claim consistency with prior claims, including implied facts.
    """
    subject = _safe_str(claim.get("subject")).strip()
    predicate = _safe_str(claim.get("predicate")).strip()
    obj = _safe_str(claim.get("object")).strip()
    if not subject or not predicate:
        return False

    prior_triples = claims_to_triples(prior_claims)
    implied_triples = derive_implied_triples(prior_triples)
    claim_triple = triple_to_key(subject, predicate, obj)

    for t in implied_triples:
        if t[0] == claim_triple[0] and t[1] == claim_triple[1]:
            if t[2] != claim_triple[2]:
                return False
    for t in prior_triples:
        if t[0] == claim_triple[0] and t[1] == claim_triple[1]:
            if t[2] != claim_triple[2]:
                return False
    return True


def extract_triple_from_text(step_text: str) -> Optional[Dict]:
    if not step_text:
        return None

    patterns = [
        (r"(.+?)\s+is\s+(?:a|an)\s+(.+)", "is_a"),
        (r"(.+?)\s+are\s+(?:a|an|the)?\s*(.+)", "is_a"),
        (r"(.+?)\s+has\s+(.+)", "has"),
        (r"(.+?)\s+have\s+(.+)", "has"),
        (r"(.+?)\s+located\s+in\s+(.+)", "located_in"),
        (r"(.+?)\s+is\s+located\s+in\s+(.+)", "located_in"),
        (r"(.+?)\s+works\s+for\s+(.+)", "works_for"),
        (r"(.+?)\s+is\s+part\s+of\s+(.+)", "part_of"),
        (r"(.+?)\s+belongs\s+to\s+(.+)", "belongs_to"),
        (r"(.+?)\s+caused\s+(.+)", "caused_by"),
        (r"(.+?)\s+produced\s+(.+)", "produced"),
        (r"(.+?)\s+discovered\s+(.+)", "discovered"),
    ]

    for pattern, pred in patterns:
        m = re.search(pattern, step_text, re.IGNORECASE)
        if m:
            subject = _safe_str(m.group(1)).strip()
            obj = _safe_str(m.group(2)).strip()
            subject = re.sub(r'^(a|an|the)\s+', '', subject, flags=re.IGNORECASE)
            obj = re.sub(r'^(a|an|the)\s+', '', obj, flags=re.IGNORECASE)
            return {"subject": subject, "predicate": pred, "object": obj}

    prompt = f"""Extract the subject, predicate, and object from the following sentence as a single JSON object with keys "subject", "predicate", "object".
Sentence: "{step_text}"
Return only JSON.
"""
    data = call_model_json(prompt, max_tokens=128)
    if data and all(k in data for k in ("subject", "predicate", "object")):
        return {
            "subject": _safe_str(data.get("subject")),
            "predicate": _safe_str(data.get("predicate")),
            "object": _safe_str(data.get("object")),
        }
    return None


def verify_vericot(step_text: str, context: str, kg) -> bool:
    triple = extract_triple_from_text(step_text)
    if not triple:
        return False

    subject = _safe_str(triple.get("subject"))
    predicate = _safe_str(triple.get("predicate"))
    obj = _safe_str(triple.get("object"))

    if query_kg_triples(subject=subject, predicate=predicate, object_=obj):
        return True

    conn = db.db_connect("reasoning")
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM implied_triples WHERE subject=? AND predicate=? AND object=?",
                (subject, predicate, obj))
    found = cur.fetchone() is not None
    conn.close()
    return found


def verify_fidelis(claim: Dict, kg) -> bool:
    subject = _safe_str(claim.get("subject")).strip()
    predicate = _safe_str(claim.get("predicate")).strip()
    obj = _safe_str(claim.get("object")).strip()
    if not subject or not predicate:
        return False

    if query_kg_triples(subject=subject, predicate=predicate, object_=obj):
        return True

    conn = db.db_connect("external_graph")
    cur = conn.cursor()
    cur.execute("SELECT global_node_id FROM global_nodes WHERE canonical_name=? OR EXISTS (SELECT 1 FROM json_each(global_nodes.aliases_json) WHERE value = ?)",
                (subject, subject))
    subj_node = cur.fetchone()
    if subj_node:
        cur.execute("SELECT global_node_id FROM global_nodes WHERE canonical_name=? OR EXISTS (SELECT 1 FROM json_each(global_nodes.aliases_json) WHERE value = ?)",
                    (obj, obj))
        obj_node = cur.fetchone()
        if obj_node:
            cur.execute("SELECT edge_id FROM global_edges WHERE source_node_id=? AND target_node_id=? AND relation_type=?",
                        (subj_node[0], obj_node[0], predicate))
            if cur.fetchone():
                conn.close()
                return True
    conn.close()
    return False


def verify_rcot(conclusion: str, kg) -> bool:
    if not conclusion:
        return False

    visited = set()
    queue = [conclusion.lower().strip()]
    found = False
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        conn = db.db_connect("reasoning")
        cur = conn.cursor()
        cur.execute("SELECT subject, predicate FROM kg_triples WHERE LOWER(object)=?", (current,))
        rows = cur.fetchall()
        conn.close()

        if rows:
            found = True
            break
    return found


def verify_ares(reasoning_chain: List[Dict]) -> float:
    if not reasoning_chain:
        return 0.0

    calibrated = getattr(config, "ENABLE_CALIBRATED_ARES", False)
    accepted = []
    scores = []
    for step in reasoning_chain:
        if not isinstance(step, dict):
            scores.append(0.0)
            continue
        if not accepted:
            score = 1.0 if step.get("grounded", False) else 0.0
        else:
            triples = claims_to_triples(accepted)
            implied = derive_implied_triples(triples)
            claim_triple = triple_to_key(
                _safe_str(step.get("subject")), _safe_str(step.get("predicate")), _safe_str(step.get("object"))
            )
            if claim_triple in implied:
                score = 1.0
            else:
                consistent = verify_symstep(step, accepted)
                if calibrated:
                    try:
                        from core.embeddings import get_embeddings_dict
                        import numpy as np
                        prior_texts = [a.get("text", "") for a in accepted if a.get("text")]
                        all_texts = [step.get("text", "")] + prior_texts
                        emb_map = get_embeddings_dict([t for t in all_texts if t], space='hyperbolic')
                        claim_emb = emb_map.get(step.get("text", ""))
                        prior_embs = [emb_map.get(t) for t in prior_texts]
                        prior_embs = [p for p in prior_embs if p is not None]
                        if claim_emb and prior_embs:
                            from core.hyperbolic import ensure_hyperbolic, hyperbolic_distance_matrix
                            claim_vec = ensure_hyperbolic(claim_emb, space='hyperbolic')[None, :]
                            pmat = np.stack([ensure_hyperbolic(p, space='hyperbolic') for p in prior_embs])
                            if getattr(config, "USE_HYPERBOLIC_RETRIEVAL", True):
                                dists = hyperbolic_distance_matrix(claim_vec, pmat)[0]
                                sims = 1.0 / (1.0 + dists)
                                max_sim = float(np.max(sims)) if len(sims) else 0.0
                            else:
                                import numpy as _np
                                claim_arr = _np.asarray(claim_vec[0], dtype=_np.float32)
                                max_sim = 0.0
                                for pv in pmat:
                                    sim = float(_np.dot(claim_arr, pv) / (_np.linalg.norm(claim_arr) * _np.linalg.norm(pv) + 1e-8))
                                    max_sim = max(max_sim, sim)
                            score = max(0.0, min(1.0, max_sim * 0.8 + (0.5 if consistent else 0.0)))
                        else:
                            score = 0.5 if consistent else 0.0
                    except Exception:
                        score = 0.5 if consistent else 0.0
                else:
                    score = 0.5 if consistent else 0.0
        scores.append(score)
        accepted.append(step)
    return sum(scores) / len(scores)


def verify_claim(claim: Dict, source_text: Optional[str] = None, kg=None) -> List[Dict]:
    results = []

    if source_text and claim.get("source_span"):
        results.append({
            "layer": "text_grounding",
            "verified": claim["source_span"] in source_text,
            "confidence": 1.0 if claim["source_span"] in source_text else 0.0,
        })

    sym = verify_symstep(claim, claim.get("_prior_claims", []))
    results.append({"layer": "symstep", "verified": sym, "confidence": 1.0 if sym else 0.0})

    vericot = verify_vericot(claim.get("text", ""), source_text, kg)
    results.append({"layer": "vericot", "verified": vericot, "confidence": 0.8 if vericot else 0.0})

    fidelis = verify_fidelis(claim, kg)
    results.append({"layer": "fidelis", "verified": fidelis, "confidence": 0.9 if fidelis else 0.0})

    rcot = verify_rcot(claim.get("conclusion", claim.get("object", "")), kg)
    results.append({"layer": "rcot", "verified": rcot, "confidence": 0.7 if rcot else 0.0})

    return results


def verify_claim_adaptive(claim, source_text=None, kg=None, threshold=0.6):
    results = []
    if source_text and claim.get("source_span"):
        results.append({
            "layer": "text_grounding",
            "verified": claim["source_span"] in source_text,
            "confidence": 1.0 if claim["source_span"] in source_text else 0.0,
        })
    sym = verify_symstep(claim, claim.get("_prior_claims", []))
    results.append({"layer": "symstep", "verified": sym, "confidence": 1.0 if sym else 0.0})

    initial_conf = sum(v["confidence"] for v in results if v["verified"]) / max(1, len(results))
    if initial_conf >= threshold:
        return results

    vericot = verify_vericot(claim.get("text", ""), source_text, kg)
    results.append({"layer": "vericot", "verified": vericot, "confidence": 0.8 if vericot else 0.0})
    fidelis = verify_fidelis(claim, kg)
    results.append({"layer": "fidelis", "verified": fidelis, "confidence": 0.9 if fidelis else 0.0})
    rcot = verify_rcot(claim.get("conclusion", claim.get("object", "")), kg)
    results.append({"layer": "rcot", "verified": rcot, "confidence": 0.7 if rcot else 0.0})
    return results
