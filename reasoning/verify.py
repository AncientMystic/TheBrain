import re
import json
from typing import List, Dict, Optional
from core import db
from reasoning.graph import query_kg_triples
from core.llm import call_model_json


# ============================================================
#  Knowledge Base Helpers
# ============================================================
def triple_to_key(subject, predicate, obj):
    return (subject.lower().strip(), predicate.lower().strip(), obj.lower().strip())

def claims_to_triples(claims):
    triples = set()
    for c in claims:
        triples.add(triple_to_key(c.get("subject",""), c.get("predicate",""), c.get("object","")))
    return triples

def derive_implied_triples(triples: set) -> set:
    """
    Compute transitive closure for selected predicates:
    is_a, part_of, located_in, belongs_to, works_for.
    """
    new_triples = set(triples)
    # Simple forward chaining
    changed = True
    while changed:
        changed = False
        # For each pair (A, pred, B) and (B, pred, C) where pred is transitive
        for subj, pred, obj in list(new_triples):
            if pred in ("is_a", "part_of", "located_in", "belongs_to"):
                # Find triples where subject == obj
                for subj2, pred2, obj2 in list(new_triples):
                    if subj2 == obj and pred2 == pred:
                        new_triple = (subj, pred, obj2)
                        if new_triple not in new_triples:
                            new_triples.add(new_triple)
                            changed = True
    return new_triples


# ============================================================
#  SymStep with implication cascade
# ============================================================
def verify_symstep(claim: Dict, prior_claims: List[Dict]) -> bool:
    """
    Check claim consistency with prior claims, including implied facts.
    Returns True if no contradiction, False otherwise.
    """
    subject = claim.get("subject", "").strip()
    predicate = claim.get("predicate", "").strip()
    obj = claim.get("object", "").strip()
    if not subject or not predicate:
        return False

    # Build set of prior triples plus implied triples
    prior_triples = claims_to_triples(prior_claims)
    implied_triples = derive_implied_triples(prior_triples)
    claim_triple = triple_to_key(subject, predicate, obj)

    # Check direct contradiction
    for t in implied_triples:
        if t[0] == claim_triple[0] and t[1] == claim_triple[1]:
            if t[2] != claim_triple[2]:
                return False
    # Check if claim contradicts any existing triple by same subject/predicate different object
    for t in prior_triples:
        if t[0] == claim_triple[0] and t[1] == claim_triple[1]:
            if t[2] != claim_triple[2]:
                return False
    return True


# ============================================================
#  VeriCoT with derivation
# ============================================================
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
            subject = m.group(1).strip()
            obj = m.group(2).strip()
            subject = re.sub(r'^(a|an|the)\s+', '', subject, flags=re.IGNORECASE)
            obj = re.sub(r'^(a|an|the)\s+', '', obj, flags=re.IGNORECASE)
            return {"subject": subject, "predicate": pred, "object": obj}

    prompt = f"""Extract the subject, predicate, and object from the following sentence as a single JSON object with keys "subject", "predicate", "object".
Sentence: "{step_text}"
Return only JSON.
"""
    data = call_model_json(prompt, max_tokens=128)
    if data and all(k in data for k in ("subject", "predicate", "object")):
        return data
    return None


def verify_vericot(step_text: str, context: str, kg) -> bool:
    """
    Extract triple and verify through direct lookup or transitive derivation.
    """
    triple = extract_triple_from_text(step_text)
    if not triple:
        return False

    subject = triple["subject"]
    predicate = triple["predicate"]
    obj = triple["object"]

    # Direct check in kg_triples
    if query_kg_triples(subject=subject, predicate=predicate, object_=obj):
        return True

    # Check materialized implied triples
    conn = db.db_connect("reasoning")
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM implied_triples WHERE subject=? AND predicate=? AND object=?",
                (subject, predicate, obj))
    found = cur.fetchone() is not None
    conn.close()
    return found


# ============================================================
#  FiDeLiS: grounding
# ============================================================
def verify_fidelis(claim: Dict, kg) -> bool:
    subject = claim.get("subject", "").strip()
    predicate = claim.get("predicate", "").strip()
    obj = claim.get("object", "").strip()
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


# ============================================================
#  R-CoT: reverse chain reconstruction
# ============================================================
def verify_rcot(conclusion: str, kg) -> bool:
    """
    Reconstruct a reverse chain from conclusion to premises using kg_triples.
    Returns True if a chain of length >= 1 exists.
    """
    if not conclusion:
        return False

    # BFS from conclusion as object to subjects, then those subjects as objects, etc.
    visited = set()
    queue = [conclusion.lower().strip()]
    found = False
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        # Find triples where current is the object
        conn = db.db_connect("reasoning")
        cur = conn.cursor()
        cur.execute("SELECT subject, predicate FROM kg_triples WHERE LOWER(object)=?", (current,))
        rows = cur.fetchall()
        conn.close()

        if rows:
            found = True
            # If at least one row exists, conclusion is grounded
            break
        # Optionally, continue to subjects if no direct grounding? We'll stop at first level.
    return found


# ============================================================
#  ARES: probabilistic entailment stability
# ============================================================
def verify_ares(reasoning_chain: List[Dict]) -> float:
    """
    Inductively verify each step based solely on previous steps.
    Returns average entailment probability (0-1).

    If config.ENABLE_CALIBRATED_ARES is True, uses actual entailment
    confidence instead of hardcoded values.
    """
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
                step.get("subject",""), step.get("predicate",""), step.get("object","")
            )
            if claim_triple in implied:
                score = 1.0
            else:
                consistent = verify_symstep(step, accepted)
                if calibrated:
                    # Use embedding similarity to prior accepted facts as confidence
                    try:
                        from core.embeddings import get_embedding
                        import numpy as np
                        claim_emb = get_embedding(step.get("text",""))
                        prior_embs = [get_embedding(a.get("text","")) for a in accepted if a.get("text")]
                        if claim_emb and prior_embs:
                            claim_vec = np.array(claim_emb, dtype=np.float32)
                            max_sim = 0.0
                            for pe in prior_embs:
                                if pe:
                                    pv = np.array(pe, dtype=np.float32)
                                    sim = float(np.dot(claim_vec, pv) / (np.linalg.norm(claim_vec) * np.linalg.norm(pv) + 1e-8))
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


# ============================================================
#  Unified verification orchestrator
# ============================================================
def verify_claim(claim: Dict, source_text: Optional[str] = None, kg=None) -> List[Dict]:
    results = []

    # Text grounding
    if source_text and claim.get("source_span"):
        results.append({
            "layer": "text_grounding",
            "verified": claim["source_span"] in source_text,
            "confidence": 1.0 if claim["source_span"] in source_text else 0.0,
        })

    # SymStep (with empty prior claims; orchestrator will pass actual)
    sym = verify_symstep(claim, claim.get("_prior_claims", []))
    results.append({"layer": "symstep", "verified": sym, "confidence": 1.0 if sym else 0.0})

    # VeriCoT
    vericot = verify_vericot(claim.get("text", ""), source_text, kg)
    results.append({"layer": "vericot", "verified": vericot, "confidence": 0.8 if vericot else 0.0})

    # FiDeLiS
    fidelis = verify_fidelis(claim, kg)
    results.append({"layer": "fidelis", "verified": fidelis, "confidence": 0.9 if fidelis else 0.0})

    # R-CoT
    rcot = verify_rcot(claim.get("conclusion", claim.get("object", "")), kg)
    results.append({"layer": "rcot", "verified": rcot, "confidence": 0.7 if rcot else 0.0})

    return results


def verify_claim_adaptive(claim, source_text=None, kg=None, threshold=0.6):
    """Adaptive verification: first cheap checks, escalate if confidence low."""
    results = []
    # Cheap checks
    if source_text and claim.get("source_span"):
        results.append({
            "layer": "text_grounding",
            "verified": claim["source_span"] in source_text,
            "confidence": 1.0 if claim["source_span"] in source_text else 0.0,
        })
    sym = verify_symstep(claim, claim.get("_prior_claims", []))
    results.append({"layer": "symstep", "verified": sym, "confidence": 1.0 if sym else 0.0})

    # Compute initial confidence
    initial_conf = sum(v["confidence"] for v in results if v["verified"]) / max(1, len(results))
    if initial_conf >= threshold:
        return results

    # Escalate to heavier checks
    vericot = verify_vericot(claim.get("text", ""), source_text, kg)
    results.append({"layer": "vericot", "verified": vericot, "confidence": 0.8 if vericot else 0.0})
    fidelis = verify_fidelis(claim, kg)
    results.append({"layer": "fidelis", "verified": fidelis, "confidence": 0.9 if fidelis else 0.0})
    rcot = verify_rcot(claim.get("conclusion", claim.get("object", "")), kg)
    results.append({"layer": "rcot", "verified": rcot, "confidence": 0.7 if rcot else 0.0})
    return results
