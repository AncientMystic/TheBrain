"""
Evidence-answer contract builder: typed Anchor -> Normalize -> Propose handoff.

Converts retrieval datapoints + query analysis into a contract-shaped dict that
validators (CI + runtime) can check before reasoning consumes it. Untyped dicts
remain the transport; this is the checked shape. Generic, no doc-specific logic.
"""
import itertools

_claim_seq = itertools.count(1)


def _truth_of(datapoint):
    try:
        from core.truth import from_fact
        return from_fact(datapoint if isinstance(datapoint, dict) else {})
    except Exception:
        return "REPUTED"


def build_contract(query, analysis=None, datapoints=None, chunks=None):
    """Build an evidence-answer/v1 handoff from retrieval outputs."""
    analysis = analysis or {}
    datapoints = datapoints or []
    chunks = chunks or []
    anchors = []
    for i, kw in enumerate(analysis.get("keywords", []) or []):
        anchors.append({"anchor_id": f"a-k{i}", "kind": "keyword",
                        "text": str(kw), "source": "query"})
    for i, ent in enumerate(analysis.get("entities", []) or []):
        name = ent.get("text") if isinstance(ent, dict) else str(ent)
        if name:
            anchors.append({"anchor_id": f"a-e{i}", "kind": "entity",
                            "text": str(name), "source": "query"})
    claims = []
    n = 0
    for dp in datapoints:
        if not isinstance(dp, dict):
            continue
        dtype = dp.get("type") or ("fact" if dp.get("fact_text") else "")
        if dtype != "fact":
            continue
        statement = str(dp.get("text", dp.get("fact_text", "")) or "").strip()
        if not statement:
            continue
        n += 1
        ev = [{"doc": str(dp.get("doc_name", dp.get("doc_hash", "unknown"))),
               "span": str(dp.get("source_span", "")),
               "chunk_id": dp.get("chunk_id")}]
        aids = [a["anchor_id"] for a in anchors
                if a["text"].lower() in statement.lower()] or ["a-k0"] if anchors else []
        try:
            conf = float(dp.get("confidence", 0) or 0)
        except Exception:
            conf = 0.0
        claims.append({"claim_id": f"c{n}", "statement": statement[:500],
                       "truth_class": _truth_of(dp), "confidence": max(0.0, min(1.0, conf)),
                       "anchor_ids": aids, "evidence": ev,
                       "explanation_status": "grounded" if ev[0]["span"] else "partial",
                       "policy_status": "review"})
    vectors = []
    for cid, sim in chunks:
        try:
            vectors.append({"chunk_id": int(cid), "similarity": max(0.0, min(1.0, float(sim)))})
        except Exception:
            continue
    return {"contract": "evidence-answer/v1", "query": str(query),
            "anchors": anchors, "claims": claims, "vector_candidates": vectors[:50]}


def validate_contract(doc):
    """Validate a handoff dict against contracts/evidence_answer.json.

    Returns list of problems (empty = valid). Uses jsonschema when available,
    else a structural fallback covering required keys and enums.
    """
    from pathlib import Path
    import json
    schema_path = Path(__file__).resolve().parent.parent / "contracts" / "evidence_answer.json"
    try:
        import jsonschema
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(doc, schema)
        return []
    except ImportError:
        pass
    except Exception as e:
        return [f"schema violation: {e}"]
    problems = []
    try:
        if doc.get("contract") != "evidence-answer/v1":
            problems.append("bad contract tag")
        if not doc.get("query"):
            problems.append("missing query")
        from core.truth import TRUTH_CLASSES
        for c in doc.get("claims", []):
            if c.get("truth_class") not in TRUTH_CLASSES:
                problems.append(f"claim {c.get('claim_id')}: bad truth_class")
            try:
                co = float(c.get("confidence", -1))
                if not 0.0 <= co <= 1.0:
                    problems.append(f"claim {c.get('claim_id')}: confidence out of range")
            except Exception:
                problems.append(f"claim {c.get('claim_id')}: bad confidence")
    except Exception as e:
        problems.append(f"validator error: {e}")
    return problems
