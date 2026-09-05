"""
Epistemic truth classes (OBSERVED / ASSERTED / INFERRED / REPUTED).

Origin-based, never collapsed into confidence: a disputed admin claim stays
ASSERTED (origin) with verification_status=disputed (state). The two axes are
orthogonal by design — ranking may use confidence, correctness must not.
Generic, no document-specific rules.
"""

TRUTH_CLASSES = ("OBSERVED", "ASSERTED", "INFERRED", "REPUTED")


def from_fact(fact):
    """Derive truth class from a fact dict's origin signals (deterministic)."""
    try:
        if not isinstance(fact, dict):
            return "REPUTED"
        truth_status = str(fact.get("truth_status", "") or "")
        if truth_status == "admin_claim":
            return "ASSERTED"
        verified_by = str(fact.get("verified_by", "") or "")
        if truth_status == "verified_true" or verified_by in ("verified_folder", "standards_match", "text_grounding"):
            return "OBSERVED"
        layers = fact.get("verification_layers") or []
        if any(isinstance(v, dict) and v.get("verified") for v in layers):
            return "OBSERVED" if fact.get("source_span") else "INFERRED"
        if str(fact.get("verification_status", "")) in ("verified", "partially_verified"):
            return "INFERRED"
        return "REPUTED"
    except Exception:
        return "REPUTED"


def validate_fact(fact):
    """Machine-checkable invariants for one fact. Returns list of violations."""
    problems = []
    try:
        tc = fact.get("truth_class", from_fact(fact))
        if tc not in TRUTH_CLASSES:
            problems.append(f"unknown truth_class {tc!r}")
        ts = str(fact.get("truth_status", "") or "")
        if ts == "admin_claim" and tc != "ASSERTED":
            problems.append("admin_claim must be ASSERTED")
        if ts == "verified_true" and tc not in ("OBSERVED", "ASSERTED"):
            problems.append("verified_true must be OBSERVED or ASSERTED")
    except Exception as e:
        problems.append(f"validator error: {e}")
    return problems
