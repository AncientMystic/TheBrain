"""
Optional batch verification module.
"""
from reasoning.verify import verify_symstep, verify_vericot, verify_rcot


def verify_batch(facts):
    """Run advisory verification on a list of facts."""
    results = []
    prior = []
    for fact in facts:
        sym = verify_symstep(fact, prior)
        # Proper triple-sharing call: reuse merged keys when present so
        # vericot skips its duplicate extraction + LLM fallback.
        _t = None
        if isinstance(fact, dict) and fact.get("subject") and fact.get("predicate"):
            _t = {"subject": fact.get("subject"), "predicate": fact.get("predicate"),
                  "object": fact.get("object", "")}
        vericot = verify_vericot(fact.get("fact_text", ""), None, None, triple=_t)
        rcot = verify_rcot(fact.get("fact_text", ""), None)
        fact["_batch_sym_ok"] = sym
        fact["_batch_vericot_ok"] = vericot
        fact["_batch_rcot_ok"] = rcot
        results.append(fact)
        prior.append(fact)
    return results
