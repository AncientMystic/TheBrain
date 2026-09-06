"""Ingest consolidation: one verify_claim call == old three-call block's fields,
with zero LLM extraction when triple keys are already present."""
import json
from reasoning.verify import verify_claim, verify_symstep, verify_rcot


def _old_fields(fact, prior, extract_fn):
    sym_contradiction = 0
    formal_repr = None
    rcot_verified = 0
    try:
        if not verify_symstep(fact, prior):
            sym_contradiction = 1
    except Exception:
        pass
    try:
        triple = extract_fn(fact.get("fact_text", ""))
        if triple and all(k in triple for k in ("subject", "predicate", "object")):
            formal_repr = json.dumps(triple)
    except Exception:
        pass
    if fact.get("confidence", 0) < 0.7:
        try:
            if verify_rcot(fact.get("fact_text", ""), None):
                rcot_verified = 1
        except Exception:
            pass
    return sym_contradiction, formal_repr, rcot_verified


def _new_fields(fact, prior):
    _ctext = fact.get("fact_text", "")
    _claim = {"text": _ctext, "conclusion": _ctext,
              "source_span": fact.get("source_span", ""),
              "subject": fact.get("subject", ""),
              "predicate": fact.get("predicate", ""),
              "object": fact.get("object", fact.get("canonical_value", "")),
              "_prior_claims": prior}
    _layers = verify_claim(_claim) or []
    _by = {l.get("layer"): l for l in _layers if isinstance(l, dict)}
    sym_contradiction = 0 if _by.get("symstep", {}).get("verified", True) else 1
    rcot_verified = 1 if _by.get("rcot", {}).get("verified") else 0
    _t = {k: fact.get(k, "") for k in ("subject", "predicate", "object")}
    if _t.get("subject") and _t.get("predicate"):
        formal_repr = json.dumps(_t)
    else:
        # Mirrors main.py: extraction fallback only when keys are absent.
        from reasoning.verify import extract_triple_from_text as _ext
        triple = _ext(_ctext)
        if triple and all(k in triple for k in ("subject", "predicate", "object")):
            formal_repr = json.dumps(triple)
    return sym_contradiction, formal_repr, rcot_verified


def _facts():
    base = {"confidence": 0.5, "source_span": "Zealandia spans seas",
            "subject": "Zealandia", "predicate": "spans", "object": "seas"}
    f1 = dict(base, fact_text="Zealandia spans seas.")
    f2 = dict(base, fact_text="Zealandia spans oceans.")
    return [f1, f2]


def test_fields_match_with_keys(monkeypatch):
    import reasoning.verify as _v
    calls = []

    def _stub_extract(text):
        calls.append(text)
        return {"subject": "Zealandia", "predicate": "spans", "object": "seas"}

    monkeypatch.setattr(_v, "extract_triple_from_text", _stub_extract)
    facts = _facts()
    for i, fact in enumerate(facts):
        prior = facts[:i]
        old = _old_fields(fact, prior, _stub_extract)
        new = _new_fields(fact, prior)
        assert new[0] == old[0], f"fact {i}: sym differs {new[0]} vs {old[0]}"
        assert (json.loads(new[1]) if new[1] else None) == (json.loads(old[1]) if old[1] else None), \
            f"fact {i}: formal_repr differs"
        assert new[2] == old[2], f"fact {i}: rcot differs {new[2]} vs {old[2]}"


def test_no_extract_when_keys_present(monkeypatch):
    import reasoning.verify as _v
    calls = []
    monkeypatch.setattr(_v, "extract_triple_from_text", lambda t: (calls.append(t), {"subject": "x"})[1])
    facts = _facts()
    for i, fact in enumerate(facts):
        _new_fields(fact, facts[:i])
    assert calls == [], f"LLM extraction fired despite present keys: {calls}"


def test_fallback_fires_without_keys(monkeypatch):
    import reasoning.verify as _v
    calls = []
    monkeypatch.setattr(_v, "extract_triple_from_text", lambda t: (calls.append(t), {"subject": "s", "predicate": "p", "object": "o"})[1])
    monkeypatch.setattr(_v, "verify_vericot", lambda *a, **k: True)
    fact = {"fact_text": "Something happened.", "confidence": 0.5}
    _new_fields(fact, [])
    # vericot without keys falls back to extraction inside verify layers
    assert len(calls) >= 1, "fallback extraction must fire when keys are absent"
