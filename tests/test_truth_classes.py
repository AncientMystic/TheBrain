"""Truth-class mapping keeps origin and state on separate axes."""
import json
from pathlib import Path
from core.truth import TRUTH_CLASSES, from_fact, validate_fact


def test_enum_stable():
    assert TRUTH_CLASSES == ("OBSERVED", "ASSERTED", "INFERRED", "REPUTED")


def test_origin_mapping():
    assert from_fact({"truth_status": "admin_claim"}) == "ASSERTED"
    assert from_fact({"truth_status": "verified_true"}) == "OBSERVED"
    assert from_fact({"verification_status": "verified", "source_span": "s",
                      "verification_layers": [{"verified": True}]}) == "OBSERVED"
    assert from_fact({"verification_status": "verified"}) == "INFERRED"
    assert from_fact({}) == "REPUTED"
    # Dispute never collapses origin: admin stays ASSERTED even when disputed.
    assert from_fact({"truth_status": "admin_claim",
                      "verification_status": "disputed"}) == "ASSERTED"


def test_invariants():
    assert validate_fact({"truth_status": "admin_claim"}) == []
    bad = dict(truth_status="admin_claim")
    bad["truth_class"] = "OBSERVED"
    assert validate_fact(bad) != []
    assert validate_fact({"truth_class": "BOGUS"}) != []


def test_golden_fixture_matches():
    fix = json.loads((Path(__file__).parent.parent / "fixtures" / "answer_golden" / "tagged_basic.json").read_text(encoding="utf-8"))
    from chat.context_builder import build_tagged_context
    facts = [dict(x) for x in fix["input"]["facts"]]
    text, ordered, _ = build_tagged_context(facts, **fix["input"]["kwargs"])
    assert [x.get("citation_tag") for x in ordered] == fix["expected"]["tags"]
    assert ordered[0].get("fact_text") == fix["expected"]["first_text"]
    for s in fix["expected"]["contains"]:
        assert s in text
