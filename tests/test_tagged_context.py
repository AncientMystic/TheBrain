"""Tagged context alignment + shared synthesis funnel."""
from chat.context_builder import build_tagged_context, build_context, build_tag_ledger
from chat import synthesize as _syn


def _facts():
    return [
        {"fact_text": "Beta claim here.", "confidence": 0.5, "doc_name": "b.pdf", "source_span": "s"},
        {"fact_text": "Alpha claim here.", "confidence": 0.9, "doc_name": "a.pdf", "source_span": "s"},
        {"fact_text": "beta CLAIM here.", "confidence": 0.7, "doc_name": "b.pdf", "source_span": "s"},
    ]


def test_tags_follow_first_retrieval_order():
    text, ordered, tagmap = build_tagged_context(_facts())
    assert [f["citation_tag"] for f in ordered] == ["S1", "S2"]
    # Rank wins the slot (first-retrieval position), best duplicate text wins it.
    assert ordered[0]["fact_text"].lower() == "beta claim here."
    assert float(ordered[0]["confidence"]) == 0.7
    assert tagmap["S1"]["doc"] == "b.pdf"
    assert "[S1]" in text and "[S2]" in text


def test_determinism_same_input_same_tags():
    t1, o1, _ = build_tagged_context(_facts())
    t2, o2, _ = build_tagged_context(_facts())
    assert t1 == t2
    assert [f["citation_tag"] for f in o1] == [f["citation_tag"] for f in o2]


def test_legacy_wrapper_still_text_only():
    assert isinstance(build_context(_facts()), str)


def test_ledger_maps_tags():
    _, ordered, _ = build_tagged_context(_facts())
    ledger = build_tag_ledger(ordered)
    assert "S1" in ledger and "b.pdf" in ledger


def test_budget_prefix_and_note():
    many = [{"fact_text": f"Distinct claim number {i} about testing.",
             "confidence": 0.9 - i * 0.01, "doc_name": "t.pdf", "source_span": "s"}
            for i in range(30)]
    text, ordered, _ = build_tagged_context(many, budget_chars=800, model_label="~test")
    assert ordered  # always keeps at least one
    assert "omitted" in text
    assert "Distinct claim number 0" in text  # highest-ranked survives
    assert "Distinct claim number 29" not in text


def test_prompt_has_no_bare_echo_token():
    prompt, label = _syn.build_prompt("Summarize the findings", "[S1] fact")
    assert label == "summary"
    assert "(summary)" not in prompt and "({})".format("summary") not in prompt
    # The [S#] notation is intentional instruction text (tag scheme), never a bare echo token.


def test_intent_instructions_cover_all_labels():
    from chat.query_intent import detect_intent  # noqa: F401 (documents the contract)
    for label in ["summary", "factual", "detail", "comparative", "causal", "temporal", "general"]:
        assert label in _syn.INTENT_INSTRUCTIONS


def test_full_coverage_rule_present():
    assert "at least once" in _syn.SHARED_QUALITY_RULES
    for label in ("summary", "detail", "general"):
        assert "at least once" in _syn.INTENT_INSTRUCTIONS[label]


def test_no_word_targets_anywhere():
    blob = _syn.SYNTHESIZE_PROMPT + _syn.SHARED_QUALITY_RULES + "".join(_syn.INTENT_INSTRUCTIONS.values())
    assert "400 words" not in blob


def test_references_and_uncertainty_scoping():
    blob = _syn.INTENT_INSTRUCTIONS["detail"] + _syn.INTENT_INSTRUCTIONS["general"]
    assert "key-references" in blob
    assert "unclear" in blob
    assert "corroboration" in _syn.SHARED_QUALITY_RULES or "provisional" in blob
