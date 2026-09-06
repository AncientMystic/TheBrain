"""Multi-hop convergence: diamond paths collapse to one fact, order kept."""
from graph.expansion import dedupe_facts_content


def _fact(fid, text, conf=0.5):
    return {"fact_id": fid, "fact_text": text, "confidence": conf}


def test_diamond_converges_to_single():
    # A->B->D and A->C->D deliver D twice under different ids.
    facts = [_fact(1, "A seeds the search."),
             _fact(2, "B follows A."), _fact(3, "C follows A."),
             _fact(4, "D concludes it all."), _fact(5, "d CONCLUDES it all. ")]
    out = dedupe_facts_content(facts)
    assert [f["fact_id"] for f in out] == [1, 2, 3, 4]


def test_shorter_hop_wins_ties():
    out = dedupe_facts_content([_fact(9, "Shared claim.", conf=0.4),
                                _fact(3, "Shared claim.", conf=0.9)])
    assert len(out) == 1 and out[0]["fact_id"] == 9


def test_empty_text_dropped_distinct_kept():
    out = dedupe_facts_content([_fact(1, "  "), _fact(2, "Real claim."), _fact(3, "Other claim.")])
    assert [f["fact_id"] for f in out] == [2, 3]


def test_no_input_safe():
    assert dedupe_facts_content([]) == []
    assert dedupe_facts_content(None) == []
