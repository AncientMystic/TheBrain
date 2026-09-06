"""FTS hostile keywords + mojibake repair (exact reported strings)."""
from core.text_utils import repair_mojibake


def test_repair_reported_strings():
    assert repair_mojibake("4.9â€¯million square kilometres") == "4.9\u202fmillion square kilometres"
    assert repair_mojibake("Newâ€¯Zealand and Newâ€¯Caledonia") == "New\u202fZealand and New\u202fCaledonia"
    out = repair_mojibake("~4000â€‘kmâ€‘long ribbon")
    assert out == "~4000\u2011km\u2011long ribbon", repr(out)


def test_repair_leaves_legitimate_text():
    assert repair_mojibake("forêt épaisse") == "forêt épaisse"
    assert repair_mojibake("São Paulo, naïve façade — plain text.") == "São Paulo, naïve façade — plain text."
    assert repair_mojibake("plain ascii, 1 < 2 math.") == "plain ascii, 1 < 2 math."
    assert repair_mojibake("") == ""
    assert repair_mojibake(None) is None


def test_safe_str_repairs():
    from extraction.cleaners import _safe_str
    assert _safe_str("4.9â€¯million", 200) == "4.9\u202fmillion"


def test_fts_hostile_keywords_no_exception():
    from graph.graph_queries import get_facts_by_keyword
    for kw in ("Patrick E. McGovern", "Elizabeth, New Jersey", 'say "hi" (loudly)',
               "Democratize technology access", "a:b*c", ""):
        res = get_facts_by_keyword(kw, limit=5)
        assert isinstance(res, list)


def test_fts_semicolon_parens_no_exception():
    from graph.graph_queries import get_facts_by_keyword
    res = get_facts_by_keyword("Stuart J. Fleming; Solomon (Katz)", limit=3)
    assert isinstance(res, list)
