"""Response-quality guards: hyphen artifacts, duplicate facts, word-safe cuts."""
from core.text_utils import dehyphenate, normalise_text
from chat.context_builder import _dedup_facts, _cut_words, build_context


def test_dehyphenate_joins_line_breaks():
    assert dehyphenate("man- tle xenoliths") == "mantle xenoliths"
    assert dehyphenate("cohe- sion of crust") == "cohesion of crust"
    assert dehyphenate("high- est point") == "highest point"


def test_dehyphenate_keeps_real_hyphens():
    assert dehyphenate("well- Known") == "well- Known"
    assert dehyphenate("1995 - 2000") == "1995 - 2000"


def test_normalise_applies_dehyphenation():
    assert "mantle" in normalise_text("man-\n tle")


def test_dedup_keeps_best_confidence_once():
    facts = [
        {"fact_text": "Xenoliths date to 2.7 Ga.", "confidence": 0.6},
        {"fact_text": "xenoliths  date to 2.7 ga.", "confidence": 0.9},
        {"fact_text": "Cato Trough separates Australia.", "confidence": 0.8},
    ]
    out = _dedup_facts(facts)
    assert len(out) == 2
    assert float(out[0]["confidence"]) == 0.9


def test_cut_words_never_mid_word():
    assert _cut_words("alpha beta gamma", 11) == "alpha beta"
    assert _cut_words("alpha beta gamma", 10) == "alpha"


def test_build_context_no_duplicate_lines():
    facts = [
        {"fact_text": "Same claim here.", "confidence": 0.5, "doc_name": "d.pdf", "source_span": "s"},
        {"fact_text": "Same claim here.", "confidence": 0.7, "doc_name": "d.pdf", "source_span": "s"},
    ]
    ctx = build_context(facts)
    assert ctx.count("Same claim here.") == 1
    assert "confidence 0.70" in ctx
