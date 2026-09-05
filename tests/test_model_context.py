"""Dynamic model context: override precedence, safe fallbacks, budget floor."""
import config
from core import model_context as mc


def _endpoints():
    return [{"url": "http://127.0.0.1:9", "model": "unreachable-test-model"}]


def test_fallback_when_backends_unreachable():
    assert mc.pool_min_context(_endpoints(), fallback=4096) == 4096
    assert mc.pool_min_context([], fallback=4096) == 4096


def test_explicit_override_wins_without_probing(monkeypatch=None):
    old_single = getattr(config, "MODEL_MAX_CONTEXT", 0)
    old_json = getattr(config, "MODEL_CONTEXT_JSON", {})
    config.MODEL_MAX_CONTEXT = 0
    config.MODEL_CONTEXT_JSON = {"unreachable-test-model": 16384}
    try:
        assert mc.detect_endpoint_context(_endpoints()[0]) == 16384
    finally:
        config.MODEL_MAX_CONTEXT = old_single
        config.MODEL_CONTEXT_JSON = old_json


def test_pool_minimum_semantics():
    # Minimum across endpoints, never zero-drag from failures (tested via fallback above).
    assert mc.pool_min_context([{"url": "http://127.0.0.1:9", "model": "a"}], fallback=8192) == 8192


def test_budget_floor_and_label():
    chars, label = mc.answer_budget()
    assert chars >= 2000
    assert "token window" in label


def test_budget_fits_facts_first():
    from chat.context_builder import build_context
    facts = [{"fact_text": f"Fact number {i} about testing.", "confidence": 0.9 - i * 0.01,
              "doc_name": "t.pdf", "source_span": "testing"} for i in range(30)]
    full = build_context(facts)
    small = build_context(facts, budget_chars=800, model_label="~test window")
    assert len(small) < len(full)
    assert "omitted" in small
    assert "Fact number 0" in small  # highest-ranked survives
