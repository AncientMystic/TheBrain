"""Offset span validation + fallback."""
from core.span_validation import validate_span


def test_exact():
    ok, sp, why = validate_span("Cats purr.", "Cats purr", "Cats purr loudly.")
    assert ok and sp == "Cats purr"


def test_offsets():
    chunk = "Hello world example"
    ok, sp, why = validate_span("world", "world", chunk, 6, 11)
    assert ok


def test_fallback_sentence():
    chunk = "Dogs bark loudly. Cats purr softly at night."
    ok, sp, why = validate_span("Cats purr softly", "missing span xyz", chunk)
    assert sp in chunk
    assert why in ("nearest-sentence-fallback", "invalid")
