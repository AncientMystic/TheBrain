"""GLiNER backend: parity-or-better vs bert NER, flag-gated with fallback."""
import config
from fast_extractor.gliner_onnx import GlinerONNXExtractor


def test_gliner_finds_key_entities():
    gx = GlinerONNXExtractor()
    assert gx.available, "GLiNER model failed to load"
    res = dict(((e[1], e[0]) for e in []))
    out = gx.extract_entities("Marie Curie won the Nobel Prize in Physics in 1903 in Paris.")
    by_type = {}
    for text, typ, conf in out:
        by_type.setdefault(typ, []).append(text)
    assert any("Marie Curie" in t for t in by_type.get("PERSON", [])), by_type
    assert any("Paris" in t for t in by_type.get("LOC", [])), by_type
    assert any("1903" in t for t in by_type.get("DATE", [])), by_type
    assert all("." not in t[-1:] or t[-1:].isalnum() for t in by_type.get("LOC", [])), by_type


def test_hybrid_gliner_path_and_fallback(monkeypatch):
    from fast_extractor.hybrid_extractor import FastExtractor
    monkeypatch.setattr(config, "GLINER_ENABLED", True)
    fx = FastExtractor()
    assert fx.gliner_extractor is not None and fx.gliner_extractor.available
    r = fx.extract("CRISPR was developed by Doudna in 2012.")
    assert any("Doudna" in e["text"] for e in r["people"]), r
    monkeypatch.setattr(config, "GLINER_ENABLED", False)
    fx2 = FastExtractor()
    assert fx2.gliner_extractor is None
    r2 = fx2.extract("CRISPR was developed by Doudna in 2012.")
    assert r2["entities"], "bert fallback must still extract"
