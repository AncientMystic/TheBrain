"""Embed-only-uncached: fully-cached chunks skip embedding with zero LLM calls."""
import config
from extraction import llm_extractor as _lx


def _seed(chunk, model):
    h = _lx._hash_text(chunk)
    rows = {
        "facts_entities_relationships": (
            {"facts": [{"fact_text": f"Seeded fact about {chunk[:20]}", "confidence": 0.9}],
             "entities": [], "relationships": []},
            _lx.FACTS_ENTITIES_PROMPT_BATCH, 8192),
        "people_locations_dates": (
            {"people": [], "locations": [], "dates": []},
            _lx.PEOPLE_LOCATIONS_DATES_PROMPT_BATCH, 4096),
        "events_discoveries_gems": (
            {"events": [], "discoveries": [], "gems": []},
            _lx.EVENTS_DISCOVERIES_GEMS_PROMPT_BATCH, 4096),
    }
    for cat, (row, tmpl, mt) in rows.items():
        _lx._set_cached(h, cat, model, mt, row, tmpl)


def test_fully_cached_indices_partial():
    _lx._init_cache()
    model = config.LLM_ENDPOINTS[0]["model"]
    chunks = ["alpha chunk one", "beta chunk two", "gamma chunk three"]
    _seed(chunks[0], model)
    _seed(chunks[1], model)
    # gamma: only 1 of 3 categories -> not fully cached
    h = _lx._hash_text(chunks[2])
    _lx._set_cached(h, "people_locations_dates", model, 4096, {"people": [], "locations": [], "dates": []},
                    _lx.PEOPLE_LOCATIONS_DATES_PROMPT_BATCH)
    got = _lx._fully_cached_indices(chunks)
    assert got == {0, 1}, f"expected {{0,1}}, got {got}"


def test_reingest_zero_embed_zero_llm(monkeypatch):
    _lx._init_cache()
    model = config.LLM_ENDPOINTS[0]["model"]
    chunks = ["delta reingest chunk", "epsilon reingest chunk"]
    for c in chunks:
        _seed(c, model)
    assert _lx._fully_cached_indices(chunks) == {0, 1}

    calls = {"embed": [], "llm": 0}

    def _no_embed(texts, **kw):
        calls["embed"].extend(texts)
        return [[0.1] * 16 for _ in texts]

    def _no_llm(*a, **k):
        calls["llm"] += 1
        raise AssertionError("LLM must not be called on full cache hit")

    monkeypatch.setattr(_lx, "get_embeddings_batch", _no_embed)
    monkeypatch.setattr(_lx, "call_model_json", _no_llm)
    monkeypatch.setattr(config, "FAST_EXTRACTOR_ENABLED", False)
    monkeypatch.setattr(config, "RECALL_AUGMENT_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "USE_DISTILLED_EXTRACTOR", False)
    monkeypatch.setattr(config, "EXTRACTION_SECOND_PASS", False, raising=False)

    out = _lx.extract_from_chunks(list(chunks))
    assert calls["embed"] == [], f"embed called for cached chunks: {calls['embed']}"
    assert calls["llm"] == 0
    assert len(out) == 2
    assert out[0]["facts"] and out[0]["facts"][0]["fact_text"].startswith("Seeded fact")
    assert out[1]["facts"] and out[1]["facts"][0]["fact_text"].startswith("Seeded fact")
