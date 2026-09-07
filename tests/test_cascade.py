"""GLiNER-first cascade: covered chunks skip the people/locations/dates LLM call."""
import config
from extraction import llm_extractor as _lx

_FAKE_EP = {"name": "fake", "backend": "lmstudio", "url": "http://127.0.0.1:9/v1",
            "model": "fake-model", "api_key": "x"}

RICH = ("Marie Curie and Pierre Curie worked in Paris France on radium research "
        "with laboratories institutes colleagues publications discoveries awards history")
THIN = "zzqx unrelated filler words without any named things whatsoever here"


def _pre_for(text, ents):
    return {"entities": [{"type": t, "text": e, "confidence": 0.9} for t, e in ents],
            "people": [{"type": "PERSON", "text": e, "confidence": 0.9} for t, e in ents if t == "PERSON"],
            "locations": [{"type": "LOC", "text": e, "confidence": 0.85} for t, e in ents if t == "LOC"],
            "dates": [], "organizations": []}


def test_coverage_flags_rich_only(monkeypatch):
    monkeypatch.setattr(config, "GLINER_CASCADE_ENABLED", True)
    pre0 = _pre_for(RICH, [("PERSON", "Marie Curie"), ("PERSON", "Pierre Curie"),
                           ("LOC", "Paris"), ("LOC", "France"), ("MISC", "radium")])
    done = _lx._gliner_entity_complete_idx([RICH, THIN], [pre0, None])
    assert done == {0}, f"expected only rich chunk covered, got {done}"


def test_people_category_skips_covered(monkeypatch):
    import uuid as _uuid
    _tag = _uuid.uuid4().hex[:8]
    _rich = RICH + f" uniquemarker{_tag}"
    _thin = THIN + f" uniquemarker{_tag}"
    monkeypatch.setattr(config, "GLINER_CASCADE_ENABLED", True)
    monkeypatch.setattr(config, "EXTRACTION_SECOND_PASS", False, raising=False)
    monkeypatch.setattr(config, "LLM_ENDPOINTS", [dict(_FAKE_EP)])
    calls = []

    def _stub(prompt, **kw):
        calls.append(prompt)
        return {}

    monkeypatch.setattr(_lx, "call_model_json", _stub)
    pre0 = _pre_for(RICH, [("PERSON", "Marie Curie"), ("PERSON", "Pierre Curie"),
                           ("LOC", "Paris"), ("LOC", "France"), ("MISC", "radium")])
    out = _lx._process_batch([_rich, _thin], model=None, logic_context="",
                             endpoint=dict(_FAKE_EP), actual_model="fake-model",
                             batch_pre_extractions=[pre0, None])
    # Identify people-category prompts by their distinctive instruction line
    # (the words "people"/"locations" appear in every template).
    _marker = "extract ALL relevant people, locations, and dates"
    people_prompts = [p for p in calls if _marker in p]
    assert people_prompts, "expected at least one people-category call"
    assert all("Marie Curie" not in p for p in people_prompts), "covered chunk reached people LLM"
    assert any("zzqx" in p for p in people_prompts), "thin chunk must still reach people LLM"
    # Covered chunk served from pre-pass conversion.
    assert any("Marie Curie" in (e.get("person_name", "") + e.get("text", ""))
               for e in out[0]["people"]), out[0]["people"]
    assert any("Paris" in (e.get("location_name", "") + e.get("text", ""))
               for e in out[0]["locations"]), out[0]["locations"]
    # Facts category still ran for both (LLM does the heavy lifting there).
    assert any("Marie Curie" in p or "zzqx" in p for p in calls)


def test_cascade_off_runs_everything(monkeypatch):
    monkeypatch.setattr(config, "GLINER_CASCADE_ENABLED", False)
    pre0 = _pre_for(RICH, [("PERSON", "Marie Curie"), ("PERSON", "Pierre Curie"),
                           ("LOC", "Paris"), ("LOC", "France"), ("MISC", "radium")])
    done = _lx._gliner_entity_complete_idx([RICH], [pre0])
    assert done == set()
