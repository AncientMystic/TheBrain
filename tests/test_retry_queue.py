"""Retry queue: same prompts, same merges as inline retries; prio-first, bounded."""
import config
from extraction import llm_extractor as _lx

_FAKE_EP = {"name": "fake", "backend": "lmstudio", "url": "http://127.0.0.1:9/v1",
            "model": "fake-model", "api_key": "x"}


def _run_batch(monkeypatch, workers, endpoints=None):
    import uuid as _uuid
    seen = {"prompts": []}
    _tag = _uuid.uuid4().hex[:8]

    def _stub(prompt, **kw):
        seen["prompts"].append(prompt)
        n_batch = prompt.count("Chunk ")
        if 'num_chunks' in prompt or n_batch > 1 or "chunks" in prompt[:500].lower():
            return {}
        return {"chunk_0": {"facts": [{"fact_text": "Recovered fact", "confidence": 0.9,
                                       "source_span": "Recovered fact"}],
                            "entities": [], "relationships": []}}

    monkeypatch.setattr(_lx, "call_model_json", _stub)
    monkeypatch.setattr(config, "LLM_ENDPOINTS", endpoints or [_FAKE_EP])
    monkeypatch.setattr(config, "RETRY_WORKERS", workers)
    monkeypatch.setattr(config, "RETRY_QUEUE_MAX", 64)
    monkeypatch.setattr(config, "EXTRACTION_SECOND_PASS", False, raising=False)
    monkeypatch.setattr(config, "FAST_EXTRACTOR_ENABLED", False)
    chunks = [f"alpha content here {_tag}", f"beta content here {_tag}"]
    out = _lx._process_batch(list(chunks), model=None, logic_context="",
                             endpoint=dict(_FAKE_EP), actual_model="fake-model",
                             batch_pre_extractions=None, batch_prio={0: False, 1: True})
    return out, seen


def test_solo_queue_deterministic_and_indexed(monkeypatch):
    out1, seen1 = _run_batch(monkeypatch, 1)
    out2, seen2 = _run_batch(monkeypatch, 1)
    assert out1 == out2, "serial drain must be deterministic"
    # 3 category batch calls (all empty -> every chunk solo-retried per category)
    n_solo = sum(1 for p in seen1["prompts"] if p.count("Chunk ") <= 1)
    assert n_solo == 2 * 3, f"expected 6 solo prompts, got {n_solo}"
    for i in (0, 1):
        assert any(f.get("fact_text") == "Recovered fact" for f in out1[i]["facts"]), \
            f"chunk {i} missing recovered fact"


def test_parallel_drain_matches_serial(monkeypatch):
    out_serial, _ = _run_batch(monkeypatch, 1)
    ep2 = dict(_FAKE_EP, name="fake2")
    out_par, _ = _run_batch(monkeypatch, 2, endpoints=[_FAKE_EP, ep2])
    assert out_par == out_serial, "parallel drain must merge identically to serial"


def test_drain_prio_first_and_bounded(monkeypatch):
    order = []

    def _stub(prompt, **kw):
        order.append(kw.get("_jid", prompt[:40]))
        return {"chunk_0": {"facts": []}}

    jobs = [{"id": f"j{i}", "seq": i, "prio": (i == 3), "prompt": f"p{i}",
             "max_tokens": 8, "category": "facts_entities_relationships",
             "endpoint": dict(_FAKE_EP), "actual_model": "fake-model"}
            for i in range(4)]
    monkeypatch.setattr(config, "RETRY_QUEUE_MAX", 64)
    monkeypatch.setattr(config, "RETRY_WORKERS", 1)
    monkeypatch.setattr(config, "LLM_ENDPOINTS", [dict(_FAKE_EP)])
    monkeypatch.setattr(_lx, "call_model_json", lambda prompt, **kw: (order.append(prompt), {"chunk_0": {}})[1])
    _lx._drain_retry_jobs(jobs)
    assert order[0] == "p3", f"priority job must run first, got {order[0]}"
    assert sorted(order) == ["p0", "p1", "p2", "p3"]

    order.clear()
    monkeypatch.setattr(config, "RETRY_QUEUE_MAX", 2)
    _lx._drain_retry_jobs(jobs)
    assert len(order) == 2 and order[0] == "p3", f"bound must keep prio + 1, got {order}"
