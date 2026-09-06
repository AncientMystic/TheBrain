"""Circuit breaker: dead endpoints fail fast, recover after cooldown/success."""
import time
import config
from core import llm as _llm
from core import breaker as _brk


class _DeadBackend:
    calls = 0

    def chat(self, *a, **k):
        type(self).calls += 1
        raise ConnectionError("refused")


class _LiveBackend:
    def chat(self, *a, **k):
        return "hello world"


def _ep():
    return {"name": "dead", "backend": "lmstudio", "url": "http://127.0.0.1:9/v1",
            "model": "dead-model", "api_key": "x"}


def test_opens_and_skips(monkeypatch):
    _brk.reset()
    _DeadBackend.calls = 0
    monkeypatch.setattr(_llm, "create_backend", lambda ep: _DeadBackend())
    monkeypatch.setattr(config, "API_RETRY_ATTEMPTS", 1)
    monkeypatch.setattr(config, "API_RETRY_BACKOFF", 0)
    monkeypatch.setattr(config, "BREAKER_ENABLED", True)
    monkeypatch.setattr(config, "BREAKER_FAILURE_THRESHOLD", 2)
    monkeypatch.setattr(config, "BREAKER_COOLDOWN_SECONDS", 60)
    ep = _ep()
    assert _llm.call_model("hi", endpoint=dict(ep)) == ""
    assert _llm.call_model("hi", endpoint=dict(ep)) == ""
    assert _brk.is_open(ep)
    t0 = time.perf_counter()
    assert _llm.call_model("hi", endpoint=dict(ep)) == ""
    dt = time.perf_counter() - t0
    assert dt < 2.0, f"open circuit still slow: {dt:.1f}s"
    assert _DeadBackend.calls == 2, "provider hit while circuit open"
    _brk.reset()


def test_success_closes(monkeypatch):
    _brk.reset()
    monkeypatch.setattr(_llm, "create_backend", lambda ep: _LiveBackend())
    monkeypatch.setattr(config, "API_RETRY_ATTEMPTS", 1)
    monkeypatch.setattr(config, "API_RETRY_BACKOFF", 0)
    monkeypatch.setattr(config, "BREAKER_ENABLED", True)
    ep = _ep()
    _brk.record_failure(ep)
    out = _llm.call_model("hi", endpoint=dict(ep))
    assert out == "hello world"
    assert not _brk.is_open(ep)
    _brk.reset()


def test_disabled_never_opens(monkeypatch):
    _brk.reset()
    monkeypatch.setattr(config, "BREAKER_ENABLED", False)
    ep = _ep()
    for _ in range(5):
        _brk.record_failure(ep)
    assert not _brk.is_open(ep)
    _brk.reset()
