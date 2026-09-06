"""Feature-uptake proofs: embed breaker skip + expansion analysis memo."""
import config
from core import breaker as _brk
from core import embeddings as _emb
from reasoning import orchestrator as _orc

_DEAD = {"name": "dead", "backend": "lmstudio", "url": "http://127.0.0.1:9/v1",
         "model": "dead-model", "api_key": "x"}
_LIVE = {"name": "live", "backend": "lmstudio", "url": "http://127.0.0.1:10/v1",
         "model": "live-model", "api_key": "x"}


class _StubProvider:
    calls = []

    def __init__(self, endpoint):
        self.ep = endpoint

    def embeddings(self, texts, model=None):
        type(self).calls.append(self.ep["url"])
        if self.ep["url"] == _DEAD["url"]:
            raise ConnectionError("refused")
        return [[0.1] * 4 for _ in texts]


def test_embed_skips_open_circuit(monkeypatch):
    _brk.reset()
    _StubProvider.calls = []
    monkeypatch.setattr(_emb, "create_backend", lambda ep: _StubProvider(ep))
    monkeypatch.setattr(config, "BREAKER_ENABLED", True)
    monkeypatch.setattr(config, "BREAKER_FAILURE_THRESHOLD", 2)
    monkeypatch.setattr(config, "BREAKER_COOLDOWN_SECONDS", 60)
    _brk.record_failure(_DEAD)
    _brk.record_failure(_DEAD)
    assert _brk.is_open(_DEAD)
    assert _emb._fetch_batch_with_endpoint(dict(_DEAD), ["hello"]) == []
    assert _StubProvider.calls == [], "provider hit while circuit open"
    out = _emb._fetch_batch_with_endpoint(dict(_LIVE), ["hello"])
    assert out == [("hello", [0.1] * 4)]
    assert not _brk.is_open(_LIVE)
    _brk.reset()


def test_analyze_many_memoizes(monkeypatch):
    calls = []
    monkeypatch.setattr(_orc, "analyze_query", lambda t: (calls.append(t), {"keywords": [t], "entities": []})[1])
    memo = {}
    texts = ["aaa", "bbb", "aaa"]
    first = _orc._analyze_many(texts, memo)
    assert calls == ["aaa", "bbb"], f"expected deduped misses, got {calls}"
    calls.clear()
    second = _orc._analyze_many(texts, memo)
    assert calls == [], "memo must serve repeats with zero calls"
    assert [a["keywords"] for a in second] == [["aaa"], ["bbb"], ["aaa"]]
    assert first == second
