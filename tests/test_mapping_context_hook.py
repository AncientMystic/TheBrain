"""Server hookup: Known block lands in chat context (flag-gated)."""
import config
import server
from core import mapping_lookup as _ml


def _capture(captured):
    def _fake(q, ctx):
        captured["ctx"] = ctx
        return "ok"
    return _fake


def test_known_block_in_context(monkeypatch):
    captured = {}
    monkeypatch.setattr(server, "generate_answer", _capture(captured))
    monkeypatch.setattr("memory.bus.retrieve", lambda *a, **k: [])
    monkeypatch.setattr(config, "MAPPING_CONTEXT", True)
    answer, facts = server._process_chat(
        [server.ChatMessage(role="user", content="Tell me about Paris")],
        session_id="test-mapping-hook")
    assert answer == "ok"
    assert "Known:" in captured["ctx"] and "Paris" in captured["ctx"], captured["ctx"][:300]


def test_flag_off_means_absent(monkeypatch):
    captured = {}
    monkeypatch.setattr(server, "generate_answer", _capture(captured))
    monkeypatch.setattr("memory.bus.retrieve", lambda *a, **k: [])
    monkeypatch.setattr(config, "MAPPING_CONTEXT", False)
    server._process_chat(
        [server.ChatMessage(role="user", content="Tell me about Paris")],
        session_id="test-mapping-hook")
    assert "Known:" not in captured["ctx"]
