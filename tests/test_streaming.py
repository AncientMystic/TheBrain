"""Streaming proofs: SSE parse, thinking-filter fuzz, stream==blocking."""
import json
import random
import config
import core.backends.base as _base
from core import llm as _llm
from core.llm import _filter_thinking_tail


def _sse_response(chunks):
    lines = []
    for c in chunks:
        lines.append("data: " + json.dumps({"choices": [{"delta": {"content": c}}]}))
    lines.append("data: [DONE]")

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            pass

        def iter_lines(self, decode_unicode=True):
            return iter(lines)

    return _Resp()


def test_sse_parser(monkeypatch):
    import requests as _rq
    monkeypatch.setattr(_rq, "post", lambda *a, **k: _sse_response(["Hello", " world", "!"]))
    got = list(_base.stream_openai_sse("http://x/v1/chat/completions", {"stream": True}, {}))
    assert "".join(got) == "Hello world!"


def _flush(buf):
    import re as _re
    rest = _re.sub(r"<thinking>.*?</thinking>", "", buf, flags=_re.DOTALL)
    rest = _re.sub(r"<reasoning>.*?</reasoning>", "", rest, flags=_re.DOTALL)
    rest = _re.sub(r"<(thinking|reasoning)\b[^>]*>.*$", "", rest, flags=_re.DOTALL)
    if _re.search(r"<$", rest):
        rest = rest[:-1]
    else:
        rest = _re.sub(r"</?[A-Za-z][^>]*$", "", rest)
    return rest


def test_filter_fuzz_no_tags_leak():
    raws = [
        "Answer <thinking>secret reasoning here</thinking> continues <reasoning>more</reasoning> end.",
        "No tags at all, just 1 < 2 math and plain text.",
        "<thinking>leading block</thinking>Tail text here.",
        "A<thinking>x</thinking>B<thinking>y</thinking>C",
    ]
    rng = random.Random(11)
    trial = 0
    for raw in raws:
        for _ in range(50):
            trial += 1
            cuts = sorted(rng.sample(range(1, len(raw)), rng.randint(1, 5)))
            chunks, prev = [], 0
            for c in cuts + [len(raw)]:
                chunks.append(raw[prev:c])
                prev = c
            buf, emitted = "", []
            for ch in chunks:
                buf += ch
                safe, buf = _filter_thinking_tail(buf)
                emitted.append(safe)
            joined = "".join(emitted)
            assert "<thinking" not in joined, f"tag leaked trial {trial}"
            assert "<reasoning" not in joined, f"tag leaked trial {trial}"
            assert "secret" not in joined and " reasoning here" not in joined
            final = joined + _flush(buf)
            assert "<" not in final or "< 2" in final, f"leftover fragment trial {trial}: {final!r}"
            assert "1 < 2 math" in final if "1 < 2" in raw else True


def test_filter_plain_text_untouched():
    buf, hold = _filter_thinking_tail("plain text with 1 < 2 math, no tags.")
    assert (buf + hold).replace("<", "") != ""
    out, _ = _filter_thinking_tail("plain text, no tags at all.")
    assert out == "plain text, no tags at all."


def test_stream_matches_blocking(monkeypatch):
    from core.backends.lmstudio import Provider as _LM

    ep = {"model": "m", "url": "http://x/v1", "api_key": "x", "backend": "lmstudio"}
    monkeypatch.setattr(_LM, "chat", lambda self, messages=None, **kw: "The answer is 42.")
    monkeypatch.setattr(config, "LLM_ENDPOINTS", [dict(ep)])
    toks = list(_llm.call_model_stream("q?", model="m", endpoint_type=None))
    assert "".join(toks) == "The answer is 42."
    assert _llm.call_model("q?", model="m") == "The answer is 42."
