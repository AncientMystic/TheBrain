"""Faithfulness/hallucination judge (phase 79): build-then-measure.

LLM-as-judge on the enrolled backend (auto-loads) under a strict rubric,
with deterministic guardrails first and JSON validation after:
  - faithfulness: no citations -> unfaithful (structural, no LLM call).
  - labels whitelisted; anything else -> review (never a silent pass).
  - every verdict carries method (rules|llm) + reason for audit.
House semantics baked into the rubric: contradicted may use
well-established world knowledge for famous facts (inventors, capitals);
unsupported covers true-but-unstated (citation-boundedness).
Never raises from public functions (returns review on failure).
"""
import json

import config

FAITHFULNESS_RUBRIC = (
    "Decide whether the ANSWER is fully supported by the CITATIONS.\n"
    "Reply with exactly one JSON object: {\"label\": \"faithful\"|\"unfaithful\", "
    "\"reason\": \"<one sentence>\"} and nothing else.\n"
    "Rules: faithful iff EVERY factual claim in ANSWER is directly stated in "
    "at least one CITATION (a subset of a citation counts; combining multiple "
    "citations counts). Unfaithful if ANY claim is unstated, contradicted, or "
    "there are no citations. Real-world truth is IRRELEVANT — only what the "
    "citations state."
)
HALLUCINATION_RUBRIC = (
    "Decide the status of the CLAIM against the CONTEXT.\n"
    "Reply with exactly one JSON object: {\"label\": "
    "\"supported\"|\"contradicted\"|\"unsupported\", \"reason\": \"<one sentence>\"} "
    "and nothing else.\n"
    "Rules: supported iff CLAIM is directly stated or strictly implied by "
    "CONTEXT. Contradicted iff CONTEXT refutes it, or it clashes with "
    "well-established famous facts (inventors, capitals, celestial basics)."
    " Otherwise unsupported — including claims that are really true but absent "
    "from CONTEXT."
)

# v2 additions (phase 80): quote-grounding + atomic decomposition +
# contradicted preference. The judge must FIRST extract evidence spans
# (exact substrings), THEN decide on spans only. Confabulated support
# (F02-class) becomes structurally impossible: no quote, no faithful.


def _extract_spans(kind, material):
    """Step 1: extract exact-substring evidence spans. Returns list[str]."""
    try:
        from core.backends import create_backend
        eps = getattr(config, "LLM_ENDPOINTS", [])
        if not eps:
            return {}
        ep = eps[0]
        be = create_backend(ep)
        if kind == "faith":
            task = ("List the exact substrings (word-for-word quotes) from CITATIONS "
                    "that state each factual claim in ANSWER. Reply with exactly one JSON "
                    "object: {\"quotes\": [\"<exact substring>\", ...]} and nothing else. "
                    "Empty list if a claim has no supporting substring. Never paraphrase.")
        else:
            task = ("List exact substrings (word-for-word quotes) from CONTEXT that either "
                    "support or refute the CLAIM, and split the CLAIM into atomic sub-claims. "
                    "Reply with exactly one JSON object: {\"subclaims\": [\"...\", ...], "
                    "\"supports\": [\"<exact substring>\", ...], "
                    "\"refutes\": [\"<exact substring>\", ...]} and nothing else. "
                    "Empty lists where nothing applies. Never paraphrase.")
        msgs = [{"role": "system", "content": "You extract evidence spans. Exact substrings only."},
                {"role": "user", "content": task + "\n\nMATERIAL:\n" + str(material)[:4000]}]
        out = be.chat(msgs, model=ep.get("model"), max_tokens=256, temperature=0.0)
        txt = str(out.get("content", out) if isinstance(out, dict) else out or "").strip()
        start, end = txt.find("{"), txt.rfind("}")
        if start < 0 or end <= start:
            return []
        import json as _js
        obj = _js.loads(txt[start:end + 1])
        spans = []
        for k in ("quotes", "supports", "refutes"):
            for q in (obj.get(k) or []):
                if isinstance(q, str) and q.strip():
                    spans.append(q.strip())
        subclaims = [s for s in (obj.get("subclaims") or []) if isinstance(s, str)]
        obj["_spans"] = spans
        obj["_subclaims"] = subclaims
        return obj
    except Exception:
        return {}


def _verify_spans(spans, sources):
    """Keep only spans that occur verbatim in the source texts (anti-confabulation)."""
    try:
        hay = "\n".join(str(s or "") for s in sources)
        return [q for q in spans if q and q in hay]
    except Exception:
        return []


def _llm_judge(rubric, material, timeout=120):
    """Single rubric call. Returns (label|None, reason, raw)."""
    try:
        from core.backends import create_backend
        eps = getattr(config, "LLM_ENDPOINTS", [])
        if not eps:
            return None, "no LLM endpoints configured", ""
        ep = eps[0]
        be = create_backend(ep)
        msgs = [{"role": "system", "content": "You are a strict verification judge."},
                {"role": "user", "content": rubric + "\n\nMATERIAL:\n" + str(material)[:4000]}]
        out = be.chat(msgs, model=ep.get("model"), max_tokens=256, temperature=0.0)
        txt = str(out.get("content", out) if isinstance(out, dict) else out or "").strip()
        start, end = txt.find("{"), txt.rfind("}")
        if start < 0 or end <= start:
            return None, "non-JSON reply", txt[:200]
        obj = json.loads(txt[start:end + 1])
        label = str(obj.get("label", "")).strip().lower()
        reason = str(obj.get("reason", ""))[:300]
        return label or None, reason, txt[:200]
    except Exception as e:
        return None, f"backend error: {str(e)[:120]}", ""


def judge_faithfulness(answer, citations):
    """(label, reason, method) for answer + list of quote strings.

    v2: extract exact-substring quotes, verify them verbatim against the
    citations (confabulated support becomes impossible), then decide on
    verified quotes only. Falls back to v1 single-call if extraction
    yields nothing usable.
    """
    try:
        quotes = [str(c.get("quote", c) if isinstance(c, dict) else c or "").strip()
                  for c in (citations or [])]
        quotes = [q for q in quotes if q]
        if not quotes:
            return "unfaithful", "no citations provided", "rules"
        material = "ANSWER: %s\nCITATIONS:\n%s" % (
            str(answer or "").strip(),
            "\n".join(f"- {q}" for q in quotes))
        ev = _extract_spans("faith", material)
        verified = _verify_spans(ev.get("quotes") or [], quotes)
        if not verified:
            if not ev:
                return _judge_faithfulness_v1(answer, quotes)
            return "unfaithful", "no verified supporting quote in citations", "llm-2step"
        from core.backends import create_backend
        eps = getattr(config, "LLM_ENDPOINTS", [])
        be = create_backend(eps[0])
        task = ("Given ONLY these verified exact quotes, is EVERY factual claim in "
                "ANSWER directly stated? Reply with exactly one JSON object: "
                "{\"label\": \"faithful\"|\"unfaithful\", \"reason\": \"<one sentence>\"} "
                "and nothing else. Real-world truth is IRRELEVANT.")
        msgs = [{"role": "system", "content": "You are a strict verification judge."},
                {"role": "user", "content": task + "\n\nANSWER: " + str(answer or "").strip()
                 + "\nVERIFIED QUOTES:\n" + "\n".join(f"- {q}" for q in verified)}]
        out = be.chat(msgs, model=eps[0].get("model"), max_tokens=256, temperature=0.0)
        txt = str(out.get("content", out) if isinstance(out, dict) else out or "").strip()
        start, end = txt.find("{"), txt.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(txt[start:end + 1])
            label = str(obj.get("label", "")).strip().lower()
            if label in ("faithful", "unfaithful"):
                return label, str(obj.get("reason", ""))[:300], "llm-2step"
        return "review", "step-2 unparseable", "llm-2step"
    except Exception as e:
        return "review", f"judge failed: {str(e)[:120]}", "rules"


def _judge_faithfulness_v1(answer, quotes):
    """Original single-call path (fallback when extraction is unusable)."""
    material = "ANSWER: %s\nCITATIONS:\n%s" % (
        str(answer or "").strip(), "\n".join(f"- {q}" for q in quotes))
    label, reason, _ = _llm_judge(FAITHFULNESS_RUBRIC, material)
    if label in ("faithful", "unfaithful"):
        return label, reason, "llm"
    return "review", f"judge unparseable: {reason}", "llm"


def judge_hallucination(claim, context):
    """(label, reason, method) for claim + context string.

    v2: extract sub-claims + supporting/refuting spans, verify spans
    verbatim, then decide: any verified refutation -> contradicted
    (contradicted-preference kills the hedge); every sub-claim with a
    verified support and none refuted -> supported; else unsupported.
    Falls back to v1 when extraction is unusable.
    """
    try:
        claim_s, ctx_s = str(claim or "").strip(), str(context or "").strip()
        if not (claim_s and ctx_s):
            return "review", "empty claim or context", "rules"
        material = "CLAIM: %s\nCONTEXT: %s" % (claim_s, ctx_s)
        ev = _extract_spans("hal", material)
        if ev:
            sup = _verify_spans(ev.get("supports") or [], [ctx_s])
            ref = _verify_spans(ev.get("refutes") or [], [ctx_s])
            subs = [s for s in (ev.get("subclaims") or []) if s.strip()]
            if ref:
                return "contradicted", "verified refuting span in context", "llm-2step"
            if subs and sup and _covers(sup, subs, strict=True):
                return "supported", "every sub-claim has verified support", "llm-2step"
            if not subs and sup:
                return "supported", "verified supporting span in context", "llm-2step"
            # No verified spans either way: famous-fact check decides
            # contradicted vs unsupported (the clause v1 never fired).
            label, reason = _famous_fact_check(claim_s)
            if label == "contradicted":
                return label, reason, "llm-2step"
            # Otherwise: strict implication check on verified spans (saves
            # H01-class implication while rejecting H03-class number traps).
            return _implication_check(subs or [claim_s], sup)
        label, reason, _ = _llm_judge(HALLUCINATION_RUBRIC, material)
        if label in ("supported", "contradicted", "unsupported"):
            return label, reason, "llm"
        return "review", f"judge unparseable: {reason}", "llm"
    except Exception as e:
        return "review", f"judge failed: {str(e)[:120]}", "rules"


_STOP = frozenset("the a an and or of in on at to for with is are was were be been by as from that this it its into over under".split())


def _covers(supports, subclaims, strict=True):
    """Strict: EVERY content token of every sub-claim appears in supports.

    Conservative by design (safe direction is human review). Near-misses
    fall through to the implication check, which allows genuine strict
    implication (H01-class) while rejecting number traps (H03-class).
    """
    try:
        hay = " ".join(supports).lower()
        for s in subclaims:
            toks = [t.strip(".,;:!?\"'()").lower() for t in str(s).split()]
            content = [t for t in toks if len(t) > 3 and t not in _STOP]
            if content and any(t not in hay for t in content):
                return False
        return True
    except Exception:
        return False


def _implication_check(subclaims, supports):
    """Do verified spans strictly imply every sub-claim? (label, reason, method)."""
    try:
        from core.backends import create_backend
        eps = getattr(config, "LLM_ENDPOINTS", [])
        be = create_backend(eps[0])
        task = ("Do these VERIFIED spans STRICTLY imply EVERY sub-claim (no bridging "
                "assumptions, no uncited numbers/nationalities/capitals)? Reply with "
                "exactly one JSON object: {\"label\": \"supported\"|\"unsupported\", "
                "\"reason\": \"<one sentence>\"} and nothing else.")
        msgs = [{"role": "system", "content": "You are a strict verification judge."},
                {"role": "user", "content": task + "\n\nSUB-CLAIMS:\n" +
                 "\n".join(f"- {s}" for s in subclaims) +
                 "\nVERIFIED SPANS:\n" + "\n".join(f"- {s}" for s in supports)}]
        out = be.chat(msgs, model=eps[0].get("model"), max_tokens=128, temperature=0.0)
        txt = str(out.get("content", out) if isinstance(out, dict) else out or "").strip()
        start, end = txt.find("{"), txt.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(txt[start:end + 1])
            label = str(obj.get("label", "")).strip().lower()
            if label in ("supported", "unsupported"):
                return label, str(obj.get("reason", ""))[:300], "llm-2step"
        return "review", "implication check unparseable", "llm-2step"
    except Exception as e:
        return "review", f"implication check failed: {str(e)[:120]}", "rules"


def _famous_fact_check(claim):
    """Decide contradicted vs unsupported for famous-fact clashes.

    Returns (label|None, reason). Only fires on direct contradiction by
    well-established facts; anything else returns (None, '') leaving the
    caller at unsupported.
    """
    try:
        from core.backends import create_backend
        eps = getattr(config, "LLM_ENDPOINTS", [])
        be = create_backend(eps[0])
        task = ("Does this CLAIM clash with a well-established famous fact "
                "(inventor of telephone/penicillin, national capitals, Moon's nature, "
                "who painted famous works)? Reply with exactly one JSON object: "
                "{\"label\": \"contradicted\"|\"unsupported\", \"reason\": \"<one sentence>\"} "
                "and nothing else. When in doubt, say unsupported.")
        msgs = [{"role": "system", "content": "You are a strict verification judge."},
                {"role": "user", "content": task + "\n\nCLAIM: " + str(claim)[:1000]}]
        out = be.chat(msgs, model=eps[0].get("model"), max_tokens=128, temperature=0.0)
        txt = str(out.get("content", out) if isinstance(out, dict) else out or "").strip()
        start, end = txt.find("{"), txt.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(txt[start:end + 1])
            label = str(obj.get("label", "")).strip().lower()
            if label in ("contradicted", "unsupported"):
                return label, str(obj.get("reason", ""))[:300]
        return None, ""
    except Exception:
        return None, ""
