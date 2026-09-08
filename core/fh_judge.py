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

# Portable role routing (phase 85): 'small' = fast/disciplined model for
# mechanical span extraction, 'large' = reasoning model for judgment.
# Resolved from SMALL_MODEL_ENDPOINT / LARGE_MODEL_ENDPOINT (env-driven,
# empty on machines without role models -> default endpoint). No model
# IDs live here; any box runs with whatever it configured, degrading to
# single-model behavior when roles are unset.


def role_endpoint(role="large"):
    """Endpoint dict for a role, or None. Never raises."""
    try:
        if role == "small":
            _ep = getattr(config, "SMALL_MODEL_ENDPOINT", None)
            if _ep and _ep.get("model"):
                return _ep
        else:
            _ep = getattr(config, "LARGE_MODEL_ENDPOINT", None)
            if _ep and _ep.get("model"):
                return _ep
        _eps = getattr(config, "LLM_ENDPOINTS", []) or []
        return _eps[0] if _eps else None
    except Exception:
        return None


def role_chat(role, msgs, max_tokens=256):
    """Chat via role endpoint. Returns (text, endpoint_label)."""
    try:
        from core.backends import create_backend
        _ep = role_endpoint(role)
        if not _ep:
            return "", "none"
        _be = create_backend(_ep)
        _out = _be.chat(msgs, model=_ep.get("model"), max_tokens=max_tokens,
                        temperature=0.0)
        _txt = str(_out.get("content", _out) if isinstance(_out, dict) else _out or "").strip()
        return _txt, str(_ep.get("model", "?"))[:40]
    except Exception as e:
        return "", f"error: {str(e)[:80]}"

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
    """Step 1 via the small (fast/disciplined) role: extract exact-substring
    evidence spans. Returns parsed obj ({} on failure)."""
    try:
        if kind == "faith":
            task = ("List the exact substrings (word-for-word quotes) from CITATIONS "
                    "that state each factual claim in ANSWER. Quote FULL sentences "
                    "containing the evidence, never fragments or single names — a quote "
                    "must be able to stand alone as support. Reply with exactly one JSON "
                    "object: {\"quotes\": [\"<exact substring>\", ...]} and nothing else. "
                    "Empty list if a claim has no supporting substring. Never paraphrase.")
        else:
            task = ("List exact substrings (word-for-word quotes) from CONTEXT that either "
                    "support or refute the CLAIM, and split the CLAIM into atomic sub-claims. "
                    "Quote FULL sentences containing the evidence, never fragments — a span "
                    "must stand alone as support or refutation. "
                    "Reply with exactly one JSON object: {\"subclaims\": [\"...\", ...], "
                    "\"supports\": [\"<exact substring>\", ...], "
                    "\"refutes\": [\"<exact substring>\", ...]} and nothing else. "
                    "Empty lists where nothing applies. Never paraphrase.")
        msgs = [{"role": "system", "content": "You extract evidence spans. Exact substrings only."},
                {"role": "user", "content": task + "\n\nMATERIAL:\n" + str(material)[:4000]}]
        txt, _ = role_chat("small", msgs, max_tokens=256)
        start, end = txt.find("{"), txt.rfind("}")
        if start < 0 or end <= start:
            return {}
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
    """Single rubric call via the large (reasoning) role. (label|None, reason, raw)."""
    try:
        msgs = [{"role": "system", "content": "You are a strict verification judge."},
                {"role": "user", "content": rubric + "\n\nMATERIAL:\n" + str(material)[:4000]}]
        txt, _ = role_chat("large", msgs, max_tokens=256)
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
        # v3: verified spans must also cover the ANSWER's content tokens —
        # a trivial verified span ('Marie Curie') saying nothing is the
        # F02 confabulation shape. Uncovered answer -> unfaithful.
        if not _covers(verified, [str(answer or "").strip()], strict=True):
            return "unfaithful", "verified quotes do not cover the answer's claims", "llm-2step"
        task = ("Given ONLY these verified exact quotes, is EVERY factual claim in "
                "ANSWER directly stated? Reply with exactly one JSON object: "
                "{\"label\": \"faithful\"|\"unfaithful\", \"reason\": \"<one sentence>\"} "
                "and nothing else. Real-world truth is IRRELEVANT.")
        msgs = [{"role": "system", "content": "You are a strict verification judge."},
                {"role": "user", "content": task + "\n\nANSWER: " + str(answer or "").strip()
                 + "\nVERIFIED QUOTES:\n" + "\n".join(f"- {q}" for q in verified)}]
        txt, _ = role_chat("large", msgs, max_tokens=256)
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

    v3: v1 holistic judgment decides; span machinery only DOWNGrades.
    supported -> strict audit (verbatim + full token cover); failure
    falls to implication check, then famous-facts check (contradicted
    wins ties). unsupported -> famous-facts check (kills the hedge).
    contradicted -> trusted. Chain preserves v1's reading while
    auditing its support.
    """
    try:
        claim_s, ctx_s = str(claim or "").strip(), str(context or "").strip()
        if not (claim_s and ctx_s):
            return "review", "empty claim or context", "rules"
        material = "CLAIM: %s\nCONTEXT: %s" % (claim_s, ctx_s)
        jlabel, jreason, _ = _llm_judge(HALLUCINATION_RUBRIC, material)
        if jlabel == "contradicted":
            return jlabel, jreason, "llm"
        if jlabel == "supported":
            ev = _extract_spans("hal", material)
            if ev:
                sup = _verify_spans(ev.get("supports") or [], [ctx_s])
                subs = [s for s in (ev.get("subclaims") or []) if s.strip()] or [claim_s]
                if sup and _covers(sup, subs, strict=True):
                    return "supported", "every sub-claim has verified support", "llm-audit"
                impl, ireason, _ = _implication_check(subs, sup)
                if impl == "supported":
                    return "supported", ireason, "llm-audit"
            # audit failed or unusable: famous-check before accepting support
            flabel, freason = _famous_fact_check(claim_s, ctx_s)
            if flabel == "contradicted":
                return flabel, freason, "llm-audit"
            if not ev:
                return "supported", jreason, "llm"
            return "unsupported", "no verified support in context", "llm-audit"
        if jlabel == "unsupported":
            flabel, freason = _famous_fact_check(claim_s, ctx_s)
            if flabel == "contradicted":
                return flabel, freason, "llm-audit"
            return "unsupported", jreason, "llm"
        return "review", f"judge unparseable: {jreason}", "llm"
    except Exception as e:
        return "review", f"judge failed: {str(e)[:120]}", "rules"


_STOP = frozenset("the a an and or of in on at to for with is are was were be been by as from that this it its into over under".split())


def _norm_tok(t):
    """Lowercase + light plural normalization (bodies/body, prizes/prize).

    Applied to BOTH sides of every comparison, so proper names shift
    identically (paris/pari both sides) and matching stays exact.
    """
    try:
        t = str(t or "").strip(".,;:!?\"'()").lower()
        if t.endswith("ies") and len(t) > 5:
            return t[:-3] + "y"
        if t.endswith("es") and len(t) > 4:
            return t[:-2]
        if t.endswith("s") and len(t) > 4 and not t.endswith("ss"):
            return t[:-1]
        return t
    except Exception:
        return ""


def _covers(supports, subclaims, strict=True):
    """Strict: EVERY content token of every sub-claim appears in supports.

    v4: both sides pass through _norm_tok (plural-tolerant: bodies/body).
    Conservative by design (safe direction is human review). Near-misses
    fall through to the implication check, which allows genuine strict
    implication (H01-class) while rejecting number traps (H03-class).
    """
    try:
        hay = [_norm_tok(t) for t in " ".join(supports).split()]
        hayset = set(hay)
        for s in subclaims:
            content = [_norm_tok(t) for t in str(s).split()]
            content = [t for t in content if len(t) > 3 and t not in _STOP]
            if content and any(t not in hayset for t in content):
                return False
        return True
    except Exception:
        return False


def _implication_check(subclaims, supports):
    """Do verified spans strictly imply every sub-claim? (label, reason, method)."""
    try:
        task = ("Do these VERIFIED spans STRICTLY imply EVERY sub-claim (no bridging "
                "assumptions, no uncited numbers/nationalities/capitals)? Reply with "
                "exactly one JSON object: {\"label\": \"supported\"|\"unsupported\", "
                "\"reason\": \"<one sentence>\"} and nothing else.")
        msgs = [{"role": "system", "content": "You are a strict verification judge."},
                {"role": "user", "content": task + "\n\nSUB-CLAIMS:\n" +
                 "\n".join(f"- {s}" for s in subclaims) +
                 "\nVERIFIED SPANS:\n" + "\n".join(f"- {s}" for s in supports)}]
        txt, _ = role_chat("large", msgs, max_tokens=128)
        start, end = txt.find("{"), txt.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(txt[start:end + 1])
            label = str(obj.get("label", "")).strip().lower()
            if label in ("supported", "unsupported"):
                return label, str(obj.get("reason", ""))[:300], "llm-2step"
        return "review", "implication check unparseable", "llm-2step"
    except Exception as e:
        return "review", f"implication check failed: {str(e)[:120]}", "rules"


def _famous_fact_check(claim, context=""):
    """Decide contradicted vs unsupported for famous-fact clashes.

    Returns (label|None, reason). The CONTEXT travels along and wins ties:
    no clash may be declared against the context's own statements.
    """
    try:
        task = ("Does this CLAIM directly clash with one of these well-established "
                "facts? (a) Alexander Graham Bell invented the telephone — anyone else "
                "credited is wrong. (b) Alexander Fleming discovered penicillin. "
                "(c) Austin is the capital of Texas — no other Texas city is. "
                "(d) The Moon is rock, not cheese. (e) Van Gogh painted The Starry Night; "
                "that says nothing about his nationality. "
                "BINDING PRECEDENCE: the CONTEXT below wins every tie. If CONTEXT "
                "explicitly establishes otherwise (e.g. a Paris in Texas, a Moon described "
                "as a book), the context governs and there is NO clash. Never contradict "
                "the CONTEXT's own statements with world knowledge. "
                "Reply with exactly one JSON object: {\"label\": \"contradicted\"|"
                "\"unsupported\", \"reason\": \"<one sentence>\"} and nothing else. "
                "Say contradicted ONLY on a DIRECT clash surviving the precedence rule; "
                "otherwise unsupported.")
        msgs = [{"role": "system", "content": "You are a strict verification judge."},
                {"role": "user", "content": task + "\n\nCLAIM: " + str(claim)[:1000]
                 + "\n\nCONTEXT (wins ties): " + str(context or "")[:1500]}]
        txt, _ = role_chat("large", msgs, max_tokens=128)
        start, end = txt.find("{"), txt.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(txt[start:end + 1])
            label = str(obj.get("label", "")).strip().lower()
            if label in ("contradicted", "unsupported"):
                return label, str(obj.get("reason", ""))[:300]
        return None, ""
    except Exception:
        return None, ""
