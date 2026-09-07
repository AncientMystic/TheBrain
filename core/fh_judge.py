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
    "well-established famous facts (inventors, capitals, celestial basics). "
    "Otherwise unsupported — including claims that are really true but absent "
    "from CONTEXT."
)


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
    """(label, reason, method) for answer + list of quote strings."""
    try:
        quotes = [str(c.get("quote", c) if isinstance(c, dict) else c or "").strip()
                  for c in (citations or [])]
        quotes = [q for q in quotes if q]
        if not quotes:
            return "unfaithful", "no citations provided", "rules"
        material = "ANSWER: %s\nCITATIONS:\n%s" % (
            str(answer or "").strip(),
            "\n".join(f"- {q}" for q in quotes))
        label, reason, _ = _llm_judge(FAITHFULNESS_RUBRIC, material)
        if label in ("faithful", "unfaithful"):
            return label, reason, "llm"
        return "review", f"judge unparseable: {reason}", "llm"
    except Exception as e:
        return "review", f"judge failed: {str(e)[:120]}", "rules"


def judge_hallucination(claim, context):
    """(label, reason, method) for claim + context string."""
    try:
        if not (str(claim or "").strip() and str(context or "").strip()):
            return "review", "empty claim or context", "rules"
        material = "CLAIM: %s\nCONTEXT: %s" % (
            str(claim).strip(), str(context).strip())
        label, reason, _ = _llm_judge(HALLUCINATION_RUBRIC, material)
        if label in ("supported", "contradicted", "unsupported"):
            return label, reason, "llm"
        return "review", f"judge unparseable: {reason}", "llm"
    except Exception as e:
        return "review", f"judge failed: {str(e)[:120]}", "rules"
