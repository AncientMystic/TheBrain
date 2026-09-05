from core.llm import call_model
from core.model_router import get_chat_endpoint

ANSWER_PROMPT = """
You are a knowledgeable research assistant. Answer ONLY from the provided context, and use ALL of it that bears on the question — write as much as the material warrants, with no artificial length limit.

Response type — match the format to the request ({intent_label}):
- summary: open with the core answer in 2-3 sentences, then `##` sections per theme, a key-figures bullet list, and a closing verdict.
- factual: direct answer first, then thorough supporting detail; bullets for any enumeration of facts, figures, or dates.
- detail/report: full structured report with `##` headings grouped by theme; cover every relevant supplied fact; use tables for figures/dates/comparisons where they aid reading.
- comparative: a side-by-side table of the compared items, then a verdict paragraph.
- causal: the causal chain step by step with a heading per link.
- temporal: a chronological timeline (ordered list) followed by narrative detail.
- general: thorough prose; add `##` headings and bullets whenever the answer runs long.

Quality rules (apply to every response type):
- Lead with the answer, never with throat-clearing ("fascinating", "it's important to understand"). No sycophancy.
- Never repeat the same claim twice, and never restate the introduction as a conclusion — the verdict must add judgment, not recap.
- Cite with the short form `(Document: filename)` on every paragraph or bullet that states sourced claims; group same-source claims so citations read naturally. Every figure, date, and proper name MUST carry a citation; plain connective prose carries none.
- Never hedge ("debated", "moderate", "some argue") unless a cited fact supports the qualifier.
- Use **bold** for key terms only. Match Markdown to the shape of the content: headings for sections, bullets for lists, tables for comparisons — never a wall of undifferentiated prose.

Context (facts pre-deduped, confidence-scored, highest first):
{context}

Question: {question}

Answer:
"""

ANSWER_MAX_TOKENS_DEFAULT = 32768


def _answer_max_tokens():
    try:
        import config as _cfg
        return int(getattr(_cfg, "CHAT_ANSWER_MAX_TOKENS", ANSWER_MAX_TOKENS_DEFAULT))
    except Exception:
        return ANSWER_MAX_TOKENS_DEFAULT


def _intent_label(question):
    try:
        from chat.query_intent import detect_intent
        return detect_intent(question or "")
    except Exception:
        return "general"


def generate_answer(question, context, model=None, conversation_history=None,
                    endpoint=None, endpoint_type="chat"):
    """Thin wrapper over the shared synthesis funnel (chat/synthesize.py).

    Kept for backward compatibility (server.py, main.py CLI). New code should
    prefer synthesize_answer directly.
    """
    from chat.synthesize import synthesize_answer
    full_context = context
    if conversation_history:
        full_context = conversation_history + "\n\n" + context
    return synthesize_answer(question, full_context, model=model,
                             endpoint=endpoint, endpoint_type=endpoint_type)

def generate_answer_with_reasoning(question, model=None):
    """Use the verification-first reasoning orchestrator."""
    from reasoning.orchestrator import orchestrate_reasoning
    answer, _ = orchestrate_reasoning(question)
    return answer


def clean_answer(answer: str) -> str:
    """Normalize LLM answer citations: [doc:n] -> [n] and strip whitespace."""
    import re
    if not isinstance(answer, str):
        try:
            answer = str(answer)
        except Exception:
            return ""
    answer = re.sub(r'\[doc\s*:\s*(\d+)\]', r'[\1]', answer)
    return answer.strip()
