from core.llm import call_model
from core.model_router import get_chat_endpoint

ANSWER_PROMPT = """
You are a precise research assistant. Answer ONLY from the provided context.

Structure (follow exactly):
1. Open with 1-2 sentences that directly answer the question.
2. Then short sections with `##` headings only if the answer has distinct parts; otherwise use 2-4 tight paragraphs.
3. Use bullet lists for enumerations of facts, figures, or dates — never bury a list inside prose.
4. End with a 1-sentence verdict. Do NOT restate the introduction as a summary.

Citation discipline (strict):
- Cite at most ONCE per paragraph or bullet, using the short form `(Document: filename)`.
- Never cite the same document twice in a row; group same-source claims into one cited sentence.
- Every figure, date, and proper name MUST carry a citation. General connective prose carries none.
- If the context lacks the answer, say so in one sentence and stop. Never hedge with uncited qualifiers ("remains debated", "moderate uptake") unless a cited fact supports them.

Style:
- Prefer short sentences. Aim for ~400 words unless the user explicitly asks for detail.
- No filler openers ("fascinating", "it's important to understand"). No sycophancy.
- Use **bold** sparingly for key terms only.

Context (facts pre-deduped, confidence-scored, highest first):
{context}

Question: {question}

Answer:
"""

ANSWER_MAX_TOKENS_DEFAULT = 4096


def _answer_max_tokens():
    try:
        import config as _cfg
        return int(getattr(_cfg, "CHAT_ANSWER_MAX_TOKENS", ANSWER_MAX_TOKENS_DEFAULT))
    except Exception:
        return ANSWER_MAX_TOKENS_DEFAULT


def generate_answer(question, context, model=None, conversation_history=None):
    full_context = context
    if conversation_history:
        full_context = conversation_history + "\n\n" + context
    prompt = ANSWER_PROMPT.format(context=full_context, question=question)
    return call_model(prompt, model=model, max_tokens=_answer_max_tokens())

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
