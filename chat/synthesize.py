"""
Single answer-synthesis funnel for every chat path (CLI, API, WebUI, reasoning or not).

One adaptive prompt, one citation law, one token policy — replacing the three
divergent prompts that previously produced different quality per mode. Context
building stays with callers (tagged via build_tagged_context); this module owns
intent wording, prompt assembly, the model call, and output cleaning.
"""
import logging

logger = logging.getLogger(__name__)

INTENT_INSTRUCTIONS = {
    "summary": ("This request asks for a summary. Open with the core answer in "
                "2-3 sentences, then headed sections per theme covering every supplied "
                "fact at least once, a key-figures bullet list, a table of what is "
                "established versus what remains unclear, a key-references list "
                "naming the source documents behind the tags, and a closing verdict "
                "that adds judgment instead of recapping."),
    "factual": ("This request asks a factual question. Lead with the direct answer, "
                "then thorough supporting detail that works through every supplied "
                "fact at least once, using bullets for any enumeration of facts, "
                "figures, or dates, and close with what would strengthen the answer "
                "if the sources are thin."),
    "detail": ("This request asks for a detailed report. Produce full headed sections "
               "covering every supplied fact at least once — cite every [S#] tag in "
               "the context at least once — grouped by theme, with tables for figures, "
               "dates, or comparisons where they aid reading; add a table of what is "
               "established versus what remains unclear, a key-references list naming "
               "the source documents behind the tags, and a bottom line stating how "
               "provisional the picture is given the sources."),
    "comparative": ("This request asks for a comparison. Build a side-by-side table "
                    "of the compared items from supplied facts only, covering every "
                    "relevant fact at least once, then a verdict paragraph. Omit the "
                    "table rather than padding it when there is nothing to compare."),
    "causal": ("This request asks for causes. Lay out the causal chain step by step "
               "with a heading per link, each grounded in supplied facts, covering "
               "every relevant fact at least once, and note where links in the chain "
               "lack evidence."),
    "temporal": ("This request asks about time. Give an ordered timeline covering "
                 "every dated supplied fact, then narrative detail around it, and "
                 "note any gaps in the record."),
    "general": ("Answer thoroughly in prose, working through every supplied fact at "
                "least once — longer is better than leaving material unused. Add "
                "`##` headings and bullets whenever the answer runs long, a table "
                "of what is established versus unclear, and a key-references list. "
                "Match Markdown shape to content: headings for sections, bullets "
                "for lists, tables for comparisons."),
}

SHARED_QUALITY_RULES = """Quality rules for every response type: lead with the answer (no throat-clearing
like "fascinating" or "it's important to understand", no sycophancy); cite every
supplied [S#] tag at least once so no material is silently dropped, while never
repeating the same claim twice and never restating the introduction as the conclusion;
cite with [S#] tags on every sourced paragraph or bullet, grouping same-source claims
naturally; every figure, date, and proper name MUST carry its tag while plain
connective prose carries none; excerpts without tags may be cited short-form
(Document: filename); when all sources come from a single document, say what kind of
independent corroboration is missing instead of overstating certainty; never hedge
("debated", "moderate", "some argue") without a cited fact behind it."""

SYNTHESIZE_PROMPT = """You are a knowledgeable research assistant. Answer ONLY from the provided context, and use ALL of it that bears on the question — write as much as the material warrants, with no artificial length limit.

{intent_instruction}

{quality_rules}

Context (facts tagged [S1]..[Sn], highest signal first; excerpts untagged):
{context}

Question: {question}

Answer:
"""

CAUTIOUS_PROMPT = """The user asked: {question}

We could not find enough verified information. Answer in at most 3 sentences from the snippets below, clearly stating uncertainty, with a single [S#] citation when one applies.

Context:
{context}

Answer:"""

ANSWER_MAX_TOKENS_DEFAULT = 32768


def detect_request_intent(question):
    try:
        from chat.query_intent import detect_intent
        label = detect_intent(question or "")
        if label in INTENT_INSTRUCTIONS:
            return label
    except Exception:
        logger.warning("Unexpected exception occurred", exc_info=True)
    return "general"


def answer_max_tokens():
    try:
        import config as _cfg
        return int(getattr(_cfg, "CHAT_ANSWER_MAX_TOKENS", ANSWER_MAX_TOKENS_DEFAULT))
    except Exception:
        return ANSWER_MAX_TOKENS_DEFAULT


def build_prompt(question, context, intent=None):
    label = intent if intent in INTENT_INSTRUCTIONS else detect_request_intent(question)
    return SYNTHESIZE_PROMPT.format(context=context, question=question,
                                    intent_instruction=INTENT_INSTRUCTIONS[label],
                                    quality_rules=SHARED_QUALITY_RULES), label


def synthesize_answer(question, context, model=None, endpoint=None, endpoint_type="chat",
                      intent=None, max_tokens=None):
    """Render prompt through one funnel and return the cleaned answer string."""
    from core.llm import call_model
    from chat.responder import clean_answer
    prompt, _ = build_prompt(question, context, intent)
    _type = endpoint_type if (model is None and endpoint is None) else None
    raw = call_model(prompt, model=model, max_tokens=max_tokens or answer_max_tokens(),
                     endpoint=endpoint, endpoint_type=_type)
    return clean_answer(raw)


def synthesize_cautious(question, context):
    """Short uncertainty answer sharing the tag citation law."""
    from core.llm import call_model
    from chat.responder import clean_answer
    import config as _cfg
    prompt = CAUTIOUS_PROMPT.format(question=question, context=context)
    raw = call_model(prompt, max_tokens=min(getattr(_cfg, "CHAT_ANSWER_MAX_TOKENS", 4096), 4096))
    return clean_answer(raw)
