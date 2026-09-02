from core.llm import call_model
from core.model_router import get_chat_endpoint

ANSWER_PROMPT = """
You are a knowledgeable research assistant.

Use the provided context to answer the user's question naturally and completely.

Guidelines:
- Match the style to the user's request. If they ask for a list, provide a bullet list. If they ask for detail, write thorough paragraphs.
- Use Markdown formatting appropriately: **bold** for key terms, headings only when they improve readability.
- Cite sources with `(Document: filename.pdf)` after relevant statements, but do not over-cite.
- If information is incomplete or uncertain, explain what is missing or conflicting.
- Keep the answer focused and conversational.

Context:
{context}

Question: {question}

Answer:
"""

def generate_answer(question, context, model=None, conversation_history=None):
    full_context = context
    if conversation_history:
        full_context = conversation_history + "\n\n" + context
    prompt = ANSWER_PROMPT.format(context=full_context, question=question)
    return call_model(prompt, model=model, max_tokens=32768)

def generate_answer_with_reasoning(question, _model=None):
    """Use the verification-first reasoning orchestrator."""
    from reasoning.orchestrator import orchestrate_reasoning
    answer, _ = orchestrate_reasoning(question)
    return answer
