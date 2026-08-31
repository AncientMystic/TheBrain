from core.llm import call_model
from core.model_router import get_chat_endpoint

ANSWER_PROMPT = """
You are a knowledgeable research assistant.

Use the provided context to write a comprehensive, detailed, and natural answer in Markdown.

Formatting requirements:
- Begin with a clear heading summarizing the topic.
- Use **bold** for key terms, names, or numbers when first mentioned.
- Write multiple paragraphs covering distinct aspects (e.g., overview, key characteristics, evidence, uncertainties, conclusion).
- Use bullet points or numbered lists for enumerated features, steps, or evidence.
- For each factual statement that comes from a specific document, include a short parenthetical citation like `(Document: filename.pdf)` at the end of the sentence or clause.
- If the information is incomplete or uncertain, explain what is missing and what remains unclear.
- If there are conflicting facts, explain the different perspectives in plain language.
- Keep the answer focused on the user's question.
- Do not mention "the context" or "provided material" explicitly.

Context:
{context}

Question: {question}

Answer (Markdown):
"""

def generate_answer(question, context, model=None, conversation_history=None):
    full_context = context
    if conversation_history:
        full_context = conversation_history + "\n\n" + context
    prompt = ANSWER_PROMPT.format(context=full_context, question=question)
    return call_model(prompt, model=model, max_tokens=32768)

def generate_answer_with_reasoning(question, model=None):
    """Use the verification-first reasoning orchestrator."""
    from reasoning.orchestrator import orchestrate_reasoning
    answer, _ = orchestrate_reasoning(question)
    return answer
