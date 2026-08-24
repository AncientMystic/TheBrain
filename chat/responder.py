from core.llm import call_model
from core.model_router import get_chat_endpoint

ANSWER_PROMPT = """
You are a knowledgeable conversational assistant.

Use the provided context to write a complete, detailed, and natural answer.

Formatting requirements:
- Format the answer using Markdown.
- Use **bold** for important key terms, names, or numbers when first mentioned.
- Write in full paragraphs as the primary style.
- Use bullet points for lists of characteristics, features, or steps when appropriate.
- Do not mention "the context" or "provided material" explicitly.
- If the information is limited, still provide a helpful, direct answer based on what is available.
- If there are conflicting facts, explain the different perspectives in plain language.
- Keep the answer focused on the user's question.

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
    return call_model(prompt, model=model, max_tokens=1024)

def generate_answer_with_reasoning(question, model=None):
    """Use the verification-first reasoning orchestrator."""
    from reasoning.orchestrator import orchestrate_reasoning
    answer, _ = orchestrate_reasoning(question)
    return answer
