from core.llm import call_model

ANSWER_PROMPT = """
You are a knowledgeable assistant.
Answer the user's question using ONLY the provided context.
Prioritize facts with confidence > 0.6.
If there are conflicting facts, mention both and explain the discrepancy.
If the answer cannot be determined from the context, say so and explain what is missing.
Use Markdown formatting (headings, bullet points) for clarity.
Keep the answer concise but complete.

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
