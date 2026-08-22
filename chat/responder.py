from core.llm import call_model

ANSWER_PROMPT = """
You are a knowledgeable assistant.
Answer the user's question using ONLY the provided context.
Cite document names and source spans when possible.
If the answer cannot be determined from the context, say so.

Context:
{context}

Question: {question}

Answer:
"""

def generate_answer(question, context, model=None):
    prompt = ANSWER_PROMPT.format(context=context, question=question)
    return call_model(prompt, model=model, max_tokens=1024)

def generate_answer_with_reasoning(question, model=None):
    """Use the verification-first reasoning orchestrator."""
    from reasoning.orchestrator import orchestrate_reasoning
    answer, _ = orchestrate_reasoning(question)
    return answer
