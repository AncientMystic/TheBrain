
from core.llm import call_model
from core.model_router import get_chat_endpoint

ANSWER_PROMPT = """
You are a knowledgeable research assistant tasked with producing a detailed, report-style answer.

Use the provided context to write a comprehensive answer in Markdown format.

Formatting requirements:
- Begin with a clear title or heading summarizing the question.
- Use **bold** for key terms, names, or numbers when first mentioned.
- Write multiple paragraphs (at least 3-4), each covering a distinct aspect.
- Use bullet points or numbered lists for enumerated facts or steps.
- Include subheadings (e.g., "Introduction", "Key Findings", "Analysis", "Conclusion") as appropriate.
- For every factual statement that comes from a specific document, insert a numeric citation in square brackets, e.g., [1] or [2]. Place the citation immediately after the statement, before punctuation.
- Do not use any other citation format like [doc:1] or [doc:2]. Use only the plain number.
- At the end, include a "References" section listing each cited document number and its exact title from the context. Use a numbered list with one document per line.
- Do not mention "the context" or "provided material" explicitly.
- If information is limited, explain what is known and what is missing, but still provide a useful answer.
- If there are conflicting facts, explain the different perspectives.
- Keep the answer focused on the user's question; do not mix in unrelated documents.

IMPORTANT: The context is divided into '### Document [n]:' sections. Use the number from the header as the citation. Only use facts from a document if it is relevant to the question.

Context:
{context}

Question: {question}

Answer (Markdown, report style):
"""

def generate_answer(question, context, model=None, conversation_history=None):
    full_context = context
    if conversation_history:
        full_context = conversation_history + "\n\n" + context
    prompt = ANSWER_PROMPT.format(context=full_context, question=question)
    return call_model(prompt, model=model, max_tokens=32768)

def generate_answer_with_reasoning(question, model=None):
    from reasoning.orchestrator import orchestrate_reasoning
    answer, _ = orchestrate_reasoning(question)
    return answer
