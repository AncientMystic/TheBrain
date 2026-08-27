import json
import config
from core.llm import call_model_json
from core.embeddings import get_embedding
import numpy as np

DECOMPOSE_PROMPT = """
Decompose the following user query into atomic sub-questions.
Each sub-question must be:
- self-contained
- answerable from a knowledge graph or text
- have a verification method from ["entity_lookup", "relation_lookup", "text_span", "logical_check", "temporal_check"]

Return JSON with key "sub_questions" as list of objects:
{
  "id": "q1",
  "question": "...",
  "verification_method": "...",
  "dependencies": []
}

Query: {query}

Return only JSON.
"""

def decompose_query(query):
    prompt = DECOMPOSE_PROMPT.replace("{query}", query)
    data = call_model_json(prompt, max_tokens=512)
    if data and "sub_questions" in data:
        return data["sub_questions"]
    return [{"id": "q0", "question": query, "verification_method": "text_span", "dependencies": []}]