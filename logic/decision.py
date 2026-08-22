import json
from core.llm import call_model
from logic.retrieve import retrieve_logic_modules


def decide_logic_modules(query, context="", top_k=5):
    candidates = retrieve_logic_modules(query, top_k=top_k)
    if not candidates:
        return []

    summary_list = []
    for sim, logic_id, name, category, summary, content in candidates:
        summary_list.append({
            "logic_id": logic_id,
            "name": name,
            "category": category,
            "summary": summary
        })

    prompt = f"""
Given the task/query and the following logic module summaries, select which modules (if any) are relevant to better understand or process the task.
Return a JSON array of integers representing the `logic_id` values to use. If none apply, return [].
Do not include any extra text.

Task/Query:
{query}

Context (first part):
{context[:1000]}

Logic module summaries:
{json.dumps(summary_list, indent=2)}

Selected logic_ids (JSON array):
"""
    raw = call_model(prompt, max_tokens=200)
    try:
        selected = json.loads(raw)
        if isinstance(selected, list):
            return selected
    except:
        pass
    return []