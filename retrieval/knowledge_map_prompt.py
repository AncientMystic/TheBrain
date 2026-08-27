"""
Knowledge map selection prompt for hierarchical retrieval.
"""


def format_knowledge_map(datapoints, query, max_items=None):
    """Create a compact text map of datapoints."""
    if max_items is None:
        import config
        max_items = getattr(config, "MAX_MAP_NODES", 200)

    lines = [f'Knowledge Map for query: "{query}"', ""]
    for dp in datapoints[:max_items]:
        prefix = dp.get("type", "item")
        text = dp.get("text", "")[:200]
        lines.append(f"- {prefix}:{dp.get('id')} | {text}")
    return "\n".join(lines)


SELECTION_PROMPT = """You are a knowledge map navigator.

Below is a compact knowledge map relevant to the user's query.

{map_text}

Select the nodes that are most relevant to answering the query in detail.

Return a JSON array of node IDs to expand.

Rules:
- Only select nodes directly related to the query.
- Prefer nodes that answer "what", "why", "how", or "detail".
- If a fact is sufficient on its own, do not select its chunk_ref.
- If the fact is too short or ambiguous, select its chunk_ref.
- Maximum {max_selected} nodes.

User query: {query}

Selected node IDs (JSON array):
"""
