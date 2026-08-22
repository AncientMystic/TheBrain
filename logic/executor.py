"""
Simple logic module executor.
Applies relevant logic modules to modify retrieval or answer generation.
"""
from core import db
from logic.retrieve import retrieve_logic_modules

def execute_logic_modules(query, context, top_k=3):
    """Return a modified context string with logic module instructions prepended."""
    if not config.LOGIC_EXECUTOR_ENABLED:
        return context
    modules = retrieve_logic_modules(query, top_k=top_k)
    instructions = []
    for sim, lid, name, category, summary, content_mod in modules:
        instructions.append(f"### Logic Module: {name} ({category})\n{content_mod[:500]}")
    if instructions:
        return "\n\n".join(instructions) + "\n\n" + context
    return context
