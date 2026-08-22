import re
from core.text_utils import tokenize
from extraction.rule_annotator import pre_annotate


def analyze_query(query: str) -> dict:
    """
    Extract keywords, entities, dates, locations from user query.
    Returns dict with lists of tokens and annotated entities.
    """
    tokens = tokenize(query)
    annotations = pre_annotate(query)
    keywords = tokens
    entities = []
    for entity_type, items in annotations.items():
        for item in items:
            entities.append({"type": entity_type, "text": item["text"]})

    return {
        "original": query,
        "keywords": keywords,
        "entities": entities,
    }