import re
import config
from core.text_utils import tokenize
from extraction.rule_annotator import pre_annotate
from chat.query_intent import detect_intent
import logging
logger = logging.getLogger(__name__)


# Lazy-loaded ONNX FastExtractor for topic extraction
_fast_extractor = None


def _get_fast_extractor():
    global _fast_extractor
    if _fast_extractor is None:
        try:
            from fast_extractor.hybrid_extractor import FastExtractor
            _fast_extractor = FastExtractor()
        except Exception:
            _fast_extractor = None
    return _fast_extractor


def extract_topic_terms(query: str) -> list:
    """
    Extract topic-focused search terms from a conversational query.
    Uses ONNX NER + rule-based annotations by default.
    Optionally calls a small LLM if config.USE_LLM_TOPIC_EXTRACTION is True.
    """
    terms = []
    seen = set()

    # 1. Rule-based annotations (dates, locations, organizations, people, events)
    try:
        annotations = pre_annotate(query)
        for key in ("locations", "people", "organizations", "dates", "years", "events"):
            for item in annotations.get(key, []):
                text = item.get("text", "").strip()
                if text and text.lower() not in seen:
                    seen.add(text.lower())
                    terms.append(text)
    except Exception:
        logger.warning("Unexpected exception occurred", exc_info=True)
        pass

    # 2. ONNX NER if enabled
    if getattr(config, "USE_ONNX_TOPIC_EXTRACTION", True):
        extractor = _get_fast_extractor()
        if extractor is not None:
            try:
                onnx_result = extractor.extract(query)
                for ent in onnx_result.get("entities", []):
                    text = ent.get("text", "").strip()
                    if text and text.lower() not in seen:
                        seen.add(text.lower())
                        terms.append(text)
            except Exception:
                logger.warning("Unexpected exception occurred", exc_info=True)
                pass

    # 3. Long tokens and bigrams as fallback
    tokens = tokenize(query)
    long_tokens = [t for t in tokens if len(t) > 3 and t.lower() not in seen]
    for t in long_tokens:
        if t.lower() not in seen:
            seen.add(t.lower())
            terms.append(t)

    # Bigrams from long tokens
    if len(long_tokens) > 1:
        for i in range(len(long_tokens) - 1):
            bigram = f"{long_tokens[i]} {long_tokens[i+1]}"
            if bigram.lower() not in seen:
                seen.add(bigram.lower())
                terms.append(bigram)

    # 4. Optional LLM refinement
    if getattr(config, "USE_LLM_TOPIC_EXTRACTION", False):
        try:
            from core.llm import call_model_json
            prompt = (
                "Extract only the main topic keywords or key phrases from this user query. "
                "Return a JSON array of strings. Do not include instruction words.\n\n"
                f"Query: {query}\n\nKeywords:"
            )
            data = call_model_json(prompt, max_tokens=128, unwrap_list=False)
            if isinstance(data, list):
                llm_terms = [str(t).strip() for t in data if str(t).strip()]
                combined = []
                for t in llm_terms:
                    if t.lower() not in seen:
                        seen.add(t.lower())
                        combined.append(t)
                combined.extend(terms)
                terms = combined
        except Exception:
            logger.warning("Unexpected exception occurred", exc_info=True)
            pass

    if not terms:
        if config.DEBUG_VERBOSE:
            print("    (Warning: extract_topic_terms returned empty list)")
    return terms[:10]


MAX_QUERY_LENGTH = 4096  # characters

def analyze_query(query: str) -> dict:
    """Extract topics, entities, and intent from a user query.
       Truncates query to MAX_QUERY_LENGTH to prevent resource exhaustion."""
    if not isinstance(query, str):
        query = ""
    query = query.strip()[:MAX_QUERY_LENGTH]

    """
    Extract topic-focused keywords, entities, dates, locations from user query.
    Returns dict with lists of tokens and annotated entities.
    """
    keywords = extract_topic_terms(query)

    annotations = pre_annotate(query)
    entities = []
    for entity_type, items in annotations.items():
        for item in items:
            entities.append({"type": entity_type, "text": item["text"]})

    return {
        "original": query,
        "keywords": keywords,
        "entities": entities,
        "intent": detect_intent(query),
    }
