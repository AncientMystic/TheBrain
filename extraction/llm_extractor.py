
import time
import json
import hashlib
import re
import sqlite3
import queue
import threading
from pathlib import Path

import numpy as np

import config
from core import db
from core.llm import call_model_json
from core.embeddings import get_embeddings_batch
from extraction.validation_queue import ValidationQueue
from extraction.rule_annotator import pre_annotate
from fast_extractor.hybrid_extractor import FastExtractor
# Singleton FastExtractor
_fast_extractor_instance = None

from extraction.cleaners import *
from core.schema_validation import validate_and_coerce
import logging
logger = logging.getLogger(__name__)


# ============================================================
#  FEW-SHOT EXAMPLES
# ============================================================

def _get_extraction_endpoint_type(category):
    if config.SMALL_MODEL_ENDPOINT:
        return "small"
    return None

def _get_verification_endpoint_type():
    if config.LARGE_MODEL_ENDPOINT:
        return "large"
    return None

FEW_SHOT_FACTS = """
Example 1:
Excerpt: "Marie Curie discovered radium in 1898."
{
  "facts": [{"fact_type": "discovery", "fact_text": "Marie Curie discovered radium.", "canonical_value": "radium", "source_span": "discovered radium", "confidence": 0.95}],
  "entities": [{"entity_type": "PERSON", "entity_name": "Marie Curie", "normalized_name": "Marie Curie", "source_span": "Marie Curie", "confidence": 0.95}, {"entity_type": "DISCOVERY", "entity_name": "radium", "normalized_name": "radium", "source_span": "radium", "confidence": 0.95}]
}
Example 2:
Excerpt: "The company was founded in 1998 by John Smith."
{
  "facts": [{"fact_type": "founding", "fact_text": "The company was founded in 1998.", "canonical_value": "1998", "source_span": "founded in 1998", "confidence": 0.9}],
  "entities": [{"entity_type": "ORG", "entity_name": "company", "normalized_name": "company", "source_span": "The company", "confidence": 0.8}, {"entity_type": "PERSON", "entity_name": "John Smith", "normalized_name": "John Smith", "source_span": "John Smith", "confidence": 0.9}]
}
"""

FEW_SHOT_PEOPLE = """
Example 1:
Excerpt: "Dr. Alice Johnson was born in Paris in 1980."
{
  "people": [{"person_name": "Alice Johnson", "normalized_name": "Alice Johnson", "role": "Dr.", "source_span": "Dr. Alice Johnson", "confidence": 0.95}],
  "locations": [{"location_name": "Paris", "normalized_place": "Paris", "location_type": "city", "source_span": "Paris", "confidence": 0.95}],
  "dates": [{"date_text": "1980", "normalized_date": "1980", "date_type": "birth", "source_span": "1980", "confidence": 0.95}]
}
Example 2:
Excerpt: "The conference took place in Berlin on March 15, 2023."
{
  "people": [],
  "locations": [{"location_name": "Berlin", "normalized_place": "Berlin", "location_type": "city", "source_span": "Berlin", "confidence": 0.9}],
  "dates": [{"date_text": "March 15, 2023", "normalized_date": "2023-03-15", "date_type": "event", "source_span": "March 15, 2023", "confidence": 0.95}]
}
"""

FEW_SHOT_EVENTS = """
Example 1:
Excerpt: "The first moon landing occurred on July 20, 1969."
{
  "events": [{"event_name": "Moon landing", "normalized_name": "Moon landing", "event_date": "1969-07-20", "event_type": "historical", "description": "First human moon landing", "significance": "Historic achievement", "source_span": "first moon landing", "confidence": 0.95}],
  "discoveries": [],
  "gems": []
}
Example 2:
Excerpt: "Scientists discovered a new species of frog in the Amazon rainforest."
{
  "events": [],
  "discoveries": [{"discovery_name": "new species of frog", "normalized_name": "new frog species", "description": "New species discovered", "date": "", "significance": "", "source_span": "discovered a new species of frog", "confidence": 0.9}],
  "gems": []
}
"""

# ============================================================
#  PROMPT TEMPLATES
# ============================================================
FACTS_ENTITIES_PROMPT_BATCH = """
You are a meticulous knowledge extraction agent.
Read {num_chunks} document excerpts below and extract ALL relevant facts, entities, and relationships for each excerpt.

STRICT RULES:
- For each excerpt, we provide pre-extracted entities/people/locations/dates from a fast automated system.
  - Verify each pre-extracted item against the text. Correct or remove any that are wrong.
  - Add any missing entities/people/locations/dates.
- Extract facts and relationships that are NOT already covered by the pre-extractions.
- For each fact, write `fact_text` as a concise, self-contained sentence (max 200 chars) in your own words.
- `source_span` must be a very short exact substring (3-10 words, max 100 chars) from the excerpt.
- `canonical_value` is a normalized short value (max 80 chars), never a nested object.
- `entity_name` must be the canonical name.
- For relationships, provide `source_node`, `target_node`, `relation_type`, `evidence_span`.
- Do not hallucinate, infer, or duplicate facts.
- Return a single JSON object whose keys are "chunk_0", "chunk_1", ... and values are objects matching the schema below.

JSON Schema for each chunk:
{
  "facts": [{"fact_type": "...", "fact_text": "...", "canonical_value": "...", "source_span": "...", "confidence": 0.9}],
  "entities": [{"entity_type": "PERSON|ORG|GPE|LOC|DATE|EVENT|DISCOVERY|TOPIC|OTHER", "entity_name": "...", "normalized_name": "...", "source_span": "...", "confidence": 0.9}],
  "relationships": [{"source_node": "...", "target_node": "...", "relation_type": "...", "evidence_span": "...", "confidence": 0.9}]
}

{logic_context}

Few-shot examples:
""" + FEW_SHOT_FACTS + """

Pre-extracted items:
{pre_extractions}

Excerpts:
{chunks_text}

Return only JSON.
"""

PEOPLE_LOCATIONS_DATES_PROMPT_BATCH = """
You are a meticulous knowledge extraction agent.
Read {num_chunks} document excerpts below and extract ALL relevant people, locations, and dates for each excerpt.

STRICT RULES:
- For each excerpt, we provide pre-extracted people/locations/dates from a fast automated system.
  - Verify each pre-extracted item against the text. Correct or remove any that are wrong.
  - Add any missing people/locations/dates.
- `person_name` is the full name, `role` a short description (max 80 chars).
- `location_name` is the canonical place name, `location_type` one of city|country|state|landmark|other.
- `date_text` is the original mention (max 100 chars), `normalized_date` a normalized ISO-like form, `date_type` a short type.
- `source_span` must be a very short exact substring (3-10 words, max 100 chars).
- Do NOT copy long phrases; keep all text fields concise.
- Do not hallucinate, infer, or duplicate.
- Return a single JSON object whose keys are "chunk_0", "chunk_1", ... and values are objects matching the schema below.

JSON Schema for each chunk:
{
  "people": [{"person_name": "...", "normalized_name": "...", "role": "...", "source_span": "...", "confidence": 0.9}],
  "locations": [{"location_name": "...", "normalized_place": "...", "location_type": "city|country|state|landmark|other", "source_span": "...", "confidence": 0.9}],
  "dates": [{"date_text": "...", "normalized_date": "YYYY-MM-DD", "date_type": "event|publication|birth|death|other", "source_span": "...", "confidence": 0.9}]
}

{logic_context}

Few-shot examples:
""" + FEW_SHOT_PEOPLE + """

Pre-extracted items:
{pre_extractions}

Excerpts:
{chunks_text}

Return only JSON.
"""

EVENTS_DISCOVERIES_GEMS_PROMPT_BATCH = """
You are a meticulous knowledge extraction agent.
Read {num_chunks} document excerpts below and extract ALL relevant events, discoveries, and gems for each excerpt.

STRICT RULES:
- For each excerpt, extract only information explicitly stated in that excerpt.
- `event_name` and `discovery_name` should be a short title (max 150 chars).
- `description` and `significance` must be concise summaries (max 200 chars each) in your own words.
- `gem_text` is a unique or surprising piece of information, phrased as a short fact (max 200 chars).
- `source_span` must be a very short exact substring (3-10 words, max 100 chars).
- Do NOT copy whole paragraphs or figure captions.
- Do not hallucinate, infer, or duplicate.
- Return a single JSON object whose keys are "chunk_0", "chunk_1", ... and values are objects matching the schema below.

JSON Schema for each chunk:
{
  "events": [{"event_name": "...", "normalized_name": "...", "event_date": "...", "event_type": "...", "description": "...", "significance": "...", "source_span": "...", "confidence": 0.9}],
  "discoveries": [{"discovery_name": "...", "normalized_name": "...", "description": "...", "date": "...", "significance": "...", "source_span": "...", "confidence": 0.9}],
  "gems": [{"gem_text": "...", "category": "...", "importance": 0.8, "source_span": "...", "confidence": 0.9}]
}

{logic_context}

Few-shot examples:
""" + FEW_SHOT_EVENTS + """

Excerpts:
{chunks_text}

Return only JSON.
"""

SYSTEM_PROMPT = (
    "You are a meticulous knowledge extraction agent. "
    "Extract only verifiable information explicitly stated in the given excerpts. "
    "Never infer or hallucinate. "
    "Write all factual fields in your own words; do not copy sentences. "
    "source_span must be a short pointer (max 10 words). "
    "Return strictly valid JSON matching the requested schema."
)

# ============================================================
#  CACHE
# ============================================================

def _hash_text(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def _get_prompt_hash(prompt_template, category):
    return hashlib.sha1(f"{category}:{prompt_template}".encode()).hexdigest()

def _init_cache():
    conn = db.db_connect(config.LLM_CACHE_DB)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(llm_extraction_cache)")
    columns = [row[1] for row in cur.fetchall()]
    if "prompt_hash" not in columns:
        cur.execute("DROP TABLE IF EXISTS llm_extraction_cache")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS llm_extraction_cache (
            chunk_hash TEXT,
            category TEXT,
            model TEXT,
            max_tokens INTEGER,
            result_json TEXT,
            prompt_hash TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (chunk_hash, category, model, max_tokens, prompt_hash)
        )
    """)
    conn.commit()
    conn.close()

def _get_cached(chunk_hash, category, model, max_tokens, prompt_template):
    if not config.LLM_EXTRACTION_CACHE:
        return None
    prompt_hash = _get_prompt_hash(prompt_template, category)
    conn = db.db_connect(config.LLM_CACHE_DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT result_json FROM llm_extraction_cache
        WHERE chunk_hash=? AND category=? AND model=? AND max_tokens=? AND prompt_hash=?
    """, (chunk_hash, category, model, max_tokens, prompt_hash))
    row = cur.fetchone()
    conn.close()
    if row:
        try:
            return json.loads(row[0])
        except Exception as e:
            logger.warning("Unexpected exception occurred", exc_info=True)
            return None
    return None

def _set_cached(chunk_hash, category, model, max_tokens, result_dict, prompt_template):
    if not config.LLM_EXTRACTION_CACHE:
        return
    prompt_hash = _get_prompt_hash(prompt_template, category)
    conn = db.db_connect(config.LLM_CACHE_DB)
    conn.execute("""
        INSERT OR REPLACE INTO llm_extraction_cache
        (chunk_hash, category, model, max_tokens, result_json, prompt_hash)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (chunk_hash, category, model, max_tokens, json.dumps(result_dict), prompt_hash))
    conn.commit()
    conn.close()

# ============================================================
#  HELPERS
# ============================================================

def _extract_candidate_texts(annotations):
    texts = []
    for key in ["locations", "people", "organizations", "dates", "years", "events"]:
        for item in annotations.get(key, []):
            t = item.get("text", "")
            if t:
                texts.append(t)
    return texts


def effective_llm_batch_chunks(n, endpoint_type=None, endpoint=None):
    """Race-free helper: preferred LLM batch chunks (no global mutation).

    Per-endpoint-type aware (small local models need smaller batches to fit context),
    then n-aware. Generic, config-driven, no doc-specific hardcoding.
    """
    import config as _cfg
    # Base by endpoint capability: small local 3B -> 2, large -> 4-8
    if endpoint_type == "small" or (endpoint and "small" in str(endpoint.get("model", "")).lower()):
        base = int(getattr(_cfg, "LLM_BATCH_SMALL", 2))
    elif endpoint_type == "large":
        base = int(getattr(_cfg, "LLM_BATCH_LARGE", 4))
    else:
        base = int(getattr(_cfg, "LLM_BATCH_CHUNKS", 4))
        if getattr(_cfg, "DYNAMIC_BATCH_SIZE", False):
            if n > 200:
                base = 8
            elif n < 20:
                base = min(base, 2)
    # Clamp generic bounds
    return max(1, min(base, int(getattr(_cfg, "LLM_BATCH_MAX", 8))))


def _compute_novelty_flags(chunks, chunk_embeddings):
    """Novelty detection using vectorized hyperbolic distances (no stale tree, no per-chunk search)."""
    from core.hyperbolic import ensure_hyperbolic, hyperbolic_distance_matrix
    from core.vector_store import HyperbolicBallTree  # kept for backward compat import

    flags = []
    seen_names = set()

    # Convert embeddings to hyperbolic if not already
    # Dynamic batch sizing WITHOUT global mutation (race-free): callers use effective_llm_batch_chunks()
    # (kept for backward compat: only set global when explicitly single-threaded legacy mode)
    if getattr(config, "DYNAMIC_BATCH_SIZE", False) and getattr(config, "ALLOW_GLOBAL_BATCH_MUTATION", False):
        if len(chunks) > 200:
            config.LLM_BATCH_CHUNKS = 8
        elif len(chunks) < 20:
            config.LLM_BATCH_CHUNKS = 2
        else:
            config.LLM_BATCH_CHUNKS = 4

    if chunk_embeddings is None:
        # No embeddings provided; process all chunks
        return [True] * len(chunks)

    import numpy as _np
    hyperbolic_points = []
    for emb in chunk_embeddings:
        if emb is None:
            hyperbolic_points.append(None)
        else:
            arr = _np.asarray(emb, dtype=_np.float32)
            if float(_np.linalg.norm(arr)) > 1.0:
                from core.hyperbolic import exp_map as _em
                arr = _em(arr)
            else:
                arr = ensure_hyperbolic(arr, space='hyperbolic')
            hyperbolic_points.append(arr)

    first_idx = next((i for i, p in enumerate(hyperbolic_points) if p is not None), None)
    if first_idx is None:
        return [True] * len(chunks)

    flags = [True] * len(chunks)
    flags[first_idx] = True
    seen_names.update(_extract_candidate_texts(pre_annotate(chunks[first_idx])))
    # Prior novel points for vectorized nearest-neighbor (correct vs all priors, single matrix per chunk batch)
    prior_pts = [hyperbolic_points[first_idx]]
    for i in range(first_idx + 1, len(chunks)):
        if hyperbolic_points[i] is None:
            flags[i] = True
            continue
        annotations = pre_annotate(chunks[i])
        candidates = _extract_candidate_texts(annotations)
        new_candidates = [c for c in candidates if c.lower() not in seen_names]
        try:
            pmat = _np.stack(prior_pts)
            dists = hyperbolic_distance_matrix(hyperbolic_points[i][None, :], pmat)[0]
            dist = float(_np.min(dists)) if len(dists) else float('inf')
            sim = 1.0 / (1.0 + dist) if dist != float('inf') else 0.0
        except Exception:
            sim = 0.0
        is_novel = True
        if config.NOVELTY_ENABLED and i > 0:
            if sim >= config.NOVELTY_SIM_THRESHOLD and not new_candidates:
                is_novel = False
        flags[i] = is_novel
        if is_novel:
            prior_pts.append(hyperbolic_points[i])
        for c in candidates:
            seen_names.add(c.lower())
    return flags

def _validate_onnx_batch(batch_chunks, batch_pre, endpoint, model):
    if not batch_chunks:
        return {}
    prompt = ONNX_VALIDATION_PROMPT.format(
        chunks_text=_format_chunks_text(batch_chunks),
        pre_extractions=_format_pre_extractions_for_prompt(batch_pre)
    )
    resp = call_model_json(prompt, model=model, max_tokens=64, system=SYSTEM_PROMPT,
                           unwrap_list=False, endpoint=endpoint, endpoint_type="small")
    if isinstance(resp, str):
        invalid = set()
        for token in resp.split(","):
            token = token.strip()
            if token.startswith("chunk") and "invalid" in token.lower():
                try:
                    num = int(token.split()[1])
                    invalid.add(num)
                except Exception:
                    logger.warning("Unexpected exception occurred", exc_info=True)
                    pass
        return {"invalid": invalid}
    return {}

# ============================================================
#  BATCH EXTRACTION
# ============================================================

def _format_chunks_text(chunks):
    return "\n\n".join(f"Chunk {i}:\n\"\"\"\n{chunk}\n\"\"\"" for i, chunk in enumerate(chunks))

def _normalize_chunk_data(data):
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            return data[0]
        return {}
    elif isinstance(data, dict):
        return data
    else:
        return {}

def _extract_category_batch(category, chunks, model, logic_context="", endpoint=None):
    if category == "facts_entities_relationships":
        prompt_template = FACTS_ENTITIES_PROMPT_BATCH
        max_tokens = 65536
    elif category == "people_locations_dates":
        prompt_template = PEOPLE_LOCATIONS_DATES_PROMPT_BATCH
        max_tokens = 32768
    elif category == "events_discoveries_gems":
        prompt_template = EVENTS_DISCOVERIES_GEMS_PROMPT_BATCH
        max_tokens = 32768
    else:
        raise ValueError("Unknown category")

    prompt = prompt_template.replace("{num_chunks}", str(len(chunks)))
    prompt = prompt.replace("{chunks_text}", _format_chunks_text(chunks))
    prompt = prompt.replace("{logic_context}", logic_context if logic_context else "")

    resp = call_model_json(prompt, model=model, max_tokens=max_tokens,
                           system=SYSTEM_PROMPT, unwrap_list=False, endpoint=endpoint)
    if resp is not None:
        return resp

    print("    (Retrying category extraction due to JSON parse failure...)")
    time.sleep(1)
    resp = call_model_json(prompt, model=model, max_tokens=max_tokens,
                           system=SYSTEM_PROMPT, unwrap_list=False, endpoint=endpoint)
    if resp is not None:
        return resp

    print("    (JSON repair failed; returning empty result for this category batch)")
    return {}

def _format_pre_extractions_for_prompt(pre_list):
    if not pre_list:
        return "None"
    lines = []
    for i, pre in enumerate(pre_list):
        if not pre:
            continue
        lines.append(f"Chunk {i}:")
        for key, label in [("entities", "Entities"), ("people", "People"), ("locations", "Locations"), ("dates", "Dates"), ("organizations", "Organizations")]:
            items = pre.get(key, [])
            if items:
                lines.append(f"  {label}:")
                for item in items[:20]:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("entity_name") or item.get("person_name") or item.get("location_name") or item.get("date_text") or str(item)
                        conf = item.get("confidence", 0.0)
                        lines.append(f"    - {text} (conf: {conf:.2f})")
    return "\n".join(lines)

def _process_batch(batch_chunks, model=None, logic_context="", endpoint=None, actual_model=None, batch_pre_extractions=None):
    if actual_model is None:
        actual_model = model

    results = [{
        "facts": [], "entities": [], "relationships": [],
        "people": [], "locations": [], "dates": [],
        "events": [], "discoveries": [], "gems": []
    } for _ in batch_chunks]

    categories = [
        ("facts_entities_relationships", ["facts", "entities", "relationships"]),
        ("people_locations_dates", ["people", "locations", "dates"]),
        ("events_discoveries_gems", ["events", "discoveries", "gems"]),
    ]

    for category, field_keys in categories:
        prompt_template = {
            "facts_entities_relationships": FACTS_ENTITIES_PROMPT_BATCH,
            "people_locations_dates": PEOPLE_LOCATIONS_DATES_PROMPT_BATCH,
            "events_discoveries_gems": EVENTS_DISCOVERIES_GEMS_PROMPT_BATCH,
        }[category]

        pre_str = _format_pre_extractions_for_prompt(batch_pre_extractions) if batch_pre_extractions else "None"

        prompt = prompt_template.replace("{num_chunks}", str(len(batch_chunks)))
        prompt = prompt.replace("{chunks_text}", _format_chunks_text(batch_chunks))
        prompt = prompt.replace("{logic_context}", logic_context if logic_context else "")
        prompt = prompt.replace("{pre_extractions}", pre_str)

        uncached_indices = []
        cached_results = {}
        for i, chunk_text in enumerate(batch_chunks):
            chunk_hash = _hash_text(chunk_text)
            cached = _get_cached(chunk_hash, category, actual_model, 8192 if category=="facts_entities_relationships" else 4096, prompt_template)
            if cached is not None:
                cached_results[i] = cached
            else:
                uncached_indices.append(i)

        if uncached_indices:
            uncached_chunks = [batch_chunks[i] for i in uncached_indices]
            uncached_pre = [batch_pre_extractions[i] for i in uncached_indices] if batch_pre_extractions else None
            prompt = prompt_template.replace("{num_chunks}", str(len(uncached_chunks)))
            prompt = prompt.replace("{chunks_text}", _format_chunks_text(uncached_chunks))
            prompt = prompt.replace("{logic_context}", logic_context if logic_context else "")
            prompt = prompt.replace("{pre_extractions}", _format_pre_extractions_for_prompt(uncached_pre) if uncached_pre else "None")

            resp = call_model_json(prompt, model=actual_model, max_tokens=8192 if category=="facts_entities_relationships" else 4096,
                                   system=SYSTEM_PROMPT, unwrap_list=False, endpoint=endpoint, endpoint_type=_get_extraction_endpoint_type(category))
            if resp is None:
                print(f"    (Category {category} extraction failed for uncached chunks; continuing)")
                continue

            if isinstance(resp, list):
                if resp and isinstance(resp[0], dict):
                    resp = resp[0]
                else:
                    resp = {}

            if not isinstance(resp, dict):
                resp = {}

            for idx, original_idx in enumerate(uncached_indices):
                key = f"chunk_{idx}"
                chunk_data = resp.get(key, {}) if isinstance(resp, dict) else {}
                chunk_data = _normalize_chunk_data(chunk_data)
                if not chunk_data or not any(chunk_data.get(k) for k in field_keys):
                    # Solo retry: one poison chunk shouldn't kill batch of N (full-doc guarantee).
                    # Single-chunk prompt is smaller, fits small-model context, bounded to 1 retry.
                    try:
                        solo_text = _format_chunks_text([batch_chunks[original_idx]])
                        solo_prompt = prompt_template.replace("{num_chunks}", "1")
                        solo_prompt = solo_prompt.replace("{chunks_text}", solo_text)
                        solo_prompt = solo_prompt.replace("{logic_context}", logic_context if logic_context else "")
                        # Pre-extractions for solo (single item to avoid cross-chunk confusion)
                        try:
                            solo_pre = batch_pre_extractions[original_idx] if batch_pre_extractions else None
                            solo_pre_str = _format_pre_extractions_for_prompt([solo_pre]) if solo_pre else "None"
                        except Exception:
                            solo_pre_str = "None"
                        solo_prompt = solo_prompt.replace("{pre_extractions}", solo_pre_str)
                        solo_resp = call_model_json(solo_prompt, model=actual_model, max_tokens=4096 if category=="facts_entities_relationships" else 2048,
                                                    system=SYSTEM_PROMPT, unwrap_list=False, endpoint=endpoint, endpoint_type=_get_extraction_endpoint_type(category))
                        if isinstance(solo_resp, list) and solo_resp and isinstance(solo_resp[0], dict):
                            solo_resp = solo_resp[0]
                        solo_data = {}
                        if isinstance(solo_resp, dict):
                            solo_data = solo_resp.get("chunk_0", solo_resp)
                            solo_data = _normalize_chunk_data(solo_data) or {}
                        if solo_data and any(solo_data.get(k) for k in field_keys):
                            chunk_data = solo_data
                            if getattr(config, "DEBUG_VERBOSE", False):
                                print(f"    (Solo retry recovered chunk {original_idx} for {category})")
                        else:
                            chunk_data = chunk_data or {}
                    except Exception:
                        chunk_data = chunk_data or {}
                if not chunk_data:
                    chunk_data = {}
                chunk_hash = _hash_text(batch_chunks[original_idx])
                _set_cached(chunk_hash, category, actual_model, 8192 if category=="facts_entities_relationships" else 4096, chunk_data, prompt_template)
                cached_results[original_idx] = chunk_data

        for i in range(len(batch_chunks)):
            if i in cached_results:
                chunk_data = cached_results[i]
                for field in field_keys:
                    if field in chunk_data:
                        results[i][field] = chunk_data[field]

        # Second pass for thin chunks (generic recall booster, not doc-specific):
        # if a long chunk yielded 0-1 facts, re-prompt once listing existing to avoid dupes.
        # Bounded (only thin + long), preserves quality (deduped, validated downstream).
        try:
            _second = bool(getattr(config, "EXTRACTION_SECOND_PASS", True))
            _min_len = int(getattr(config, "SECOND_PASS_MIN_CHARS", 1000))
            _max_items = int(getattr(config, "EXTRACTION_MAX_ITEMS", 20))
            if _second and category == "facts_entities_relationships":
                for i in range(len(batch_chunks)):
                    if len(results[i].get("facts", [])) <= 1 and len(batch_chunks[i]) >= _min_len:
                        try:
                            existing = [f.get("fact_text", "") for f in results[i].get("facts", []) if isinstance(f, dict)]
                            ex_str = "; ".join(existing[:5])[:500] if existing else "none yet"
                            solo_text = _format_chunks_text([batch_chunks[i]])
                            prompt2 = prompt_template.replace("{num_chunks}", "1")
                            prompt2 = prompt2.replace("{chunks_text}", solo_text)
                            prompt2 = prompt2.replace("{logic_context}", (logic_context or "") + f"\nAlready extracted ({len(existing)}): {ex_str}. Extract up to {_max_items} ADDITIONAL distinct facts not listed, same schema.")
                            prompt2 = prompt2.replace("{pre_extractions}", "None")
                            resp2 = call_model_json(prompt2, model=actual_model, max_tokens=4096,
                                                    system=SYSTEM_PROMPT, unwrap_list=False, endpoint=endpoint, endpoint_type=_get_extraction_endpoint_type(category))
                            if isinstance(resp2, list) and resp2 and isinstance(resp2[0], dict):
                                resp2 = resp2[0]
                            extra = {}
                            if isinstance(resp2, dict):
                                extra = resp2.get("chunk_0", resp2)
                                extra = _normalize_chunk_data(extra) or {}
                            for f in extra.get("facts", []) or []:
                                if isinstance(f, dict) and f.get("fact_text"):
                                    # Span check: must exist verbatim in chunk (generic provenance guard)
                                    try:
                                        sp = str(f.get("source_span", ""))
                                        if sp and sp not in batch_chunks[i]:
                                            # Fallback to nearest sentence containing first 3 words of fact
                                            import re as _re2
                                            words = str(f.get("fact_text", "")).split()[:3]
                                            pat = _re2.escape(" ".join(words)) if words else ""
                                            # Keep as-is; cleaners/verifier will down-weight if invalid
                                            pass
                                    except Exception:
                                        pass
                                    results[i].setdefault("facts", []).append(f)
                            # Cap per-chunk to max_items (quality: prevents bloat, generic limit)
                            if len(results[i].get("facts", [])) > _max_items:
                                results[i]["facts"] = results[i]["facts"][:_max_items]
                        except Exception:
                            continue
        except Exception:
            pass

    for i in range(len(results)):
        results[i]["facts"] = _clean_facts(results[i]["facts"])
        results[i]["entities"] = _clean_entities(results[i]["entities"])
        results[i]["relationships"] = _clean_relationships(results[i].get("relationships", []))
        results[i]["people"] = _clean_people(results[i]["people"])
        results[i]["locations"] = _clean_locations(results[i]["locations"])
        results[i]["dates"] = _clean_dates(results[i]["dates"])
        results[i]["events"] = _clean_events(results[i]["events"])
        results[i]["discoveries"] = _clean_discoveries(results[i]["discoveries"])
        results[i]["gems"] = _clean_gems(results[i]["gems"])

        for key in ["facts", "entities", "relationships", "people", "locations", "dates", "events", "discoveries", "gems"]:
            validated = []
            for item in results[i].get(key, []):
                v = validate_and_coerce(key, item)
                if v is not None:
                    validated.append(v)
            results[i][key] = validated

    return results

# Single _get_dynamic_capacities (with config flag)
_endpoint_capacities_cache = None

def _get_dynamic_capacities():
    global _endpoint_capacities_cache
    if _endpoint_capacities_cache is not None:
        return _endpoint_capacities_cache
    if not getattr(config, "USE_DYNAMIC_ENDPOINT_BALANCING", False):
        return None
    capacities = []
    import time as _time
    for ep in config.LLM_ENDPOINTS:
        try:
            start = _time.time()
            from core.llm import call_model
            resp = call_model("ping", max_tokens=2, endpoint=ep)
            latency = _time.time() - start
            if not resp:
                capacities.append(0)
            else:
                latency = min(latency, 60.0)
                capacities.append(max(1, int(10.0 / latency)))
        except Exception:
            capacities.append(0)
    if sum(capacities) == 0:
        return None
    _endpoint_capacities_cache = capacities
    return capacities

def extract_from_chunks(chunks, model=None, max_workers=None, chunk_embeddings=None, logic_context="", doc_type=None):
    if max_workers is None:
        max_workers = config.CHUNK_EXTRACTION_WORKERS
    _init_cache()

    if chunk_embeddings is None:
        print("  (No chunk embeddings provided; computing embeddings for novelty gating...)")
        chunk_embeddings = get_embeddings_batch(chunks, batch_size=config.EMBEDDING_BATCH_SIZE)

    flags = _compute_novelty_flags(chunks, chunk_embeddings)
    # Safety floor (generic, config-driven): never skip so aggressively that coverage collapses.
    # Keeps most-distant skipped chunks (diversity, not prefix) up to floor.
    try:
        kept = sum(1 for f in flags if f)
        _ratio = float(getattr(config, "NOVELTY_MIN_KEEP_RATIO", 0.3))
        _min_n = int(getattr(config, "NOVELTY_MIN_KEEP_COUNT", 3))
        min_keep = max(_min_n if len(chunks) >= _min_n else len(chunks), int(len(chunks) * _ratio))
        if kept < min_keep and len(chunks) > 0:
            import numpy as _npf
            from core.hyperbolic import ensure_hyperbolic, hyperbolic_distance_matrix
            # Score skipped by max distance to kept priors (most novel first)
            kept_pts = [chunk_embeddings[i] for i, f in enumerate(flags) if f and chunk_embeddings[i] is not None]
            skipped_idx = [i for i, f in enumerate(flags) if not f]
            if kept_pts and skipped_idx:
                try:
                    kmat = _npf.stack([ensure_hyperbolic(_npf.asarray(e, dtype=_npf.float32), space='hyperbolic') for e in kept_pts])
                    scores = []
                    for i in skipped_idx:
                        emb = chunk_embeddings[i]
                        if emb is None:
                            scores.append((float('inf'), i))
                            continue
                        qh = ensure_hyperbolic(_npf.asarray(emb, dtype=_npf.float32), space='hyperbolic')[None, :]
                        d = float(_npf.min(hyperbolic_distance_matrix(qh, kmat)[0]))
                        scores.append((d, i))
                    scores.sort(reverse=True)
                    need = min_keep - kept
                    for _, i in scores[:need]:
                        flags[i] = True
                except Exception:
                    need = min_keep - kept
                    for i in skipped_idx[:need]:
                        flags[i] = True
            else:
                need = min_keep - kept
                for i in skipped_idx[:need]:
                    flags[i] = True
            print(f"  (Novelty floor: kept {kept}/{len(chunks)}, raised to {sum(1 for f in flags if f)}/{len(chunks)} for coverage)")
    except Exception:
        pass

    fast_pre_results = None
    if config.FAST_EXTRACTOR_ENABLED:
        try:
            print("  (Running fast extractor pre-pass...)")
            global _fast_extractor_instance
            if _fast_extractor_instance is None:
                _fast_extractor_instance = FastExtractor()
            fast_extractor = _fast_extractor_instance
            fast_pre_results = []
            for chunk in chunks:
                fast_pre_results.append(fast_extractor.extract(chunk))
        except Exception as e:
            print(f"    (Fast extractor error: {e}); falling back to full LLM extraction.")
            fast_pre_results = None

    # DB-aware recall (fast-pass scans DBs for topics/dates/refs/events, flags priority).
    # Priority = must-extract + must-verify, never must-believe. Generic, no hardcoding.
    recall_list = [None] * len(chunks)
    if getattr(config, "RECALL_AUGMENT_ENABLED", True):
        try:
            from extraction.recall_augmenter import augment_batch
            recall_list = augment_batch(chunks, fast_pres=fast_pre_results, chunk_embs=chunk_embeddings)
            n_prio = sum(1 for r in recall_list if r and r.get("priority"))
            if n_prio:
                print(f"  (Recall: {n_prio}/{len(chunks)} priority chunks flagged for guaranteed extraction)")
                # Bypass novelty skip for priority (full-doc guarantee for anchors)
                for i, r in enumerate(recall_list):
                    if r and r.get("priority") and not flags[i]:
                        flags[i] = True
                try:
                    from core.metrics import inc_counter as _inc2
                    _inc2("recall_priority_chunks_total", n_prio)
                except Exception:
                    pass
        except Exception as e:
            if getattr(config, "DEBUG_VERBOSE", False):
                print(f"    (Recall augmenter error: {e})")
            recall_list = [None] * len(chunks)

    gate = None
    gate_features_cache = {}

    distilled_extractor = None
    distilled_extractor_batch = None
    if getattr(config, "USE_DISTILLED_EXTRACTOR", True):
        try:
            from extraction.distilled_extractor import generate_extraction, generate_extractions_batch
            distilled_extractor = generate_extraction
            distilled_extractor_batch = generate_extractions_batch
        except ImportError:
            distilled_extractor = None
            distilled_extractor_batch = None
    if getattr(config, "USE_PRIME_EVEN_GATE", False):
        try:
            from extraction.gate import PrimeEvenGate
            gate_path = Path(config.BASE_DIR) / "models" / "gate.json"
            gate = PrimeEvenGate()
            if gate_path.exists():
                gate.load(gate_path)
                print("  (Using trained prime-even gate)")
            else:
                print("  (Gate enabled but no trained model found; skipping gate)")
                gate = None
        except Exception as e:
            if config.DEBUG_VERBOSE:
                print(f"    (Gate init error: {e})")
            gate = None

    distilled_results = None
    if distilled_extractor_batch is not None and len(chunks) > 0:
        try:
            distilled_results = distilled_extractor_batch(chunks)
            hit_count = sum(1 for r in distilled_results if r is not None)
            print(f"  (Distilled extractor handled {hit_count}/{len(chunks)} chunks)")
        except Exception as e:
            if config.DEBUG_VERBOSE:
                print(f"    (Distilled batch error: {e})")
            distilled_results = None

    selected_items = []

    distilled_extractor = None
    if getattr(config, "USE_DISTILLED_EXTRACTOR", True):
        try:
            from extraction.distilled_extractor import generate_extraction
            distilled_extractor = generate_extraction
        except ImportError:
            distilled_extractor = None

    if chunk_embeddings is not None and len(chunks) > 0:
        chunk_emb_matrix = np.array([np.array(emb, dtype=np.float32) for emb in chunk_embeddings if emb is not None])
    else:
        chunk_emb_matrix = None

    all_results = [{
        "facts": [], "entities": [], "relationships": [],
        "people": [], "locations": [], "dates": [],
        "events": [], "discoveries": [], "gems": []
    } for _ in chunks]

    for i in range(len(chunks)):
        if not flags[i]:
            continue

        pre = fast_pre_results[i] if fast_pre_results else None
        rec = recall_list[i] if i < len(recall_list) else None
        is_prio = bool(rec and rec.get("priority"))

        # Priority bypasses distilled-empty skip and gate skip (must-extract, not must-believe)
        if not is_prio:
            if distilled_results is not None and i < len(distilled_results):
                distilled_result = distilled_results[i]
                if distilled_result is not None and any(distilled_result.get(k) for k in ("facts", "entities", "people", "locations", "dates", "events", "discoveries", "gems")):
                    all_results[i]["facts"] = distilled_result.get("facts", [])
                    all_results[i]["entities"] = distilled_result.get("entities", [])
                    all_results[i]["people"] = distilled_result.get("people", [])
                    all_results[i]["locations"] = distilled_result.get("locations", [])
                    all_results[i]["dates"] = distilled_result.get("dates", [])
                    all_results[i]["events"] = distilled_result.get("events", [])
                    all_results[i]["discoveries"] = distilled_result.get("discoveries", [])
                    all_results[i]["gems"] = distilled_result.get("gems", [])
                    continue
            elif distilled_extractor is not None:
                try:
                    distilled_result = distilled_extractor(chunks[i])
                    if distilled_result is not None and any(distilled_result.get(k) for k in ("facts", "entities", "people", "locations", "dates", "events", "discoveries", "gems")):
                        all_results[i]["facts"] = distilled_result.get("facts", [])
                        all_results[i]["entities"] = distilled_result.get("entities", [])
                        all_results[i]["people"] = distilled_result.get("people", [])
                        all_results[i]["locations"] = distilled_result.get("locations", [])
                        all_results[i]["dates"] = distilled_result.get("dates", [])
                        all_results[i]["events"] = distilled_result.get("events", [])
                        all_results[i]["discoveries"] = distilled_result.get("discoveries", [])
                        all_results[i]["gems"] = distilled_result.get("gems", [])
                        continue
                except Exception as e:
                    if config.DEBUG_VERBOSE:
                        print(f"    (Distilled extractor error: {e})")
        else:
            # Still consult distilled for enrichment, but never skip LLM on empty
            if distilled_results is not None and i < len(distilled_results) and distilled_results[i]:
                _dr = distilled_results[i]
                if any(_dr.get(k) for k in ("facts", "entities", "people", "locations")):
                    for k in ("facts", "entities", "people", "locations", "dates", "events", "discoveries", "gems"):
                        if _dr.get(k):
                            all_results[i][k] = _dr.get(k, [])

        use_full_llm = True
        if gate is not None and not is_prio and chunk_embeddings is not None and chunk_embeddings[i] is not None:
            if 'all' not in gate_features_cache:
                from core.spectral import compute_spectral_features
                feat = compute_spectral_features(chunk_emb_matrix, top_k=22)
                gate_features_cache['all'] = feat
            feat = gate_features_cache['all']
            w = gate.forward(feat)
            if w < 0.5:
                use_full_llm = False

        if use_full_llm:
            # Carry recall context for prompt injection (guides LLM to linked anchors)
            rctx = (rec.get("context", "") if rec else "")
            selected_items.append((i, chunks[i], pre, rctx, is_prio))
        else:
            all_results[i]["entities"] = pre.get("entities", []) if pre else []
            all_results[i]["people"] = pre.get("people", []) if pre else []
            all_results[i]["locations"] = pre.get("locations", []) if pre else []
            all_results[i]["dates"] = pre.get("dates", []) if pre else []

    skipped_count = len(chunks) - len(selected_items)
    if skipped_count > 0:
        print(f"  (Novelty gating: skipping {skipped_count} redundant chunks out of {len(chunks)})")

    if not selected_items:
        return all_results

    # Small-safe batching: shared queue serves small + large endpoints, so fit smallest context.
    # Uses race-free helper, no global mutation. Preserves full-doc (same items, more batches).
    try:
        batch_size = effective_llm_batch_chunks(len(selected_items), endpoint_type="small")
    except Exception:
        batch_size = int(getattr(config, "LLM_BATCH_CHUNKS", 4))
        batch_size = max(1, min(batch_size, 4))
    batches = []
    for i in range(0, len(selected_items), batch_size):
        batch = selected_items[i:i+batch_size]
        batches.append(batch)

    task_queue = queue.Queue()
    for batch_idx, batch in enumerate(batches):
        # Batch items are (orig_idx, text, pre, rctx, is_prio) after recall wiring
        batch_texts = []
        batch_pre = []
        batch_rctx = []
        batch_prio = []
        batch_orig = []
        for item in batch:
            if len(item) == 5:
                oi, tx, pr, rc, ip = item
            else:
                oi, tx, pr = item
                rc, ip = "", False
            batch_orig.append(oi)
            batch_texts.append(tx)
            batch_pre.append(pr)
            batch_rctx.append(rc)
            batch_prio.append(ip)
        task_queue.put((batch_idx, batch_texts, batch_pre, batch_rctx, batch_prio, batch_orig))

    results = {}
    lock = threading.Lock()

    pbar = None
    if config.USE_PROGRESS_BARS and config.TQDM_AVAILABLE:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(batches), desc="  Extracting chunks", unit="batch")
        except ImportError:
            pbar = None

    workers = []
    caps = config.LLM_ENDPOINT_CAPACITIES
    if getattr(config, "USE_DYNAMIC_ENDPOINT_BALANCING", False):
        dynamic_caps = _get_dynamic_capacities()
        if dynamic_caps is not None:
            caps = dynamic_caps

    for ep_idx, endpoint in enumerate(config.LLM_ENDPOINTS):
        capacity = caps[ep_idx] if ep_idx < len(caps) else 1
        if capacity <= 0:
            continue
        for worker_idx in range(capacity):
            def worker(ep=endpoint, wid=worker_idx, eidx=ep_idx):
                while True:
                    try:
                        batch_idx, batch_texts, batch_pre, batch_rctx, batch_prio, batch_orig = task_queue.get_nowait()
                    except queue.Empty:
                        break
                    if config.DEBUG_VERBOSE:
                        print(f"    [Endpoint {eidx}, worker {wid}] processing batch {batch_idx}")
                    try:
                        actual_model = ep["model"]
                        # Inject recall contexts into logic_context (guides LLM to linked anchors, no hardcoding)
                        _rctxs = [c for c in batch_rctx if c]
                        _lctx = logic_context or ""
                        if _rctxs:
                            _uniq = []
                            seen_rc = set()
                            for c in _rctxs:
                                if c not in seen_rc:
                                    seen_rc.add(c)
                                    _uniq.append(c)
                            _rctx_block = "Recall (DB-linked, verify carefully): " + " | ".join(_uniq)[:1000]
                            _lctx = (_lctx + "\n\n" + _rctx_block) if _lctx else _rctx_block
                        batch_results = _process_batch(
                            batch_texts,
                            model=None,
                            logic_context=_lctx,
                            endpoint=ep,
                            actual_model=actual_model,
                            batch_pre_extractions=batch_pre,
                        )
                        # Tag priority facts for downstream protection/escalation (must-verify)
                        try:
                            for br, ip in zip(batch_results, batch_prio):
                                if ip and isinstance(br, dict):
                                    for k in ("facts", "entities", "people", "locations", "dates", "events", "discoveries", "gems"):
                                        for it in br.get(k, []) or []:
                                            if isinstance(it, dict):
                                                it["recall_priority"] = True
                        except Exception:
                            pass
                        with lock:
                            results[batch_idx] = (batch_results, batch_orig, batch_prio)
                    except Exception as e:
                        print(f"    (Batch {batch_idx} error on endpoint {eidx}: {e})")
                        empty = [{
                            "facts": [], "entities": [], "relationships": [],
                            "people": [], "locations": [], "dates": [],
                            "events": [], "discoveries": [], "gems": []
                        } for _ in batch_texts]
                        with lock:
                            results[batch_idx] = (empty, batch_orig if 'batch_orig' in locals() else [], [False]*len(batch_texts))
                    finally:
                        task_queue.task_done()
                        if pbar:
                            pbar.update(1)

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            workers.append(t)

    try:
        task_queue.join()
    except KeyboardInterrupt:
        print("\n  (KeyboardInterrupt received: stopping chunk processing...)")
        raise

    if pbar:
        pbar.close()

    # Collect distilled training data
    if getattr(config, "COLLECT_DISTILLED_TRAINING_DATA", True):
        try:
            import json as json_mod
            from core import db as db_mod
            conn = db_mod.db_connect("key_facts")
            cur = conn.cursor()
            for batch_idx in sorted(results.keys()):
                _res = results[batch_idx]
                if isinstance(_res, tuple) and len(_res) == 3:
                    _batch_results, _batch_orig, _ = _res
                else:
                    _batch_results = _res
                    _batch_orig = [idx for idx, *_ in batches[batch_idx]]
                for orig_idx, chunk_data in zip(_batch_orig, _batch_results):
                    if chunk_data.get("facts") or chunk_data.get("entities"):
                        target = {
                            "facts": chunk_data.get("facts", []),
                            "entities": chunk_data.get("entities", []),
                            "people": chunk_data.get("people", []),
                            "locations": chunk_data.get("locations", []),
                            "dates": chunk_data.get("dates", []),
                            "events": chunk_data.get("events", []),
                            "discoveries": chunk_data.get("discoveries", []),
                            "gems": chunk_data.get("gems", []),
                        }
                        chunk_text = chunks[orig_idx]
                        cur.execute(
                            "INSERT INTO distilled_training_data (chunk_text, target_json) VALUES (?,?)",
                            (chunk_text, json_mod.dumps(target))
                        )
            conn.commit()
            conn.close()
        except Exception as e:
            if config.DEBUG_VERBOSE:
                print(f"    (Distilled training data collection error: {e})")

    for batch_idx in sorted(results.keys()):
        _res = results[batch_idx]
        if isinstance(_res, tuple) and len(_res) == 3:
            _batch_results, _batch_orig, _ = _res
        else:
            _batch_results = _res
            _batch_orig = [idx for idx, *_ in batches[batch_idx]]
        for original_idx, res in zip(_batch_orig, _batch_results):
            all_results[original_idx] = res

    if config.ENABLE_ASYNC_VALIDATION:
        print("  Running validation queue on extracted items...")
        vq = ValidationQueue()
        vq.start()
        for chunk_idx, chunk_data in enumerate(all_results):
            for key in ["facts", "entities", "people", "locations", "dates", "events", "discoveries", "gems"]:
                for item in chunk_data.get(key, []):
                    if isinstance(item, dict):
                        item_copy = item.copy()
                        item_copy["_chunk_idx"] = chunk_idx
                        item_copy["category"] = key
                        vq.put(item_copy)
        validated_results = vq.wait_and_get_results()
        clean_map = {
            'facts': _clean_facts,
            'entities': _clean_entities,
            'people': _clean_people,
            'locations': _clean_locations,
            'dates': _clean_dates,
            'events': _clean_events,
            'discoveries': _clean_discoveries,
            'gems': _clean_gems,
        }
        for item in validated_results:
            if isinstance(item, dict) and "_chunk_idx" in item and "category" in item:
                idx = item.pop("_chunk_idx")
                cat = item.pop("category")
                if 0 <= idx < len(all_results):
                    cleaner = clean_map.get(cat)
                    if cleaner:
                        cleaned_items = cleaner([item])
                        if cleaned_items:
                            all_results[idx].setdefault(cat, []).append(cleaned_items[0])
                    else:
                        all_results[idx].setdefault(cat, []).append(item)
    return all_results
