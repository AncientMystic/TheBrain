import time
import json
import hashlib
import re
import sqlite3
import queue
import threading

import numpy as np

import config
from core import db
_init_cache()
from core.llm import call_model_json
from core.embeddings import get_embeddings_batch
from extraction.validation_queue import ValidationQueue
from extraction.rule_annotator import pre_annotate
from fast_extractor.hybrid_extractor import FastExtractor
from extraction.cleaners import *



# ============================================================
#  FEW-SHOT EXAMPLES
# ============================================================

def _get_extraction_endpoint_type(category):
    """Determine endpoint_type for extraction based on available models.
    If small model configured, use it for initial extraction; else None (default)."""
    if config.SMALL_MODEL_ENDPOINT:
        return "small"
    return None

def _get_verification_endpoint_type():
    """If large model configured, use it for verification."""
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
#  PROMPT TEMPLATES (plain strings with placeholders)
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
        except:
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

def _compute_novelty_flags(chunks, chunk_embeddings):
    flags = []
    processed_embeddings = []
    seen_names = set()

    for i, chunk in enumerate(chunks):
        if chunk_embeddings is None or chunk_embeddings[i] is None:
            flags.append(True)
            if chunk_embeddings and chunk_embeddings[i] is not None:
                processed_embeddings.append(chunk_embeddings[i])
            annotations = pre_annotate(chunk)
            for c in _extract_candidate_texts(annotations):
                seen_names.add(c.lower())
            continue

        annotations = pre_annotate(chunk)
        candidates = _extract_candidate_texts(annotations)
        new_candidates = [c for c in candidates if c.lower() not in seen_names]

        max_sim = 0.0
        if processed_embeddings:
            cur_emb = np.array(chunk_embeddings[i], dtype=np.float32)
            cur_norm = np.linalg.norm(cur_emb)
            if cur_norm > 0:
                for prev_emb in processed_embeddings:
                    prev_emb = np.array(prev_emb, dtype=np.float32)
                    prev_norm = np.linalg.norm(prev_emb)
                    if prev_norm > 0:
                        sim = float(np.dot(cur_emb, prev_emb) / (cur_norm * prev_norm + 1e-8))
                        if sim > max_sim:
                            max_sim = sim

        if config.NOVELTY_ENABLED and i > 0:
            if max_sim >= config.NOVELTY_SIM_THRESHOLD and not new_candidates:
                flags.append(False)
                continue
            else:
                flags.append(True)
        else:
            flags.append(True)

        if chunk_embeddings[i] is not None:
            processed_embeddings.append(chunk_embeddings[i])
        for c in candidates:
            seen_names.add(c.lower())

    return flags


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
        max_tokens = 8192
    elif category == "people_locations_dates":
        prompt_template = PEOPLE_LOCATIONS_DATES_PROMPT_BATCH
        max_tokens = 4096
    elif category == "events_discoveries_gems":
        prompt_template = EVENTS_DISCOVERIES_GEMS_PROMPT_BATCH
        max_tokens = 4096
    else:
        raise ValueError("Unknown category")

    prompt = prompt_template.replace("{num_chunks}", str(len(chunks)))
    prompt = prompt.replace("{chunks_text}", _format_chunks_text(chunks))
    prompt = prompt.replace("{logic_context}", logic_context if logic_context else "")

    # First attempt
    resp = call_model_json(prompt, model=model, max_tokens=max_tokens,
                           system=SYSTEM_PROMPT, unwrap_list=False, endpoint=endpoint)
    if resp is not None:
        return resp

    # Retry once
    print("    (Retrying category extraction due to JSON parse failure...)")
    time.sleep(1)
    resp = call_model_json(prompt, model=model, max_tokens=max_tokens,
                           system=SYSTEM_PROMPT, unwrap_list=False, endpoint=endpoint)
    if resp is not None:
        return resp

    # Final fallback: return empty dict so the chunk isn't lost entirely
    print("    (JSON repair failed; returning empty result for this category batch)")
    return {}



def _format_pre_extractions_for_prompt(pre_list):
    """Format pre-extracted items for prompt inclusion."""
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
    """Process a batch using the three original prompts, enhanced with pre-extractions."""
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

        # Build pre-extractions string if available
        pre_str = _format_pre_extractions_for_prompt(batch_pre_extractions) if batch_pre_extractions else "None"

        # Cache logic can be skipped for simplicity; we'll always call LLM (or use cache in _extract_category_batch)
        # We'll use the existing _extract_category_batch but pass pre_str via replacing placeholder.
        # Since _extract_category_batch currently replaces {chunks_text} and {logic_context}, we need to modify it.
        # For now, we'll handle by temporarily modifying the prompt inside this loop.
        prompt = prompt_template.replace("{num_chunks}", str(len(batch_chunks)))
        prompt = prompt.replace("{chunks_text}", _format_chunks_text(batch_chunks))
        prompt = prompt.replace("{logic_context}", logic_context if logic_context else "")
        prompt = prompt.replace("{pre_extractions}", pre_str)

        # Cache-aware per-chunk processing
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
            # Build prompt for uncached chunks only
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

            for idx, original_idx in enumerate(uncached_indices):
                key = f"chunk_{idx}"
                chunk_data = resp.get(key, {}) if isinstance(resp, dict) else {}
                chunk_data = _normalize_chunk_data(chunk_data)
                # Cache this chunk's result
                chunk_hash = _hash_text(batch_chunks[original_idx])
                _set_cached(chunk_hash, category, actual_model, 8192 if category=="facts_entities_relationships" else 4096, chunk_data, prompt_template)
                cached_results[original_idx] = chunk_data

        # Merge cached and newly fetched results
        for i in range(len(batch_chunks)):
            if i in cached_results:
                chunk_data = cached_results[i]
                for field in field_keys:
                    if field in chunk_data:
                        results[i][field] = chunk_data[field]

    # Apply cleaners
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

    return results

def extract_from_chunks(chunks, model=None, max_workers=None, chunk_embeddings=None, logic_context="", doc_type=None):
    """Extract from chunks using fast pre-extraction + focused LLM prompts."""
    if max_workers is None:
        max_workers = config.CHUNK_EXTRACTION_WORKERS
    # _init_cache is now idempotent and called once at module import
    pass

    if chunk_embeddings is None:
        print("  (No chunk embeddings provided; computing embeddings for novelty gating...)")
        chunk_embeddings = get_embeddings_batch(chunks, batch_size=config.EMBEDDING_BATCH_SIZE)

    flags = _compute_novelty_flags(chunks, chunk_embeddings)

    # Fast pre-extraction (if enabled)
    fast_pre_results = None
    if config.FAST_EXTRACTOR_ENABLED:
        try:
            print("  (Running fast extractor pre-pass...)")
            fast_extractor = FastExtractor()
            fast_pre_results = []
            for chunk in chunks:
                fast_pre_results.append(fast_extractor.extract(chunk))
        except Exception as e:
            print(f"    (Fast extractor error: {e}); falling back to full LLM extraction.")
            fast_pre_results = None

    selected_items = []
    for i in range(len(chunks)):
        if flags[i]:
            pre = fast_pre_results[i] if fast_pre_results else None
            selected_items.append((i, chunks[i], pre))

    skipped_count = len(chunks) - len(selected_items)
    if skipped_count > 0:
        print(f"  (Novelty gating: skipping {skipped_count} redundant chunks out of {len(chunks)})")

    all_results = [{
        "facts": [], "entities": [], "relationships": [],
        "people": [], "locations": [], "dates": [],
        "events": [], "discoveries": [], "gems": []
    } for _ in chunks]

    if not selected_items:
        return all_results

    batch_size = config.LLM_BATCH_CHUNKS
    batches = []
    for i in range(0, len(selected_items), batch_size):
        batch = selected_items[i:i+batch_size]
        batches.append(batch)

    task_queue = queue.Queue()
    for batch_idx, batch in enumerate(batches):
        batch_texts = [text for _, text, _ in batch]
        batch_pre = [pre for _, _, pre in batch]
        task_queue.put((batch_idx, batch_texts, batch_pre))

    results = {}
    lock = threading.Lock()

    # Progress bar for chunk batches
    pbar = None
    if config.USE_PROGRESS_BARS and config.TQDM_AVAILABLE:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(batches), desc="  Extracting chunks", unit="batch")
        except ImportError:
            pbar = None

    workers = []
    for ep_idx, endpoint in enumerate(config.LLM_ENDPOINTS):
        capacity = config.LLM_ENDPOINT_CAPACITIES[ep_idx] if ep_idx < len(config.LLM_ENDPOINT_CAPACITIES) else 1
        for _ in range(capacity):
            def worker(ep=endpoint):
                while True:
                    try:
                        batch_idx, batch_texts, batch_pre = task_queue.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        actual_model = ep["model"]
                        batch_results = _process_batch(
                            batch_texts,
                            model=None,
                            logic_context=logic_context,
                            endpoint=ep,
                            actual_model=actual_model,
                            batch_pre_extractions=batch_pre,
                        )
                        with lock:
                            results[batch_idx] = batch_results
                    except Exception as e:
                        print(f"    (Batch {batch_idx} error: {e})")
                        empty = [{
                            "facts": [], "entities": [], "relationships": [],
                            "people": [], "locations": [], "dates": [],
                            "events": [], "discoveries": [], "gems": []
                        } for _ in batch_texts]
                        with lock:
                            results[batch_idx] = empty
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

    for batch_idx in sorted(results.keys()):
        batch_indices = [idx for idx, _, _ in batches[batch_idx]]
        batch_results = results[batch_idx]
        for original_idx, res in zip(batch_indices, batch_results):
            all_results[original_idx] = res

    if config.ENABLE_ASYNC_VALIDATION:
        print("  Running validation queue on extracted items...")
        vq = ValidationQueue()
        vq.start()
        # Flatten items with metadata
        for chunk_idx, chunk_data in enumerate(all_results):
            for key in ["facts", "entities", "people", "locations", "dates", "events", "discoveries", "gems"]:
                for item in chunk_data.get(key, []):
                    if isinstance(item, dict):
                        item_copy = item.copy()
                        item_copy["_chunk_idx"] = chunk_idx
                        item_copy["_category"] = key
                        vq.put(item_copy)
        validated_results = vq.wait_and_get_results()
        # Reconstruct all_results from validated items
        for item in validated_results:
            if isinstance(item, dict) and "_chunk_idx" in item and "_category" in item:
                idx = item.pop("_chunk_idx")
                cat = item.pop("_category")
                if 0 <= idx < len(all_results):
                    all_results[idx].setdefault(cat, [])
                    all_results[idx][cat].append(item)
    return all_results
