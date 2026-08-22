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
from core.llm import call_model_json
from core.embeddings import get_embeddings_batch
from extraction.rule_annotator import pre_annotate

# ============================================================
#  FEW-SHOT EXAMPLES
# ============================================================
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
- For each excerpt, extract only information explicitly stated in that excerpt.
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

Excerpts:
{chunks_text}

Return only JSON.
"""

PEOPLE_LOCATIONS_DATES_PROMPT_BATCH = """
You are a meticulous knowledge extraction agent.
Read {num_chunks} document excerpts below and extract ALL relevant people, locations, and dates for each excerpt.

STRICT RULES:
- For each excerpt, extract only information explicitly stated in that excerpt.
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
def _truncate(value, max_len):
    if not isinstance(value, str):
        return value
    if len(value) <= max_len:
        return value
    return value[:max_len-3] + "..."

def _safe_str(value, max_len):
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return _truncate(value, max_len)

def _normalize_text_for_compare(text):
    return re.sub(r'\s+', ' ', str(text).lower()).strip()

def _is_verbatim_copy(fact_text, source_span):
    fact_text = fact_text or ""
    source_span = source_span or ""
    if not fact_text or not source_span:
        return False
    ft = _normalize_text_for_compare(fact_text)
    ss = _normalize_text_for_compare(source_span)
    if ft in ss or ss in ft:
        return True
    ft_tokens = set(ft.split())
    ss_tokens = set(ss.split())
    if not ft_tokens or not ss_tokens:
        return False
    overlap = len(ft_tokens & ss_tokens) / max(len(ft_tokens), len(ss_tokens))
    return overlap > 0.9 and abs(len(ft) - len(ss)) < 50


# ============================================================
#  IMPROVED CLEANING HELPERS
# ============================================================
def _shorten_source_span(span, max_words=4):
    if not span:
        return ""
    span = _safe_str(span, 200)
    words = span.split()
    if len(words) <= max_words:
        return span
    return " ".join(words[:max_words]) + " ..."

def _normalize_name_text(name):
    if not name:
        return ""
    name = re.sub(r'[^\w\s]', '', name.lower())
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def _is_redundant_span(span, text):
    if not span or not text:
        return False
    span_norm = _normalize_text_for_compare(span)
    text_norm = _normalize_text_for_compare(text)
    if span_norm == text_norm:
        return True
    span_tokens = set(span_norm.split())
    text_tokens = set(text_norm.split())
    if not span_tokens or not text_tokens:
        return False
    overlap = len(span_tokens & text_tokens) / max(len(span_tokens), len(text_tokens))
    return overlap > 0.8

def _is_atomic(text):
    """Reject if text contains conjunctions joining two claims."""
    if not text:
        return True
    lower = text.lower()
    conjunctions = [" and ", " or ", " also ", " but ", " while ", " whereas "]
    for conj in conjunctions:
        if conj in lower:
            # Simple heuristic: if after conjunction there is another subject/predicate,
            # treat as non-atomic.
            # We just flag any occurrence; user can refine later.
            return False
    return True

def _clean_facts(facts):
    cleaned = []
    for f in facts:
        if not isinstance(f, dict):
            continue
        fact_text = _safe_str(f.get("fact_text"), 200)
        source_span = _safe_str(f.get("source_span"), 200)

        # Atomicity gate
        if not _is_atomic(fact_text):
            continue

        if source_span:
            if len(source_span.split()) > 4 or _is_redundant_span(source_span, fact_text):
                source_span = _shorten_source_span(source_span, max_words=4)
                if _is_redundant_span(source_span, fact_text):
                    source_span = " ".join(fact_text.split()[:3])

        if _is_verbatim_copy(fact_text, source_span):
            continue

        f["fact_text"] = fact_text
        f["canonical_value"] = _safe_str(f.get("canonical_value"), 80)
        f["source_span"] = source_span
        f["fact_type"] = _safe_str(f.get("fact_type"), 80)
        try:
            f["confidence"] = float(f.get("confidence", 0.0))
        except:
            f["confidence"] = 0.0
        if fact_text.strip():
            cleaned.append(f)
    return cleaned

# Similar modifications to _clean_entities, _clean_people, etc.:
# - Use _normalize_name_text for normalized fields.
# - Shorten source_span if >4 words or redundant.
# - Apply _is_atomic to any text field that should be atomic (fact_text, event_name, discovery_name, gem_text).
# For brevity, I'll provide one example and note that you should apply the same pattern to all.

def _clean_entities(entities):
    cleaned = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        entity_name = _safe_str(e.get("entity_name"), 150)
        normalized = _normalize_name_text(entity_name)
        source_span = _safe_str(e.get("source_span"), 200)

        if source_span:
            if len(source_span.split()) > 4 or _is_redundant_span(source_span, entity_name):
                source_span = _shorten_source_span(source_span, max_words=4)
                if _is_redundant_span(source_span, entity_name):
                    source_span = " ".join(entity_name.split()[:3])

        e["entity_name"] = entity_name
        e["normalized_name"] = normalized
        e["source_span"] = source_span
        e["entity_type"] = _safe_str(e.get("entity_type"), 40)
        try:
            e["confidence"] = float(e.get("confidence", 0.0))
        except:
            e["confidence"] = 0.0
        if entity_name.strip():
            cleaned.append(e)
    return cleaned

def _clean_people(people):
    cleaned = []
    for p in people:
        if not isinstance(p, dict):
            continue
        person_name = _safe_str(p.get("person_name"), 150)
        normalized = _normalize_name_text(person_name)
        source_span = _safe_str(p.get("source_span"), 200)

        if source_span:
            if len(source_span.split()) > 4 or _is_redundant_span(source_span, person_name):
                source_span = _shorten_source_span(source_span, max_words=4)
                if _is_redundant_span(source_span, person_name):
                    source_span = " ".join(person_name.split()[:3])

        p["person_name"] = person_name
        p["normalized_name"] = normalized
        p["role"] = _safe_str(p.get("role"), 80)
        p["source_span"] = source_span
        try:
            p["confidence"] = float(p.get("confidence", 0.0))
        except:
            p["confidence"] = 0.0
        if person_name.strip():
            cleaned.append(p)
    return cleaned

def _clean_locations(locations):
    cleaned = []
    for l in locations:
        if not isinstance(l, dict):
            continue
        location_name = _safe_str(l.get("location_name"), 150)
        normalized = _normalize_name_text(location_name)
        source_span = _safe_str(l.get("source_span"), 200)

        if source_span:
            if len(source_span.split()) > 4 or _is_redundant_span(source_span, location_name):
                source_span = _shorten_source_span(source_span, max_words=4)
                if _is_redundant_span(source_span, location_name):
                    source_span = " ".join(location_name.split()[:3])

        l["location_name"] = location_name
        l["normalized_place"] = normalized
        l["location_type"] = _safe_str(l.get("location_type"), 40)
        l["source_span"] = source_span
        try:
            l["confidence"] = float(l.get("confidence", 0.0))
        except:
            l["confidence"] = 0.0
        if location_name.strip():
            cleaned.append(l)
    return cleaned

def _clean_dates(dates):
    cleaned = []
    for d in dates:
        if not isinstance(d, dict):
            continue
        date_text = _safe_str(d.get("date_text"), 100)
        normalized_date = _safe_str(d.get("normalized_date"), 50)
        source_span = _safe_str(d.get("source_span"), 200)

        if source_span:
            if len(source_span.split()) > 4 or _is_redundant_span(source_span, date_text):
                source_span = _shorten_source_span(source_span, max_words=4)
                if _is_redundant_span(source_span, date_text):
                    source_span = " ".join(date_text.split()[:3])

        d["date_text"] = date_text
        d["normalized_date"] = normalized_date
        d["date_type"] = _safe_str(d.get("date_type"), 40)
        d["source_span"] = source_span
        try:
            d["confidence"] = float(d.get("confidence", 0.0))
        except:
            d["confidence"] = 0.0
        if date_text.strip():
            cleaned.append(d)
    return cleaned

def _clean_events(events):
    cleaned = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        event_name = _safe_str(ev.get("event_name"), 150)
        normalized = _normalize_name_text(event_name)
        source_span = _safe_str(ev.get("source_span"), 200)

        if source_span:
            if len(source_span.split()) > 4 or _is_redundant_span(source_span, event_name):
                source_span = _shorten_source_span(source_span, max_words=4)
                if _is_redundant_span(source_span, event_name):
                    source_span = " ".join(event_name.split()[:3])

        ev["event_name"] = event_name
        ev["normalized_name"] = normalized
        ev["event_date"] = _safe_str(ev.get("event_date"), 50)
        ev["event_type"] = _safe_str(ev.get("event_type"), 80)
        ev["description"] = _safe_str(ev.get("description"), 200)
        ev["significance"] = _safe_str(ev.get("significance"), 200)
        ev["source_span"] = source_span
        try:
            ev["confidence"] = float(ev.get("confidence", 0.0))
        except:
            ev["confidence"] = 0.0
        if event_name.strip():
            cleaned.append(ev)
    return cleaned

def _clean_discoveries(discoveries):
    cleaned = []
    for disc in discoveries:
        if not isinstance(disc, dict):
            continue
        discovery_name = _safe_str(disc.get("discovery_name"), 150)
        normalized = _normalize_name_text(discovery_name)
        source_span = _safe_str(disc.get("source_span"), 200)

        if source_span:
            if len(source_span.split()) > 4 or _is_redundant_span(source_span, discovery_name):
                source_span = _shorten_source_span(source_span, max_words=4)
                if _is_redundant_span(source_span, discovery_name):
                    source_span = " ".join(discovery_name.split()[:3])

        disc["discovery_name"] = discovery_name
        disc["normalized_name"] = normalized
        disc["description"] = _safe_str(disc.get("description"), 200)
        disc["date"] = _safe_str(disc.get("date"), 50)
        disc["significance"] = _safe_str(disc.get("significance"), 200)
        disc["source_span"] = source_span
        try:
            disc["confidence"] = float(disc.get("confidence", 0.0))
        except:
            disc["confidence"] = 0.0
        if discovery_name.strip():
            cleaned.append(disc)
    return cleaned

def _clean_gems(gems):
    cleaned = []
    for g in gems:
        if not isinstance(g, dict):
            continue
        gem_text = _safe_str(g.get("gem_text"), 200)
        source_span = _safe_str(g.get("source_span"), 200)

        if source_span:
            if len(source_span.split()) > 4 or _is_redundant_span(source_span, gem_text):
                source_span = _shorten_source_span(source_span, max_words=4)
                if _is_redundant_span(source_span, gem_text):
                    source_span = " ".join(gem_text.split()[:3])

        g["gem_text"] = gem_text
        g["category"] = _safe_str(g.get("category"), 80)
        g["source_span"] = source_span
        try:
            g["importance"] = float(g.get("importance", 0.0))
        except:
            g["importance"] = 0.0
        try:
            g["confidence"] = float(g.get("confidence", 0.0))
        except:
            g["confidence"] = 0.0
        if gem_text.strip():
            cleaned.append(g)
    return cleaned

def _clean_relationships(rels):
    cleaned = []
    for r in rels:
        if not isinstance(r, dict):
            continue
        source_node = _safe_str(r.get("source_node"), 150)
        target_node = _safe_str(r.get("target_node"), 150)
        evidence_span = _safe_str(r.get("evidence_span"), 200)

        # For relationships, evidence_span can be slightly longer but still short
        if evidence_span:
            if len(evidence_span.split()) > 5:
                evidence_span = _shorten_source_span(evidence_span, max_words=5)

        r["source_node"] = source_node
        r["target_node"] = target_node
        r["relation_type"] = _safe_str(r.get("relation_type"), 80)
        r["evidence_span"] = evidence_span
        try:
            r["confidence"] = float(r.get("confidence", 0.0))
        except:
            r["confidence"] = 0.0
        if source_node.strip() and target_node.strip():
            cleaned.append(r)
    return cleaned


# ============================================================
#  NOVELTY GATING
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

def _process_batch(batch_chunks, model=None, logic_context="", endpoint=None):
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
        uncached_indices = []
        for i, chunk in enumerate(batch_chunks):
            chunk_hash = _hash_text(chunk)
            prompt_template = {
                "facts_entities_relationships": FACTS_ENTITIES_PROMPT_BATCH,
                "people_locations_dates": PEOPLE_LOCATIONS_DATES_PROMPT_BATCH,
                "events_discoveries_gems": EVENTS_DISCOVERIES_GEMS_PROMPT_BATCH,
            }[category]
            cached = _get_cached(chunk_hash, category, model, 0, prompt_template)
            if cached is None:
                uncached_indices.append(i)
            else:
                for key in field_keys:
                    if key in cached:
                        results[i][key] = cached[key]

        if not uncached_indices:
            continue

        resp = _extract_category_batch(category, batch_chunks, model, logic_context, endpoint)
        if resp is None:
            resp = {}
        if isinstance(resp, list):
            if resp and isinstance(resp[0], dict):
                resp = resp[0]
            else:
                resp = {}

        for i in uncached_indices:
            key = f"chunk_{i}"
            chunk_data = resp.get(key, {}) if isinstance(resp, dict) else {}
            chunk_data = _normalize_chunk_data(chunk_data)
            chunk_hash = _hash_text(batch_chunks[i])
            prompt_template = {
                "facts_entities_relationships": FACTS_ENTITIES_PROMPT_BATCH,
                "people_locations_dates": PEOPLE_LOCATIONS_DATES_PROMPT_BATCH,
                "events_discoveries_gems": EVENTS_DISCOVERIES_GEMS_PROMPT_BATCH,
            }[category]
            _set_cached(chunk_hash, category, model, 0, chunk_data, prompt_template)
            for field in field_keys:
                if field in chunk_data:
                    results[i][field] = chunk_data[field]

    # Cleaning
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


def extract_from_chunks(chunks, model=None, max_workers=None, chunk_embeddings=None, logic_context=""):
    _init_cache()

    if chunk_embeddings is None:
        print("  (No chunk embeddings provided; computing embeddings for novelty gating...)")
        chunk_embeddings = get_embeddings_batch(chunks, batch_size=config.EMBEDDING_BATCH_SIZE)

    flags = _compute_novelty_flags(chunks, chunk_embeddings)

    selected_items = [(i, chunks[i]) for i in range(len(chunks)) if flags[i]]
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

    # Build task queue
    task_queue = queue.Queue()
    for batch_idx, batch in enumerate(batches):
        batch_texts = [text for _, text in batch]
        task_queue.put((batch_idx, batch_texts))

    results = {}
    lock = threading.Lock()

    # Create worker threads per endpoint capacity
    workers = []
    for ep_idx, endpoint in enumerate(config.LLM_ENDPOINTS):
        capacity = config.LLM_ENDPOINT_CAPACITIES[ep_idx] if ep_idx < len(config.LLM_ENDPOINT_CAPACITIES) else 1
        for _ in range(capacity):
            def worker(ep=endpoint):
                while True:
                    try:
                        batch_idx, batch_texts = task_queue.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        batch_results = _process_batch(batch_texts, model=None, logic_context=logic_context, endpoint=ep)
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

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            workers.append(t)

    # Wait for all tasks to be processed
    try:
        task_queue.join()
    except KeyboardInterrupt:
        print("\n  (KeyboardInterrupt received: stopping chunk processing...)")
        # daemon threads will be killed when process exits
        raise

    # Map results back to all_results in correct order
    for batch_idx in sorted(results.keys()):
        batch_indices = [idx for idx, _ in batches[batch_idx]]
        batch_results = results[batch_idx]
        for original_idx, res in zip(batch_indices, batch_results):
            all_results[original_idx] = res

    return all_results