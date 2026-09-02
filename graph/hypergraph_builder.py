import json
from core import db
import logging
logger = logging.getLogger(__name__)


def _safe_str(value):
    """Convert any value to string, or return empty string for None/dict/list."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        logger.warning("Unexpected exception occurred", exc_info=True)
        return ""


def build_hypergraph(doc_hash: str, extracted_data: dict, chunk_map: dict) -> None:
    """
    Build intra-document graph nodes and edges from LLM extraction results.
    Includes relationship edges and co-occurrence edges (if chunk_map provided).
    """
    conn = db.db_connect("hypergraph")
    cur = conn.cursor()

    node_id_map = {}

    def get_or_create_node(node_type, node_text, normalized_name, source_span="", confidence=0.0):
        key = (node_type, normalized_name)
        if key in node_id_map:
            return node_id_map[key]
        cur.execute("""
            INSERT INTO nodes (doc_hash, node_type, node_text, normalized_name, source_span, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (doc_hash, _safe_str(node_type), _safe_str(node_text), _safe_str(normalized_name), _safe_str(source_span), confidence))
        node_id = cur.lastrowid
        node_id_map[key] = node_id
        return node_id

    # Create nodes for facts
    for fact in extracted_data.get("facts", []):
        get_or_create_node("FACT", fact.get("fact_text"), fact.get("canonical_value"), fact.get("source_span"), fact.get("confidence", 0.0))

    # Entities
    for ent in extracted_data.get("entities", []):
        get_or_create_node(ent.get("entity_type", "ENTITY"), ent.get("entity_name"), ent.get("normalized_name"), ent.get("source_span"), ent.get("confidence", 0.0))

    # People
    for person in extracted_data.get("people", []):
        get_or_create_node("PERSON", person.get("person_name"), person.get("normalized_name"), person.get("source_span"), person.get("confidence", 0.0))

    # Locations
    for loc in extracted_data.get("locations", []):
        get_or_create_node("LOCATION", loc.get("location_name"), loc.get("normalized_place"), loc.get("source_span"), loc.get("confidence", 0.0))

    # Dates
    for date in extracted_data.get("dates", []):
        get_or_create_node("DATE", date.get("date_text"), date.get("normalized_date"), date.get("source_span"), date.get("confidence", 0.0))

    # Events
    for event in extracted_data.get("events", []):
        get_or_create_node("EVENT", event.get("event_name"), event.get("normalized_name"), event.get("source_span"), event.get("confidence", 0.0))

    # Discoveries
    for disc in extracted_data.get("discoveries", []):
        get_or_create_node("DISCOVERY", disc.get("discovery_name"), disc.get("normalized_name"), disc.get("source_span"), disc.get("confidence", 0.0))

    # Gems
    for gem in extracted_data.get("gems", []):
        get_or_create_node("GEM", gem.get("gem_text"), gem.get("gem_text"), gem.get("source_span"), gem.get("confidence", 0.0))

    # Relationships from LLM
    for rel in extracted_data.get("relationships", []):
        src_name = _safe_str(rel.get("source_node"))
        tgt_name = _safe_str(rel.get("target_node"))
        if not src_name or not tgt_name:
            continue
        # Find or create nodes
        cur.execute("SELECT node_id FROM nodes WHERE doc_hash=? AND (node_text=? OR normalized_name=?)",
                    (doc_hash, src_name, src_name))
        src_row = cur.fetchone()
        if not src_row:
            src_id = get_or_create_node("OTHER", src_name, src_name, _safe_str(rel.get("evidence_span")), rel.get("confidence", 0.0))
        else:
            src_id = src_row[0]

        cur.execute("SELECT node_id FROM nodes WHERE doc_hash=? AND (node_text=? OR normalized_name=?)",
                    (doc_hash, tgt_name, tgt_name))
        tgt_row = cur.fetchone()
        if not tgt_row:
            tgt_id = get_or_create_node("OTHER", tgt_name, tgt_name, _safe_str(rel.get("evidence_span")), rel.get("confidence", 0.0))
        else:
            tgt_id = tgt_row[0]

        cur.execute("""
            INSERT OR IGNORE INTO edges (doc_hash, source_node_id, target_node_id, relation_type, weight, evidence_span, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (doc_hash, src_id, tgt_id, _safe_str(rel.get("relation_type", "related")), 1.0, _safe_str(rel.get("evidence_span")), rel.get("confidence", 0.0)))

    # Optional: co-occurrence edges if chunk_map provided (not implemented in current call)
    # If chunk_map contains chunk_index -> list of node_ids, we could add co-occurrence edges here.
    # For now, rely on explicit relationships.

    # Populate doc_entity_nodes for entity-type nodes
    for (node_type, normalized_name), node_id in node_id_map.items():
        if node_type in ('ENTITY', 'PERSON', 'LOCATION', 'DATE', 'EVENT', 'DISCOVERY', 'GEM'):
            cur.execute("""
                INSERT OR IGNORE INTO doc_entity_nodes (doc_hash, entity_type, entity_name, node_id)
                VALUES (?, ?, ?, ?)
            """, (doc_hash, node_type, normalized_name, node_id))

    conn.commit()
    conn.close()
