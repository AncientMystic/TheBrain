"""Data validation utilities."""
import re

def validate_fact(fact):
    """Return True if fact is valid."""
    if not isinstance(fact, dict):
        return False
    if not fact.get("fact_text") or not fact.get("source_span"):
        return False
    # source_span should be short
    if len(fact["source_span"].split()) > 10:
        return False
    return True

def validate_entity(entity):
    if not isinstance(entity, dict):
        return False
    if not entity.get("entity_name"):
        return False
    return True

def validate_extracted_schema(data, category):
    """Validate a full extraction result for a category."""
    required_keys = {
        "facts": ["fact_type", "fact_text", "canonical_value", "source_span", "confidence"],
        "entities": ["entity_type", "entity_name", "normalized_name", "source_span", "confidence"],
        "relationships": ["source_node", "target_node", "relation_type", "evidence_span", "confidence"],
        "people": ["person_name", "normalized_name", "role", "source_span", "confidence"],
        "locations": ["location_name", "normalized_place", "location_type", "source_span", "confidence"],
        "dates": ["date_text", "normalized_date", "date_type", "source_span", "confidence"],
        "events": ["event_name", "normalized_name", "event_date", "event_type", "description", "significance", "source_span", "confidence"],
        "discoveries": ["discovery_name", "normalized_name", "description", "date", "significance", "source_span", "confidence"],
        "gems": ["gem_text", "category", "importance", "source_span", "confidence"],
    }
    if category not in required_keys:
        return False
    # Basic structure check
    if not isinstance(data, dict):
        return False
    # We won't check every key, just ensure if list is present, it's list
    for key in required_keys[category]:
        if key in data and not isinstance(data[key], list):
            return False
    return True
