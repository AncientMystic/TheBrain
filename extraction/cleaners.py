RELAXED_MODE = False
"""Utility functions for cleaning extracted data."""
import re
__all__ = [
    '_truncate', '_safe_str', '_normalize_text_for_compare', '_is_verbatim_copy',
    '_shorten_source_span', '_normalize_name_text', '_is_redundant_span',
    '_clean_facts', '_clean_entities', '_clean_relationships',
    '_clean_people', '_clean_locations', '_clean_dates',
    '_clean_events', '_clean_discoveries', '_clean_gems',
]


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

def _clean_facts(facts):
    cleaned = []
    for f in facts:
        if isinstance(f, str):
            f = {
                "fact_type": "other",
                "fact_text": f,
                "canonical_value": "",
                "source_span": "",
                "confidence": 0.7,
            }
        elif not isinstance(f, dict):
            continue
        fact_text = _safe_str(f.get("fact_text"), 200)
        source_span = _safe_str(f.get("source_span"), 200)

        if RELAXED_MODE:
            # In relaxed mode, skip aggressive verbatim/redundancy checks.
            # Just ensure fact_text is non-empty and confidence >= 0.0.
            if not fact_text.strip():
                continue
            f["fact_text"] = fact_text
            f["canonical_value"] = _safe_str(f.get("canonical_value"), 80)
            f["source_span"] = source_span
            f["fact_type"] = _safe_str(f.get("fact_type"), 80)
            try:
                f["confidence"] = float(f.get("confidence", 0.0))
            except:
                f["confidence"] = 0.0
            cleaned.append(f)
            continue

        # Normal (strict) mode - original logic
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
        gem_text = _safe_str(g.get("gem_text") or g.get("discovery_name"), 200)
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
