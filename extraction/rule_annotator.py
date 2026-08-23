import re
from pathlib import Path
import config

# Precompiled patterns for performance
_YEAR_RE = re.compile(r'\b(17|18|19|20)\d{2}\b')
_DATE_RE = re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE)
_PERSON_RE = re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b')

# Load gazetteers lazily
_gazetteers = None

def load_gazetteers():
    global _gazetteers
    if _gazetteers is not None:
        return _gazetteers

    _gazetteers = {
        "countries": set(),
        "us_states": set(),
        "world_cities": set(),
        "first_names": set(),
        "last_names": set(),
        "organization_suffixes": set(),
        "event_triggers": set(),
    }

    # Try to load from files if present; otherwise use minimal defaults
    gaz_dir = config.GAZETTEERS_DIR
    for key, filename in [
        ("countries", "countries.txt"),
        ("us_states", "us_states.txt"),
        ("world_cities", "world_cities.txt"),
        ("first_names", "first_names.txt"),
        ("last_names", "last_names.txt"),
        ("organization_suffixes", "organizations_suffixes.txt"),
        ("event_triggers", "event_triggers.txt"),
    ]:
        path = gaz_dir / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                _gazetteers[key] = {line.strip().lower() for line in f if line.strip()}
        else:
            # Minimal fallback sets
            if key == "organization_suffixes":
                _gazetteers[key] = {"inc", "corp", "ltd", "university", "institute", "agency", "department", "foundation"}
            elif key == "event_triggers":
                _gazetteers[key] = {"discovered", "founded", "invented", "first", "occurred", "published", "launched", "created", "established"}
            # others empty by default
    return _gazetteers


def pre_annotate(text: str) -> dict:
    """
    Use regex and gazetteers to detect candidate entities before LLM.
    Returns dict of annotation spans grouped by type.
    """
    gaz = load_gazetteers()

    annotations = {
        "years": [],
        "dates": [],
        "locations": [],
        "people": [],
        "organizations": [],
        "events": [],
    }

    # Years
    for m in _YEAR_RE.finditer(text):
        annotations["years"].append({"text": m.group(), "start": m.start(), "end": m.end()})

    # Full dates: January 1, 2020 or 1 January 2020
    date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b'
    for m in _DATE_RE.finditer(text):
        annotations["dates"].append({"text": m.group(), "start": m.start(), "end": m.end()})

    # Locations: match gazetteer entries (exact word boundary)
    for loc in gaz["countries"] | gaz["us_states"] | gaz["world_cities"]:
        if not loc:
            continue
        pattern = r'\b' + re.escape(loc) + r'\b'
        for m in re.finditer(pattern, text, re.IGNORECASE):
            annotations["locations"].append({"text": m.group(), "start": m.start(), "end": m.end()})

    # People: capital word + capital word (simple heuristic)
    person_pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
    for m in _PERSON_RE.finditer(text):
        name = m.group()
        first = name.split()[0].lower()
        if first in gaz["first_names"] or len(gaz["first_names"]) == 0:
            annotations["people"].append({"text": name, "start": m.start(), "end": m.end()})

    # Organizations: word + suffix
    suffixes = gaz["organization_suffixes"]
    if suffixes:
        suffix_pattern = r'\b[A-Za-z0-9&]+(?:\s+[A-Za-z0-9&]+)*\s+(?:' + '|'.join(re.escape(s) for s in suffixes) + r')\b'
        for m in re.finditer(suffix_pattern, text, re.IGNORECASE):
            annotations["organizations"].append({"text": m.group(), "start": m.start(), "end": m.end()})

    # Events: trigger words
    triggers = gaz["event_triggers"]
    if triggers:
        trigger_pattern = r'\b(?:' + '|'.join(re.escape(t) for t in triggers) + r')\b'
        for m in re.finditer(trigger_pattern, text, re.IGNORECASE):
            # Extend a bit to capture surrounding sentence
            start = max(0, m.start() - 100)
            end = min(len(text), m.end() + 200)
            annotations["events"].append({"text": text[start:end], "start": start, "end": end})

    return annotations