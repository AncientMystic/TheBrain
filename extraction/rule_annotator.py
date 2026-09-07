import re
from pathlib import Path
import config
import logging
logger = logging.getLogger(__name__)

# Precompiled patterns for performance
_YEAR_RE = re.compile(r'\b(17|18|19|20)\d{2}\b')
_DATE_RE = re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE)
_PERSON_RE = re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b')
# P1 extensions: numeric/ISO dates, ranges, qualifiers (no behavior change to existing buckets)
_DATE_ISO_RE = re.compile(r'\b(17|18|19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b')
_DATE_NUM_RE = re.compile(r'\b(0?[1-9]|1[0-2])[/.](0?[1-9]|[12]\d|3[01])[/.](17|18|19|20)\d{2}\b|\b(0?[1-9]|[12]\d|3[01])[.](0?[1-9]|1[0-2])[.](17|18|19|20)\d{2}\b')
_DATE_MY_RE = re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[,.]?\s+(17|18|19|20)\d{2}\b', re.IGNORECASE)
_DATE_RANGE_RE = re.compile(r'\b(17|18|19|20)\d{2}\s*[–—-]\s*(17|18|19|20)\d{2}\b')
_DATE_QUAL_RE = re.compile(r'\b(?:c\.|ca\.|circa|fl\.|floruit)\s*(17|18|19|20)\d{2}\b', re.IGNORECASE)
# Page references: p. 42, pp. 42-45, pg. 7, Page 12, S. 42 (German), fol. 3
_PAGE_RE = re.compile(r'\b(?:pp?g?\.|pages?|S\.|fol\.|fols\.)\s*\d+(?:\s*[–—-]\s*\d+)?\b', re.IGNORECASE)
# Identifiers: ISBN-10/13, ISSN, DOI
_ISBN_RE = re.compile(r'\bISBN(?:-1[03])?:?\s*(?=[0-9Xx][0-9Xx\- ]{9,16}[0-9Xx])(?:[0-9Xx][\- ]?){10,13}\b')
_ISSN_RE = re.compile(r'\bISSN:?\s*\d{4}-\d{3}[\dXx]\b', re.IGNORECASE)
_DOI_RE = re.compile(r'\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b')
# Bylines: By Jane Smith / BY JOHN SMITH / Author: X / Written by X
_BYLINE_RE = re.compile(r'(?m)^(?:by|author(?:\(s\))?|written by|reported by)\s*[:\-]?\s*([A-Z][\w.\'-]+(?:\s+[A-Z][\w.\'-]+){0,3})\s*$', re.IGNORECASE)
# Datelines: PARIS — / Paris, France, Jan 5 — / LONDON, Jan. 5 (AP):
_DATELINE_RE = re.compile(r'(?m)^([A-Z][A-Za-z .\'-]{1,40}?)(?:,\s*([A-Za-z .\'-]{1,40}?))?\s*(?:—|--|–|:)\s*')
# Transcript speaker turns: NAME: ... / [00:12:33] Name: ... / SPEAKER 1: ...
_SPEAKER_RE = re.compile(r'(?m)^(?:\[(\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?\]\s*)?([A-Z][A-Za-z .\'-]{1,32}?|SPEAKER\s+\d+|UNKNOWN):\s+')
# Citations: Author (Year) / Author & Other (Year) / Author et al. (Year)
_CITE_RE = re.compile(r'\b([A-Z][a-z]+(?:\s+(?:&|and)\s+[A-Z][a-z]+)?(?:\s+et al\.?)?)\s*\(((?:17|18|19|20)\d{2}[a-z]?)\)')
# Chapter/section headers
_CHAPTER_RE = re.compile(r'(?m)^(?:chapter|section|part)\s+([IVXLCDM\d]+|[A-Z][\w .\'-]{0,60})\s*$', re.IGNORECASE)

# Lazy-loaded Aho-Corasick automaton
_automaton = None

def _build_automaton():
    global _automaton
    if _automaton is not None:
        return _automaton
    import ahocorasick
    A = ahocorasick.Automaton()
    gaz = load_gazetteers()
    for kind in ("countries", "us_states", "world_cities", "first_names", "last_names", "organization_suffixes"):
        for term in gaz.get(kind, []):
            term_lower = term.lower()
            A.add_word(term_lower, (kind, term_lower))
    A.make_automaton()
    _automaton = A
    return A

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
        "pages": [],
        "identifiers": [],
        "bylines": [],
        "datelines": [],
        "speakers": [],
        "citations": [],
        "chapters": [],
    }

    def _add(_bucket, _m, _group=0):
        try:
            annotations[_bucket].append({"text": _m.group(_group), "start": _m.start(_group), "end": _m.end(_group)})
        except Exception:
            pass

    # Years
    for m in _YEAR_RE.finditer(text):
        annotations["years"].append({"text": m.group(), "start": m.start(), "end": m.end()})

    # Full dates: January 1, 2020 or 1 January 2020
    for m in _DATE_RE.finditer(text):
        annotations["dates"].append({"text": m.group(), "start": m.start(), "end": m.end()})

    # P1: ISO, numeric, month-year, ranges, qualified dates (all -> dates bucket)
    for _rx in (_DATE_ISO_RE, _DATE_NUM_RE, _DATE_MY_RE, _DATE_RANGE_RE, _DATE_QUAL_RE):
        for m in _rx.finditer(text):
            _add("dates", m)

    # P1: page references
    for m in _PAGE_RE.finditer(text):
        _add("pages", m)

    # P1: ISBN / ISSN / DOI
    for _rx in (_ISBN_RE, _ISSN_RE, _DOI_RE):
        for m in _rx.finditer(text):
            _add("identifiers", m)

    # P1: bylines (group 1 = name)
    for m in _BYLINE_RE.finditer(text):
        _add("bylines", m, 1)

    # P1: datelines (group 1 = place, group 2 = region/date-ish).
    # Skip single-word-plus-colon (that's a transcript speaker turn, which
    # gets its own bucket below) unless a region followed the place.
    for m in _DATELINE_RE.finditer(text):
        try:
            _g1 = (m.group(1) or "").strip()
            if not m.group(2) and m.group(0).rstrip().endswith(":") and " " not in _g1:
                continue
        except Exception:
            pass
        _add("datelines", m, 1)
        try:
            if m.group(2):
                annotations["datelines"].append({"text": m.group(2).strip(), "start": m.start(2), "end": m.end(2)})
        except Exception:
            pass

    # P1: transcript speaker turns (group 2 = speaker)
    for m in _SPEAKER_RE.finditer(text):
        _add("speakers", m, 2)

    # P1: Author (Year) citations (group 1 = authors, group 2 = year)
    for m in _CITE_RE.finditer(text):
        _add("citations", m, 0)

    # P1: chapter/section headers
    for m in _CHAPTER_RE.finditer(text):
        _add("chapters", m, 0)

    # Locations: use Aho-Corasick automaton for efficient exact word-boundary matching
    A = _build_automaton()
    for end_idx, (kind, term_lower) in A.iter(text.lower()):
        if kind in ("countries", "us_states", "world_cities"):
            start_idx = end_idx - len(term_lower) + 1
            # Verify word boundaries (Aho-Corasick doesn't enforce by default)
            if (start_idx == 0 or not text[start_idx-1].isalnum()) and                (end_idx == len(text)-1 or not text[end_idx+1].isalnum()):
                annotations["locations"].append({
                    "text": text[start_idx:end_idx+1],
                    "start": start_idx,
                    "end": end_idx+1
                })

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

    # Events: trigger words, extended to sentence boundaries (deterministic stdlib)
    triggers = gaz["event_triggers"]
    if triggers:
        trigger_pattern = r'\b(?:' + '|'.join(re.escape(t) for t in triggers) + r')\b'
        try:
            from extraction.nlp_primitives import split_sentences
            _sentences = split_sentences(text)
        except Exception:
            _sentences = []
        for m in re.finditer(trigger_pattern, text, re.IGNORECASE):
            start, end = max(0, m.start() - 100), min(len(text), m.end() + 200)
            for s_text, s_start, s_end in _sentences:
                if s_start <= m.start() and m.end() <= s_end:
                    start, end = s_start, s_end
                    break
            annotations["events"].append({"text": text[start:end], "start": start, "end": end})

    return annotations
