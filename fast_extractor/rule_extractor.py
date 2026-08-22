"""
Enhanced rule-based extractor for fast extraction.
"""
import re
from pathlib import Path
import config

# Additional gazetteers can be loaded from files, but we include some inline patterns.
ORG_SUFFIXES = {"inc", "corp", "ltd", "llc", "university", "institute", "agency", "department", "foundation", "company"}

def extract_entities_rules(text):
    entities = []
    # Dates
    date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b|\b(?:17|18|19|20)\d{2}\b'
    for m in re.finditer(date_pattern, text, re.IGNORECASE):
        entities.append(("DATE", m.group(), 0.95))
    # Organizations
    org_pattern = r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\s+(?:' + '|'.join(ORG_SUFFIXES) + r')\b'
    for m in re.finditer(org_pattern, text, re.IGNORECASE):
        entities.append(("ORG", m.group(), 0.85))
    # Locations (simple: capitalized words after 'in' or 'at')
    loc_pattern = r'\b(?:in|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
    for m in re.finditer(loc_pattern, text):
        entities.append(("LOC", m.group(1), 0.7))
    return entities
