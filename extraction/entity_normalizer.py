import re
from datetime import datetime


def normalize_person_name(name: str) -> str:
    """Normalize a person name: title case, collapse spaces."""
    name = re.sub(r'\s+', ' ', name).strip()
    parts = name.split()
    if len(parts) >= 2:
        return ' '.join(parts)  # preserve case for now
    return name.title()


def normalize_location(location: str, gazetteers=None) -> str:
    """Basic normalization: strip punctuation, title case."""
    loc = re.sub(r'[^\w\s]', '', location).strip()
    return loc.title()


def normalize_date(date_text: str) -> str:
    """Convert common date strings to YYYY-MM-DD or YYYY."""
    # Try year only
    year_match = re.search(r'(17|18|19|20)\d{2}', date_text)
    if year_match:
        year = year_match.group()
        # Try full date
        date_patterns = [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b',
            r'\b(\d{1,2})\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
        ]
        for pat in date_patterns:
            m = re.search(pat, date_text, re.IGNORECASE)
            if m:
                month = m.group(1) if pat.startswith(r'\b\d') else m.group(1)
                day = m.group(2) if pat.startswith(r'\b\d') else m.group(2)
                year = m.group(3)
                month_num = {
                    'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
                    'july':7,'august':8,'september':9,'october':10,'november':11,'december':12
                }.get(month.lower(), 1)
                try:
                    return f"{year}-{month_num:02d}-{int(day):02d}"
                except:
                    pass
        return year
    return date_text