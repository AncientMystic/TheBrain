import re
from datetime import datetime
import logging
logger = logging.getLogger(__name__)


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
    """
    Normalize various date formats to ISO 8601 (YYYY-MM-DD) if possible,
    otherwise return the year or the original text.
    """
    if not date_text:
        return date_text

    month_map = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }

    # Pattern 1: "March 15, 2023" or "Mar 15, 2023"
    m = re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})\b', date_text, re.IGNORECASE)
    if m:
        month_name, day, year = m.groups()
        month_num = month_map.get(month_name.lower(), 1)
        try:
            return f"{year}-{month_num:02d}-{int(day):02d}"
        except (ValueError, TypeError):
            logger.warning("Unexpected exception occurred", exc_info=True)
            pass

    # Pattern 2: "15 March 2023" or "15 Mar 2023"
    m = re.search(r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})\b', date_text, re.IGNORECASE)
    if m:
        day, month_name, year = m.groups()
        month_num = month_map.get(month_name.lower(), 1)
        try:
            return f"{year}-{month_num:02d}-{int(day):02d}"
        except (ValueError, TypeError):
            logger.warning("Unexpected exception occurred", exc_info=True)
            pass

    # Pattern 3: Just a year
    m = re.search(r'\b(17|18|19|20)\d{2}\b', date_text)
    if m:
        return m.group()

    # Fallback: return original
    return date_text
