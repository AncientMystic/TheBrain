import config
import re
from collections import Counter
import logging
logger = logging.getLogger(__name__)

# Minimal stopwords set; can be expanded by gazetteer later.
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
    'where', 'which', 'with', 'without', 'from', 'into', 'onto', 'of',
    'for', 'on', 'at', 'by', 'in', 'to', 'is', 'was', 'are', 'were',
    'be', 'been', 'being', 'am', 'as', 'it', 'its', 'this', 'that',
    'these', 'those', 'he', 'she', 'they', 'them', 'their', 'his', 'her',
    'we', 'you', 'i', 'me', 'my', 'your', 'our', 'has', 'have', 'had',
    'do', 'does', 'did', 'not', 'no', 'yes', 'can', 'could', 'should',
    'would', 'may', 'might', 'must', 'shall', 'will', 'about', 'such',
    'some', 'any', 'each', 'every', 'all', 'both', 'few', 'more', 'most',
    'other', 'same', 'so', 'than', 'too', 'very', 'just', 'also',
    'there', 'here', 'what', 'who', 'whom', 'whose', 'why', 'how',
    'know', 'tell', 'give', 'info', 'information',
    'detail', 'details', 'explain', 'explanation', 'show', 'list',
    'summarize', 'summary', 'extensively', 'please', 'provide',
    'can', 'could', 'would', 'should', 'will', 'shall', 'may', 'might',
    'must', 'do', 'does', 'did', 'done', 'doing', 'get', 'got',
    'want', 'need', 'like',
    'topic', 'topics', 'subject', 'subjects', 'question', 'questions',
    'un', 'una', 'uno', 'unas', 'unos', 'el', 'la', 'los', 'las', 'de',
    'del', 'que', 'en', 'con', 'por', 'para', 'como', 'pero', 'más',
    'mas', 'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas',
    'lo', 'le', 'se', 'su', 'sus', 'al',
    'il', 'lo', 'i', 'gli', 'le', 'un', 'uno', 'una', 'per', 'con', 'su',
    'tra', 'fra', 'che', 'di', 'da', 'in', 'a', 'e', 'sono', 'era',
    'come', 'più',
    'le', 'la', 'les', 'des', 'du', 'une', 'un', 'et', 'ou', 'donc',
    'mais', 'que', 'qui', 'quoi', 'dans', 'pour', 'sur', 'avec', 'sans',
    'sous', 'est', 'sont', 'être', 'avoir', 'aux', 'ce', 'cette', 'ces',
    'il', 'elle', 'ils', 'elles', 'nous', 'vous', 'je', 'tu', 'ne', 'pas',
    'plus', 'moins', 'où', 'ni', 'car', 'quand',
    'der', 'die', 'das', 'und', 'oder', 'nicht', 'mit', 'von', 'für',
    'auf', 'ein', 'eine', 'einen', 'einem', 'einer', 'eines', 'dem',
    'den', 'des', 'zu', 'im', 'an', 'sich', 'ist', 'sind', 'war',
    'wurde', 'wird', 'wie', 'auch', 'bei', 'aus', 'nach', 'über',
    'unter', 'zwischen',
    'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas', 'em', 'com',
    'para', 'por', 'não', 'sim', 'como', 'mais', 'mas', 'se', 'de',
    'da', 'do', 'das', 'dos',
}


def dehyphenate(text: str) -> str:
    """Join PDF line-break artifacts like 'man- tle' -> 'mantle'.

    Only joins when a lowercase letter follows the break (line-wrap split).
    Intentional compounds before uppercase/digits keep their hyphen.
    Generic, no document-specific word lists.
    """
    if not isinstance(text, str) or "-" not in text:
        return text
    return re.sub(r"(\w)-\s+([a-zà-öø-ÿ])", r"\1\2", text)


def normalise_text(text: str) -> str:
    """Collapse whitespace and strip leading/trailing spaces. Handles None/dict/list."""
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            logger.warning("Unexpected exception occurred", exc_info=True)
            return ""
    text = dehyphenate(text)
    return re.sub(r'\s+', ' ', text).strip()


def tokenize(text: str):
    """Lowercase, replace non-alphanumeric with space, split, remove stopwords."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1 and t not in STOPWORDS]
    return tokens


def get_bigrams(tokens):
    """Return set of bigrams from token list."""
    return set(zip(tokens, tokens[1:])) if len(tokens) >= 2 else set()


def chunk_text(text, chunk_size=None, overlap=None):
    """
    Split text into chunks of approximately chunk_size characters,
    with a specified overlap between chunks. Splits at sentence boundaries
    when possible, then falls back to fixed-size splitting.
    """
    if chunk_size is None:
        import config
        chunk_size = config.CHUNK_SIZE
    if overlap is None:
        import config
        overlap = config.CHUNK_OVERLAP

    text = normalise_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    # Try sentence-based chunking
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > chunk_size and current:
            chunks.append(current.strip())
            # Keep overlap by starting with the tail of current
            if overlap > 0:
                overlap_len = min(overlap, len(current))
                current = current[-overlap_len:]
            else:
                current = ""
        current = (current + " " + sent).strip() if current else sent

    if current.strip():
        chunks.append(current.strip())

    # Further split any chunks that are too long (e.g., very long sentences)
    final_chunks = []
    for chunk in chunks:
        while len(chunk) > chunk_size:
            # Find a good split point
            split_at = chunk.rfind(' ', 0, chunk_size)
            if split_at == -1:
                split_at = chunk_size
            final_chunks.append(chunk[:split_at].strip())
            # Move to next part, taking overlap from previous chunk
            if overlap > 0:
                chunk = chunk[max(0, split_at - overlap):].strip()
            else:
                chunk = chunk[split_at:].strip()
        if chunk:
            final_chunks.append(chunk.strip())

    return final_chunks
