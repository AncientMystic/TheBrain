"""
Lightweight statistical keyword extraction using RAKE-style heuristics.
"""
import re
from collections import Counter

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
}


def extract_rake_phrases(text: str, max_phrases=20):
    """
    Extract keyword phrases using stopword delimiters and term frequency.
    """
    # Split into sentences then phrases by stopwords
    phrases = []
    # Simple: split on stopwords and punctuation
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    current_phrase = []
    for w in words:
        if w in STOPWORDS:
            if current_phrase:
                phrases.append(' '.join(current_phrase))
                current_phrase = []
        else:
            current_phrase.append(w)
    if current_phrase:
        phrases.append(' '.join(current_phrase))

    # Count frequency
    freq = Counter(phrases)
    # Remove very short or single character phrases
    filtered = [(p, c) for p, c in freq.items() if len(p.split()) >= 1 and len(p) > 2]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return [p for p, c in filtered[:max_phrases]]
