"""
Deterministic linguistic primitives upstream of the LLM (Holmes layer 2).

Always-available stdlib primitives (sentence segmentation) plus optional
spaCy/stanza backends that activate only when installed — never a hard
dependency, never a behavior change when absent (rule path is the fallback).
Optional backends never download models on their own; missing models fall back
silently with a debug note. Generic, no document-specific rules.
"""
import re
import logging

logger = logging.getLogger(__name__)

_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "etc", "vs",
           "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
           "oct", "nov", "dec"}


def split_sentences(text):
    """Deterministic sentence splitter (stdlib only).

    Protects common abbreviations and decimal numbers before splitting on
    end punctuation. Returns list of (sentence, start, end) in chunk coordinates.
    """
    if not text:
        return []
    protected = []
    try:
        # Protect abbreviations: replace the period with a placeholder.
        def _prot(m):
            protected.append(m.group(0))
            return f"\x00{len(protected) - 1}\x00"

        tmp = re.sub(r"\b(?:%s)\." % "|".join(sorted(_ABBREV)), _prot, text, flags=re.IGNORECASE)
        # Protect decimal numbers: 4.9 stays together.
        tmp = re.sub(r"(\d)\.(\d)", lambda m: m.group(1) + "\x01" + m.group(2), tmp)
        parts = re.split(r"(?<=[.!?])\s+", tmp)
        out = []
        pos = 0
        for p in parts:
            p = p.replace("\x01", ".")
            p = re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], p)
            s = p.strip()
            if not s:
                continue
            start = text.find(s[: min(20, len(s))], pos)
            if start < 0:
                start = pos
            out.append((s, start, start + len(s)))
            pos = start + len(s)
        return out
    except Exception:
        logger.warning("Unexpected exception occurred", exc_info=True)
        return [(text, 0, len(text))] if text.strip() else []


_SPACY_NLP = None
_SPACY_TRIED = False


def _spacy_doc(text):
    """Return spaCy Doc or None (never raises, never downloads).

    Model load attempted at most once per process; failures fall back silently.
    """
    global _SPACY_NLP, _SPACY_TRIED
    try:
        import spacy
    except ImportError:
        return None
    try:
        if _SPACY_NLP is None and not _SPACY_TRIED:
            _SPACY_TRIED = True
            try:
                import config as _cfg
                _name = getattr(_cfg, "SPACY_MODEL", "en_core_web_sm")
            except Exception:
                _name = "en_core_web_sm"
            try:
                _SPACY_NLP = spacy.load(_name)
            except OSError:
                try:
                    if getattr(__import__("config"), "DEBUG_VERBOSE", False):
                        print(f"    (spaCy model {_name} not installed; rule path active)")
                except Exception:
                    pass
                _SPACY_NLP = None
        if _SPACY_NLP is None:
            return None
        return _SPACY_NLP(text[:100000])
    except Exception:
        return None


def pos_filtered_entities(rule_entities, text):
    """Filter rule/ONNX entities through spaCy POS tags when available.

    Drops person candidates whose tokens are not proper nouns and organization
    candidates headed by verbs — the two largest false-positive classes of the
    capital-word heuristic. Without spaCy, returns input unchanged.
    """
    doc = _spacy_doc(text)
    if doc is None:
        return rule_entities
    try:
        pos_by_offset = {}
        for tok in doc:
            for i in range(tok.idx, tok.idx + len(tok.text)):
                pos_by_offset[i] = tok.pos_
        kept = []
        for ent in rule_entities:
            txt = ent.get("text", "") if isinstance(ent, dict) else str(ent)
            try:
                start = text.lower().find(str(txt).lower())
            except Exception:
                start = -1
            if start < 0:
                kept.append(ent)
                continue
            tags = {pos_by_offset.get(start + k, "") for k in range(len(str(txt)))}
            kind = ent.get("type", "") if isinstance(ent, dict) else ""
            if kind == "PERSON" and not (tags & {"PROPN"}):
                continue
            if kind == "ORG" and (tags & {"VERB"}):
                continue
            kept.append(ent)
        return kept
    except Exception:
        return rule_entities
