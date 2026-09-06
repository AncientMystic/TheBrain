"""
PII redaction for stored memories (Irene-style masking, identity-sensitive).

Patterns (always generic, no locale hardcoding beyond digit shapes):
emails, international phone runs, SSN-like triples, credit-card runs verified
by Luhn (avoids false positives on order numbers). Person names only when
explicitly enabled, via the optional linguistic backend (never regex names).
Redaction happens BEFORE embedding so vectors cannot leak raw identifiers.
Returns (redacted_text, hits) where hits list the masked classes only.
"""
import re
import logging

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d .()-]{7,}\d)(?!\d)")
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_CARD_RUN_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def _luhn_ok(digits):
    """Luhn checksum over digit string (even-length sensitive, principled)."""
    try:
        total = 0
        for i, ch in enumerate(reversed(digits)):
            d = ord(ch) - 48
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            total += d
        return total % 10 == 0
    except Exception:
        return False


def _mask_names(text):
    """Replace PERSON entities with indexed placeholders (optional backend)."""
    try:
        import config as _cfg
        if not getattr(_cfg, "MEMORY_PII_REDACT_NAMES", False):
            return text, []
    except Exception:
        return text, []
    try:
        from extraction.nlp_primitives import _spacy_doc
    except Exception:
        return text, []
    try:
        doc = _spacy_doc(text)
        if doc is None:
            return text, []
        spans = [(ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ in ("PERSON", "PER")]
        if not spans:
            return text, []
        out, pos, hits = [], 0, []
        for idx, (s, e) in enumerate(sorted(spans)):
            if s < pos:
                continue
            out.append(text[pos:s])
            tag = f"[PERSON_{idx + 1}]"
            out.append(tag)
            hits.append("person-name")
            pos = e
        out.append(text[pos:])
        return "".join(out), hits
    except Exception:
        return text, []


def redact(text):
    """Redact PII from memory-bound text. Returns (redacted, hit_classes)."""
    if not text:
        return text, []
    hits = []
    try:
        import config as _cfg
        if not getattr(_cfg, "MEMORY_PII_REDACT", True):
            return text, []
    except Exception:
        pass
    try:
        out = text

        def _card_sub(m):
            digits = re.sub(r"\D", "", m.group(0))
            if 13 <= len(digits) <= 19 and _luhn_ok(digits):
                hits.append("credit-card")
                return "[CARD]"
            return m.group(0)

        out, n = _CARD_RUN_RE.subn(_card_sub, out)
        out, n = _SSN_RE.subn("[SSN]", out)
        if n:
            hits.append("ssn")
        out, n = _EMAIL_RE.subn("[EMAIL]", out)
        if n:
            hits.append("email")
        out, n = _PHONE_RE.subn("[PHONE]", out)
        if n:
            hits.append("phone")
        out, name_hits = _mask_names(out)
        hits.extend(name_hits)
        return out, sorted(set(hits))
    except Exception:
        logger.warning("Unexpected exception occurred", exc_info=True)
        return text, []
