"""Trivium rule detector (phase 87): patterns live in trivium.db, never hardcoded.

Loads detection_pattern rows (REGEX + TARGET canonical fallacy/device),
applies them to input text, and cross-references matched fallacy targets
against their DB records. STRUCT: patterns route to structural checks
(abba-reversal, repeated-openings, sense-shift). Returns findings as
(fal_id, span, confidence, rule_id) — the query/validate loop consumes
these against mapping.db entity references downstream. Read-only on both
DBs. Never raises.
"""
import re

_PAT_CACHE = {}


def load_patterns(conn=None):
    """[(rule_id, compiled_rx_or_STRUCT, target_cid, note)]. Cached per conn id."""
    global _PAT_CACHE
    try:
        own = False
        if conn is None:
            from core import db as _db
            conn = _db.db_connect("trivium")
            own = True
        try:
            key = id(conn)
            if key in _PAT_CACHE:
                return _PAT_CACHE[key]
            out = []
            for row in conn.execute("SELECT canonical_id, description FROM entities"
                                    " WHERE entity_type='detection_pattern'"):
                try:
                    desc = row[1] or ""
                    target, rx = "", ""
                    for part in desc.split(". "):
                        if part.startswith("TARGET:"):
                            target = part[len("TARGET:"):].strip()
                        elif part.startswith("REGEX:"):
                            rx = part[len("REGEX:"):].strip()
                    if not (target and rx):
                        continue
                    if rx.startswith("STRUCT:"):
                        out.append((row[0], ("STRUCT", rx[len("STRUCT:"):]), target))
                    else:
                        out.append((row[0], re.compile(rx, re.IGNORECASE), target))
                except Exception:
                    continue
            _PAT_CACHE[key] = out
            return out
        finally:
            if own:
                try:
                    conn.close()
                except Exception:
                    pass
    except Exception:
        return []


def _struct_abba(text):
    """Mirrored phrase reversal (chiasmus signal): finds a split where the
    right side reversed equals the left, ignoring middle filler words
    (not/but/rather/than/for). Minimum 3 content words per side."""
    try:
        words = [w.strip(".,;:!?\"'").lower() for w in str(text or "").split()]
        words = [w for w in words if w]
        fill = {"not", "but", "rather", "than", "for", "and", "yet", "nor"}
        n = len(words)
        for k in range(3, n - 2):
            left, right = words[:k], words[k:]
            right = [w for w in right if w not in fill]
            left = [w for w in left if w not in fill]
            if len(left) >= 3 and len(right) >= 3 and right[::-1] == left:
                return " ".join(words)
            if len(left) >= 3 and right[::-1][:len(left)] == left and len(right) >= len(left):
                return " ".join(words)
        return ""
    except Exception:
        return ""


def _struct_openings(text):
    """3+ sentences sharing an opening 3-gram (anaphora signal)."""
    try:
        import re as _re2
        sents = [_re2.split(r"\s+", s.strip().lower())[:3]
                 for s in _re2.split(r"[.!?]+", str(text or "")) if s.strip()]
        seen = {}
        for s in sents:
            if len(s) == 3:
                k = tuple(s)
                seen[k] = seen.get(k, 0) + 1
        for k, n in seen.items():
            if n >= 3:
                return " ".join(k)
        return ""
    except Exception:
        return ""


def detect(text, conn=None, max_hits=10):
    """Scan text against DB patterns. Returns [(fal_id, span, conf, rule_id)]."""
    out = []
    try:
        t = str(text or "")
        if not t.strip():
            return out
        for rule_id, rx, target in load_patterns(conn):
            try:
                if isinstance(rx, tuple) and rx[0] == "STRUCT":
                    kind = rx[1]
                    if kind == "abba-reversal":
                        span = _struct_abba(t)
                    elif kind == "repeated-openings":
                        span = _struct_openings(t)
                    else:
                        continue
                    if span:
                        out.append((target, span, 0.6, rule_id))
                else:
                    m = rx.search(t)
                    if m:
                        out.append((target, m.group(0)[:120], 0.7, rule_id))
                if len(out) >= max_hits:
                    break
            except Exception:
                continue
    except Exception:
        pass
    return out


def describe(conn, fal_id):
    """Fallacy record for a finding: display + description + detection note."""
    try:
        own = False
        if conn is None:
            from core import db as _db
            conn = _db.db_connect("trivium")
            own = True
        try:
            row = conn.execute("SELECT display_name, description FROM entities"
                               " WHERE canonical_id=?", (fal_id,)).fetchone()
            if not row:
                return None
            return {"display_name": row[0], "description": row[1] or ""}
        finally:
            if own:
                try:
                    conn.close()
                except Exception:
                    pass
    except Exception:
        return None
