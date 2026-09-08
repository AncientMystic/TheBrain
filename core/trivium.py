"""Trivium stage awareness (phase 86): intrinsic reasoning-stage layer.

classify_stage(text) maps input to grammar/logic/rhetoric by weighted
signal scoring (deterministic, no LLM). trivium_context_block() builds a
bounded one-line orientation + stage guidance pulled live from trivium.db.
Advisory-only: never raises, never blocks.
"""
import re

_WORD_RE = re.compile(r"[a-z0-9]+")

# signal phrase -> (stage, weight). Tuned for precision over recall.
SIGNALS = [
    # grammar: fact-seeking
    ("what is", ("grammar", 2)), ("what are", ("grammar", 2)),
    ("who is", ("grammar", 2)), ("who was", ("grammar", 2)),
    ("where is", ("grammar", 2)), ("when did", ("grammar", 2)),
    ("define", ("grammar", 3)), ("definition", ("grammar", 2)),
    ("list", ("grammar", 2)), ("name", ("grammar", 1)),
    ("meaning of", ("grammar", 2)), ("what does", ("grammar", 2)),
    # logic: relation-seeking
    ("why", ("logic", 3)), ("how does", ("logic", 2)),
    ("how do", ("logic", 2)), ("compare", ("logic", 3)),
    ("difference between", ("logic", 3)), ("because", ("logic", 1)),
    ("evidence", ("logic", 2)), ("fallacy", ("logic", 3)),
    ("valid", ("logic", 1)), ("contradict", ("logic", 2)),
    ("assumption", ("logic", 2)), ("implies", ("logic", 2)),
    ("cause", ("logic", 1)), ("therefore", ("logic", 1)),
    # rhetoric: expression-seeking
    ("persuade", ("rhetoric", 3)), ("convince", ("rhetoric", 3)),
    ("write", ("rhetoric", 2)), ("essay", ("rhetoric", 2)),
    ("speech", ("rhetoric", 2)), ("present", ("rhetoric", 1)),
    ("argue", ("rhetoric", 2)), ("debate", ("rhetoric", 2)),
    ("teach", ("rhetoric", 2)), ("explain to", ("rhetoric", 2)),
    ("should i", ("rhetoric", 2)), ("opinion", ("rhetoric", 1)),
]

STAGE_CANONICAL = ("grammar", "logic", "rhetoric")
STAGE_QUESTIONS = {
    "grammar": "What? Who? Where? When?",
    "logic": "Why? How? What relates to what?",
    "rhetoric": "How to express and apply this well?",
}


def classify_stage(text):
    """Return (stage, confidence, signals). Confidence = winner share of
    total signal weight, 0.0 when nothing fires (defaults to grammar)."""
    try:
        t = str(text or "").lower()
        scores = {"grammar": 0, "logic": 0, "rhetoric": 0}
        hits = []
        for phrase, (stage, w) in SIGNALS:
            if phrase in t:
                scores[stage] += w
                hits.append(phrase)
        total = sum(scores.values())
        if total <= 0:
            return "grammar", 0.0, []
        best = max(STAGE_CANONICAL, key=lambda s: scores[s])
        # deterministic tie-break: grammar < logic < rhetoric order above
        return best, round(scores[best] / total, 3), hits
    except Exception:
        return "grammar", 0.0, []


def lookup_stage(stage, conn=None):
    """Stage guidance record from trivium.db (description of the art entity)."""
    try:
        if stage not in STAGE_CANONICAL:
            return None
        own = False
        if conn is None:
            from core import db as _db
            conn = _db.db_connect("trivium")
            own = True
        try:
            row = conn.execute("SELECT display_name, description FROM entities"
                               " WHERE canonical_id=?", (f"trivium:art:{stage}",)).fetchone()
            if not row:
                return None
            return {"display_name": row[0], "description": row[1] or "",
                    "question": STAGE_QUESTIONS[stage]}
        finally:
            if own:
                try:
                    conn.close()
                except Exception:
                    pass
    except Exception:
        return None


def trivium_context_block(query_text, max_chars=600, conn=None):
    """One-line stage orientation + art guidance. Bounded, read-only."""
    try:
        stage, conf, _ = classify_stage(query_text)
        card = lookup_stage(stage, conn=conn)
        if not card:
            return ""
        line = (f"Trivium: processing at {stage.upper()} stage "
                f"(confidence {conf:.2f}; {card['question']}) — "
                f"{card['description'][:300]}")
        return line[:int(max_chars)]
    except Exception:
        return ""


def trivium_overview(conn=None):
    """Dashboard rollup: row count, arts, stages, shards. Read-only dict."""
    out = {"entities": 0, "arts": 0, "stages": 0, "shards": 0}
    try:
        own = False
        if conn is None:
            from core import db as _db
            conn = _db.db_connect("trivium")
            own = True
        try:
            out["entities"] = int(conn.execute(
                "SELECT COUNT(*) FROM entities").fetchone()[0])
            out["arts"] = int(conn.execute(
                "SELECT COUNT(*) FROM entities WHERE entity_type='art'").fetchone()[0])
            out["stages"] = int(conn.execute(
                "SELECT COUNT(*) FROM entities WHERE entity_type='stage'").fetchone()[0])
            out["shards"] = int(conn.execute(
                "SELECT COUNT(DISTINCT shard_key) FROM entities").fetchone()[0])
        finally:
            if own:
                try:
                    conn.close()
                except Exception:
                    pass
    except Exception:
        pass
    return out
