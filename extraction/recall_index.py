"""
Recall index preloaded once per ingestion run for DB-aware priority extraction.

Scans databases during fast pass (no LLM) to match similar topics, dates,
references, events and flag priority — generic, learned from DB content,
no hardcoded doc-specific values. Preserves hyperbolic geometry and
verification-first (priority = must-extract + must-verify, not must-believe).
"""
import re
import config
from core import db
import logging
logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r'\b(17|18|19|20)\d{2}\b')

_index_cache = None


class RecallIndex:
    def __init__(self):
        self.standards = []  # [{statement, negation, confidence, embedding}]
        self.topic_centroids = []  # [(cluster_id, centroid np array)]
        self.alias_map = {}  # lower -> canonical (first wins, backward compat)
        self.alias_ambiguity = {}  # lower -> list[canon] (cap 5, preserves ambiguity)
        self.alias_automaton = None
        self.date_anchors = set()  # normalized date strings + years
        self.event_triggers = set()
        self.open_contradictions = []  # [{fact_id, statement, embedding}]
        self.logic_keywords = set()
        self.memory_keywords = set()

    @classmethod
    def load(cls, force=False):
        global _index_cache
        if _index_cache is not None and not force:
            return _index_cache
        idx = cls()
        try:
            idx._load_standards()
        except Exception as e:
            logger.warning(f"RecallIndex standards load failed: {e}", exc_info=True)
        try:
            idx._load_topics()
        except Exception as e:
            logger.warning(f"RecallIndex topics load failed: {e}", exc_info=True)
        try:
            idx._load_aliases()
        except Exception as e:
            logger.warning(f"RecallIndex aliases load failed: {e}", exc_info=True)
        try:
            idx._load_dates()
        except Exception as e:
            logger.warning(f"RecallIndex dates load failed: {e}", exc_info=True)
        try:
            idx._load_triggers()
        except Exception as e:
            logger.warning(f"RecallIndex triggers load failed: {e}", exc_info=True)
        try:
            idx._load_contradictions()
        except Exception as e:
            logger.warning(f"RecallIndex contradictions load failed: {e}", exc_info=True)
        try:
            idx._load_logic_memory()
        except Exception:
            pass
        _index_cache = idx
        return idx

    def _load_standards(self):
        conn = db.db_connect("verification_standards")
        try:
            cur = conn.cursor()
            cur.execute("SELECT statement, negation, confidence FROM verified_standards WHERE truth_status IN ('admin_claim','verified_true') LIMIT 2000")
            rows = cur.fetchall()
        finally:
            conn.close()
        if not rows:
            return
        statements = [r["statement"] for r in rows if r["statement"]]
        if not statements:
            return
        from core.embeddings import get_embeddings_batch
        embs = get_embeddings_batch(statements, space='hyperbolic', model=getattr(config, "EMBEDDING_MODEL", None))
        for r, emb in zip(rows, embs):
            if emb is not None:
                self.standards.append({"statement": r["statement"], "negation": r["negation"] or 0, "confidence": r["confidence"] or 1.0, "embedding": emb})

    def _load_topics(self):
        try:
            from core.hyperbolic_topic_index import load_topic_index
            centroids, _ = load_topic_index()
            import numpy as _np
            for cid, c in centroids:
                try:
                    self.topic_centroids.append((cid, _np.asarray(c, dtype=_np.float32)))
                except Exception:
                    continue
        except Exception:
            pass
        if not self.topic_centroids:
            # Fallback: degree-top global nodes as pseudo-centroids is handled lazily by caller
            pass

    def _load_aliases(self):
        conn = db.db_connect("external_graph")
        try:
            cur = conn.cursor()
            try:
                cur.execute("SELECT canonical_name, aliases_json FROM global_nodes ORDER BY rowid DESC LIMIT 20000")
                rows = cur.fetchall()
            except Exception:
                rows = []
        finally:
            try:
                conn.close()
            except Exception:
                pass
        import json as _json
        # Ambiguity-preserving map: term_lower -> list[canon] (cap 5, insertion order).
        # Single-canon alias_map kept for backward compat (first canon wins).
        self.alias_ambiguity = {}
        for r in rows:
            try:
                canon = r["canonical_name"]
                if canon:
                    self.alias_map.setdefault(canon.lower(), canon)
                    _lst = self.alias_ambiguity.setdefault(canon.lower(), [])
                    if canon not in _lst and len(_lst) < 5:
                        _lst.append(canon)
                aj = r["aliases_json"] if "aliases_json" in r.keys() else None
                if aj:
                    try:
                        aliases = _json.loads(aj) if isinstance(aj, str) else aj
                        if isinstance(aliases, list):
                            for a in aliases:
                                if isinstance(a, str) and a:
                                    self.alias_map.setdefault(a.lower(), canon)
                                    _lst = self.alias_ambiguity.setdefault(a.lower(), [])
                                    if canon not in _lst and len(_lst) < 5:
                                        _lst.append(canon)
                    except Exception:
                        pass
            except Exception:
                continue
        # Build Aho-Corasick with ambiguity lists as values (generic, capped)
        try:
            import ahocorasick
            A = ahocorasick.Automaton()
            for term_lower, canons in list(self.alias_ambiguity.items())[:20000]:
                A.add_word(term_lower, (list(canons), term_lower))
            A.make_automaton()
            self.alias_automaton = A
        except Exception:
            self.alias_automaton = None

    def _load_dates(self):
        # Years + normalized dates from standards + key_facts dates (generic, no hardcoding)
        for s in self.standards:
            try:
                for m in _YEAR_RE.finditer(s["statement"]):
                    self.date_anchors.add(m.group())
            except Exception:
                continue
        try:
            conn = db.db_connect("key_facts")
            cur = conn.cursor()
            try:
                cur.execute("SELECT normalized_date, date_text FROM dates LIMIT 5000")
                for r in cur.fetchall():
                    for k in ("normalized_date", "date_text"):
                        try:
                            v = r[k]
                            if v:
                                self.date_anchors.add(str(v).strip())
                                for m in _YEAR_RE.finditer(str(v)):
                                    self.date_anchors.add(m.group())
                        except Exception:
                            continue
            except Exception:
                pass
            conn.close()
        except Exception:
            pass

    def _load_triggers(self):
        try:
            from extraction.rule_annotator import load_gazetteers
            gaz = load_gazetteers()
            self.event_triggers = set(gaz.get("event_triggers", set()))
        except Exception:
            self.event_triggers = set()
        # Augment with distinct event_types from DB (learned, not hardcoded)
        try:
            conn = db.db_connect("key_facts")
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT event_type FROM events LIMIT 100")
            for r in cur.fetchall():
                try:
                    v = r["event_type"]
                    if v:
                        self.event_triggers.add(str(v).lower())
                except Exception:
                    continue
            conn.close()
        except Exception:
            pass

    def _load_contradictions(self):
        # Open review queue: unresolved contradictions (generic table probe, no hardcoding)
        for table, col in (("contradiction_log", "fact_text"), ("review_queue", "statement")):
            try:
                conn = db.db_connect("reasoning") if "contradiction" in table else db.db_connect("verification_standards")
                cur = conn.cursor()
                try:
                    cur.execute(f"SELECT {col} FROM {table} LIMIT 200")
                    rows = cur.fetchall()
                except Exception:
                    rows = []
                conn.close()
                for r in rows:
                    try:
                        s = r[col] if col in r.keys() else list(r)[0]
                        if s:
                            self.open_contradictions.append({"statement": str(s)})
                    except Exception:
                        continue
            except Exception:
                continue
        if self.open_contradictions:
            try:
                from core.embeddings import get_embeddings_batch
                texts = [c["statement"] for c in self.open_contradictions]
                embs = get_embeddings_batch(texts, space='hyperbolic')
                for c, e in zip(self.open_contradictions, embs):
                    c["embedding"] = e
            except Exception:
                pass

    def _load_logic_memory(self):
        try:
            conn = db.db_connect("logic")
            cur = conn.cursor()
            cur.execute("SELECT keyword FROM logic_keywords LIMIT 2000")
            for r in cur.fetchall():
                try:
                    self.logic_keywords.add(str(r["keyword"]).lower())
                except Exception:
                    continue
            conn.close()
        except Exception:
            pass
        try:
            conn = db.db_connect("memories")
            cur = conn.cursor()
            cur.execute("SELECT keyword FROM memory_keywords LIMIT 2000")
            for r in cur.fetchall():
                try:
                    self.memory_keywords.add(str(r["keyword"]).lower())
                except Exception:
                    continue
            conn.close()
        except Exception:
            pass
