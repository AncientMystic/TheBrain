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
    def _cache_key_parts(cls):
        try:
            import config as _cfgk
            import re as _rek
            _dim = int(getattr(_cfgk, "EMBEDDING_DIM", 1024))
            _model = str(getattr(_cfgk, "EMBEDDING_MODEL", "mxbai"))
            _safe = _rek.sub(r'[^A-Za-z0-9._-]+', '_', _model)[:60]
            return _dim, _safe
        except Exception:
            return 1024, "mxbai"

    @classmethod
    def _cache_path(cls):
        try:
            from pathlib import Path as _P
            import config as _cfg
            _dim, _safe = cls._cache_key_parts()
            return str(_P(_cfg.BASE_DIR) / "data" / f"recall_index_cache_d{_dim}_{_safe}.npz")
        except Exception:
            return ""

    @classmethod
    def _db_fingerprint(cls):
        import os as _os
        _max = 0.0
        try:
            import config as _cfg2
            from core import db as _db2
            for _k in ("verification_standards", "embeddings", "external_graph", "key_facts", "logic", "memories", "reasoning"):
                try:
                    _p = _db2.DB_FILES.get(_k, "")
                    if _p and _os.path.exists(_p):
                        _max = max(_max, _os.path.getmtime(_p))
                except Exception:
                    continue
        except Exception:
            pass
        return _max

    @classmethod
    def load(cls, force=False):
        global _index_cache
        if _index_cache is not None and not force:
            return _index_cache
        # Disk cache when DBs unchanged (minutes saved per run, same data — no quality change)
        try:
            import os as _os3
            _cp = cls._cache_path()
            if _cp and not force and _os3.path.exists(_cp):
                try:
                    import numpy as _np_ld
                    _fp = cls._db_fingerprint()
                    if _os3.path.getmtime(_cp) >= _fp:
                        _d = _np_ld.load(_cp, allow_pickle=True)
                        try:
                            _meta = _d["meta"].item() if "meta" in _d else {}
                            _dim0, _model0 = cls._cache_key_parts()
                            if int(_meta.get("dim", _dim0)) != int(_dim0):
                                raise ValueError("recall cache dim mismatch")
                        except Exception:
                            raise ValueError("recall cache metadata mismatch")
                        idx = cls()
                        try:
                            import config as _cfg_v
                            _exp = int(getattr(_cfg_v, "EMBEDDING_DIM", 1024))
                            _std = list(_d["standards"])
                            _kept = []
                            for s, n, c, e in _std:
                                try:
                                    import numpy as _np_v
                                    _arr = _np_v.asarray(e)
                                    if _arr.shape[-1] != _exp:
                                        continue
                                    _kept.append({"statement": s, "negation": int(n), "confidence": float(c), "embedding": e})
                                except Exception:
                                    continue
                            idx.standards = _kept
                        except Exception:
                            idx.standards = []
                        try:
                            _tc = list(_d["centroids"])
                            import numpy as _np_c
                            import config as _cfg_c
                            _exp2 = int(getattr(_cfg_c, "EMBEDDING_DIM", 1024))
                            _tkept = []
                            for cid, c in _tc:
                                try:
                                    _arr = _np_c.asarray(c, dtype=_np_c.float32)
                                    if _arr.shape[-1] != _exp2:
                                        continue
                                    _tkept.append((int(cid), _arr))
                                except Exception:
                                    continue
                            idx.topic_centroids = _tkept
                        except Exception:
                            idx.topic_centroids = []
                        try:
                            import json as _js
                            idx.alias_map = dict(_d["alias_map"].item()) if "alias_map" in _d else {}
                            _amb = _d["alias_ambiguity"].item() if "alias_ambiguity" in _d else {}
                            idx.alias_ambiguity = {k: list(v)[:5] for k, v in dict(_amb).items()}
                            idx.date_anchors = set(_d["date_anchors"].tolist()) if "date_anchors" in _d else set()
                            idx.event_triggers = set(_d["event_triggers"].tolist()) if "event_triggers" in _d else set()
                            idx.logic_keywords = set(_d["logic_keywords"].tolist()) if "logic_keywords" in _d else set()
                            idx.memory_keywords = set(_d["memory_keywords"].tolist()) if "memory_keywords" in _d else set()
                        except Exception:
                            pass
                        # Rebuild automaton from map (fast, in-memory, no DB)
                        try:
                            import ahocorasick
                            A = ahocorasick.Automaton()
                            for term_lower, canons in list(idx.alias_ambiguity.items())[:20000]:
                                A.add_word(term_lower, (list(canons), term_lower))
                            A.make_automaton()
                            idx.alias_automaton = A
                        except Exception:
                            idx.alias_automaton = None
                        _index_cache = idx
                        return idx
                except Exception:
                    pass
        except Exception:
            pass
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
        # Persist for next run (same data when DBs unchanged — no quality change).
        # Metadata-stamped (dim/model) so renamed/copied files never load silently.
        try:
            import numpy as _np_sv
            _cp = cls._cache_path()
            if _cp:
                _dim0, _model0 = cls._cache_key_parts()
                _meta0 = {"dim": int(_dim0), "model": str(_model0)}
                _std = [(s["statement"], s["negation"], s["confidence"], s["embedding"]) for s in idx.standards if s.get("embedding") is not None]
                _tc = [(int(cid), c) for cid, c in idx.topic_centroids[:20]]
                _np_sv.savez_compressed(_cp,
                    standards=np.array(_std, dtype=object),
                    centroids=np.array(_tc, dtype=object),
                    alias_map=np.array(idx.alias_map, dtype=object),
                    alias_ambiguity=np.array(idx.alias_ambiguity, dtype=object),
                    date_anchors=np.array(sorted(idx.date_anchors)[:5000]),
                    event_triggers=np.array(sorted(idx.event_triggers)[:500]),
                    logic_keywords=np.array(sorted(idx.logic_keywords)[:2000]),
                    memory_keywords=np.array(sorted(idx.memory_keywords)[:2000]),
                    meta=np.array(_meta0, dtype=object))
        except Exception:
            pass
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
