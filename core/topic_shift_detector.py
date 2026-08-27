"""
Semantic topic shift detector for chat retrieval.

Uses embedding similarity and entity/keyword changes to determine when
the user has moved to a new topic. Produces an augmented retrieval query
that includes active topic terms only when the current query is a follow-up.
"""

import numpy as np
import config


try:
    from core.topic_shift_model import TopicShiftModelDetector
    _model_detector = TopicShiftModelDetector()
except Exception:
    _model_detector = None

class TopicShiftDetector:
    def __init__(self, similarity_threshold=None):
        self.similarity_threshold = similarity_threshold or getattr(config, "TOPIC_SHIFT_SIMILARITY", 0.35)
        self.active_terms = []
        self.active_centroid = None
        self.active_history = []  # recent query texts
        self.max_history = 5

        # Load local embedder if available, else fall back to LM Studio embeddings
        self.embedder = None
        try:
            from core.local_embedder import get_local_embedder
            local = get_local_embedder()
            if local.available:
                self.embedder = local
        except Exception:
            pass

    def _embed(self, text):
        """Return embedding vector from local embedder or LM Studio."""
        if not text:
            return None
        try:
            if self.embedder is not None:
                vecs = self.embedder.encode([text])
                if vecs:
                    return vecs[0]
        except Exception:
            pass
        try:
            from core.embeddings import get_embedding
            return get_embedding(text)
        except Exception:
            return None

    def _cosine(self, a, b):
        if a is None or b is None:
            return 0.0
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    def _extract_topic_terms(self, text):
        from chat.query_analyzer import extract_topic_terms
        return extract_topic_terms(text)

    def is_new_topic(self, query):
        """Return True if query likely starts a new topic."""
        if _model_detector is not None and getattr(_model_detector, 'model', None) is not None:
            return _model_detector.is_new_topic(query, self.active_history)
        q_emb = self._embed(query)
        if q_emb is None:
            # Fallback: if no active terms or no overlap, treat as new
            if not self.active_terms:
                return False
            q_terms = set(self._extract_topic_terms(query))
            overlap = len(q_terms & set(self.active_terms))
            return overlap == 0

        if self.active_centroid is None:
            return False

        sim = self._cosine(q_emb, self.active_centroid)
        if sim < self.similarity_threshold:
            return True

        # Entity/keyword shift check
        new_entities = set(self._extract_topic_terms(query))
        if new_entities:
            known = set(self.active_terms)
            if not new_entities & known:
                return True
        return False

    def update(self, query, answer=""):
        """Update active topic state after assistant response."""
        q_terms = self._extract_topic_terms(query)
        if q_terms:
            self.active_terms = q_terms[:10]

        q_emb = self._embed(query)
        if q_emb is not None:
            if self.active_centroid is None:
                self.active_centroid = q_emb
            else:
                # Exponential moving centroid
                alpha = 0.6
                self.active_centroid = (
                    alpha * np.array(self.active_centroid, dtype=np.float32)
                    + (1 - alpha) * np.array(q_emb, dtype=np.float32)
                ).tolist()

        self.active_history.append(query)
        if len(self.active_history) > self.max_history:
            self.active_history.pop(0)

    def get_retrieval_query(self, query):
        """
        Return query augmented with active terms if follow-up,
        or just original query if new topic.
        """
        if self.is_new_topic(query):
            # Start fresh; no augmentation
            return query
        # Add active terms if available
        if self.active_terms:
            extra = " ".join(self.active_terms)
            return f"{query} {extra}"
        return query


_shift_detector = None


def get_topic_shift_detector():
    global _shift_detector
    if _shift_detector is None:
        _shift_detector = TopicShiftDetector()
    return _shift_detector
