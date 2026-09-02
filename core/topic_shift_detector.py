"""
Semantic topic shift detector for chat retrieval.

Uses embedding similarity and entity/keyword changes to determine when
the user has moved to a new topic. Produces an augmented retrieval query
that includes active topic terms only when the current query is a follow-up.
"""

import numpy as np
import config
import logging
logger = logging.getLogger(__name__)


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
            logger.warning("Unexpected exception occurred", exc_info=True)
            pass

        def _embed(self, text):
            if not text:
                return None
            try:
                if self.embedder is not None:
                    vecs = self.embedder.encode([text])
                    if vecs:
                        from core.hyperbolic import exp_map
                        return exp_map(np.array(vecs[0], dtype=np.float32))
            except Exception:
                logger.warning("Unexpected exception occurred", exc_info=True)
                pass
            try:
                from core.embeddings import get_embedding
                return get_embedding(text, space='hyperbolic')
            except Exception:
                logger.warning("Unexpected exception occurred", exc_info=True)
                return None
