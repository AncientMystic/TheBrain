"""
Cross-encoder reranker using HuggingFace Transformers.
Falls back to no-op if dependencies are unavailable.
"""
import numpy as np
import config
from pathlib import Path

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None


class Reranker:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.available = False
        if torch and getattr(config, "RERANKER_ENABLED", True):
            self._load_model()

    def _load_model(self):
        model_dir = Path(config.RERANKER_MODEL_DIR)
        if not model_dir.exists():
            print("Reranker model directory not found.")
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
            self.model.eval()
            if torch.cuda.is_available():
                self.model.to("cuda")
            self.available = True
            print("Reranker loaded successfully.")
        except Exception as e:
            print(f"Failed to load reranker: {e}")

    def score(self, query, texts):
        """Return list of relevance scores in [0,1]."""
        if not self.available or not texts:
            return [0.0] * len(texts)

        try:
            pairs = [(query, t) for t in texts]
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits
                scores = torch.sigmoid(logits).cpu().numpy().flatten().tolist()
            return [float(s) for s in scores]
        except Exception as e:
            if config.DEBUG_VERBOSE:
                print(f"Reranker score error: {e}")
            return [0.0] * len(texts)


_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
