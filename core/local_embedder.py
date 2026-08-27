"""
Local sentence embedder using HuggingFace Transformers.
Falls back to no-op if dependencies are unavailable.
"""
import numpy as np
import torch
import config
from pathlib import Path

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    AutoTokenizer = None
    AutoModel = None


class LocalEmbedder:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.available = False
        if getattr(config, "LOCAL_EMBEDDER_ENABLED", True):
            self._load_model()

    def _load_model(self):
        model_dir = Path(config.LOCAL_EMBEDDER_MODEL_DIR)
        if not model_dir.exists():
            print("Local embedder model directory not found.")
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            self.model = AutoModel.from_pretrained(str(model_dir))
            self.model.eval()
            if torch.cuda.is_available():
                self.model.to("cuda")
            self.available = True
            print("Local embedder loaded successfully.")
        except Exception as e:
            print(f"Failed to load local embedder: {e}")

    def encode(self, texts):
        """Return list of embedding vectors."""
        if not self.available or not texts:
            return []
        try:
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"]
            masked = hidden * mask.unsqueeze(-1)
            summed = masked.sum(dim=1)
            counts = mask.sum(dim=1, keepdims=True)
            mean = summed / counts
            return mean.cpu().numpy().tolist()
        except Exception as e:
            if config.DEBUG_VERBOSE:
                print(f"Local embedder encode error: {e}")
            return []


_local_embedder = None


def get_local_embedder():
    global _local_embedder
    if _local_embedder is None:
        _local_embedder = LocalEmbedder()
    return _local_embedder
