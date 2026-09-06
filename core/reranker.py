"""
Cross-encoder reranker: ONNX-first, torch fallback, no-op last resort.

Bake-off verdict (identical queries): ONNX fp32 ranks bit-identically to
torch (max abs score diff 0.0000) at 2.8-5.3x lower latency with 0.6s vs
2.1s load; int8-quantized keeps identical rankings with tiny score drift
for weak machines. Torch stays as automatic fallback when ONNX is absent.
"""
import numpy as np
import config
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError:
    ort = None

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
        self.session = None
        self.tokenizer = None
        self.available = False
        self.backend = None
        if getattr(config, "RERANKER_ENABLED", True):
            self._load_model()

    def _load_model(self):
        model_dir = Path(config.RERANKER_MODEL_DIR)
        if not model_dir.exists():
            print("Reranker model directory not found.")
            return
        try:
            from transformers import AutoTokenizer as _AT
            self.tokenizer = _AT.from_pretrained(str(model_dir))
        except Exception as e:
            print(f"Failed to load reranker tokenizer: {e}")
            return
        # ONNX first (exact parity, no torch/CUDA needed).
        if ort and getattr(config, "RERANKER_BACKEND", "onnx") == "onnx":
            _quant = str(getattr(config, "RERANKER_QUANT", "fp32")).lower()
            _cands = []
            if _quant in ("int8", "qint8", "quant"):
                _cands = [model_dir / "onnx" / "model_qint8_avx2.onnx",
                          model_dir / "onnx" / "model_quantized.onnx"]
            _cands += [model_dir / "onnx" / "model.onnx"]
            for _p in _cands:
                if not _p.exists():
                    continue
                try:
                    import os as _os
                    so = ort.SessionOptions()
                    try:
                        so.intra_op_num_threads = int(getattr(config, "ONNX_INTRA_THREADS", _os.cpu_count() or 4))
                        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                        so.log_severity_level = 3
                    except Exception:
                        pass
                    from core.onnx_lock import resolve_providers
                    providers = resolve_providers()
                    self.session = ort.InferenceSession(str(_p), sess_options=so, providers=providers)
                    self.input_names = [i.name for i in self.session.get_inputs()]
                    self.backend = "onnx"
                    self.available = True
                    print(f"Reranker loaded via ONNX ({_p.name}).")
                    return
                except Exception as e:
                    print(f"  (Reranker ONNX {_p.name} failed: {e})")
        # Torch fallback (original path, unchanged semantics).
        if torch and AutoModelForSequenceClassification:
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
                self.model.eval()
                if torch.cuda.is_available():
                    self.model.to("cuda")
                self.backend = "torch"
                self.available = True
                print("Reranker loaded successfully (torch fallback).")
            except Exception as e:
                print(f"Failed to load reranker: {e}")

    def _score_onnx(self, query, texts):
        import numpy as _np
        pairs = [(query, t) for t in texts]
        enc = self.tokenizer(pairs, padding=True, truncation=True,
                             max_length=256, return_tensors="np")
        feed = {}
        for _n in self.input_names:
            if _n in enc:
                feed[_n] = _np.asarray(enc[_n]).astype(_np.int64)
        from core.onnx_lock import run as _ort_run
        logits = _np.asarray(_ort_run(self.session, None, feed)[0]).flatten()
        return [float(1.0 / (1.0 + _np.exp(-v))) for v in logits]

    def score(self, query, texts):
        """Return list of relevance scores in [0,1]."""
        if not self.available or not texts:
            return [0.0] * len(texts)

        try:
            if self.backend == "onnx":
                return self._score_onnx(query, texts)
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
            logger.warning("Unexpected exception occurred", exc_info=True)
            return [0.0] * len(texts)


_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
