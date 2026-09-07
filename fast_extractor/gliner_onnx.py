"""
GLiNER zero-shot NER over ONNX Runtime (replaces bert-base-NER).

Why: bert-base-NER knows 4 fixed types with no confidence routing; GLiNER
small-v2.1 is zero-shot over any labels with per-span scores in one pass.
Input arrays are built with the official gliner processor (no guesswork);
only inference runs on ORT (serialized via core.onnx_lock like every other
session) and span decoding is dependency-free numpy.
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

# Project label set (superset of the old PER/ORG/LOC/MISC mapping).
DEFAULT_LABELS = ["person", "organization", "location", "date", "event", "product"]
_LABEL_MAP = {"person": "PERSON", "organization": "ORG", "location": "LOC",
              "date": "DATE", "event": "EVENT", "product": "MISC"}


class GlinerONNXExtractor:
    def __init__(self, model_dir=None, labels=None, threshold=None):
        if threshold is None:
            try:
                threshold = float(getattr(config, "GLINER_THRESHOLD", 0.35))
            except Exception:
                threshold = 0.35
        self.session = None
        self.tokenizer = None
        self.processor = None
        self.available = False
        self.labels = labels or list(DEFAULT_LABELS)
        self.threshold = float(threshold)
        self.input_names = None
        if ort:
            self._load_model(model_dir)

    def _load_model(self, model_dir):
        import os as _os
        from transformers import AutoTokenizer
        root = Path(model_dir or getattr(config, "GLINER_MODEL_DIR",
                                         str(Path(config.BASE_DIR) / "models" / "gliner_ner")))
        # Prefer int8-quantized build on weak machines (~175MB over ~583MB fp32).
        cands = [root / "onnx" / "model_quantized.onnx",
                 root / "model_quantized.onnx",
                 root / "onnx" / "model.onnx",
                 root / "model.onnx"]
        onnx_path = next((p for p in cands if p.exists()), None)
        if onnx_path is None:
            print(f"GLiNER onnx not found under {root}")
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(root))
            print("Loaded GLiNER tokenizer via transformers.")
        except Exception as e:
            print(f"GLiNER tokenizer load error: {e}")
            return
        try:
            import json as _json
            from gliner.config import GLiNERConfig
            from gliner.data_processing.processor import UniEncoderSpanProcessor
            from gliner.data_processing.tokenizer import WhitespaceTokenSplitter
            _cfg_path = root / "gliner_config.json"
            _cfg = GLiNERConfig.from_dict(_json.loads(_cfg_path.read_text()))
            self.processor = UniEncoderSpanProcessor(
                _cfg, self.tokenizer, WhitespaceTokenSplitter())
            self.max_width = int(getattr(_cfg, "max_width", 12))
        except Exception as e:
            print(f"GLiNER processor init error: {e}")
            return
        try:
            so = ort.SessionOptions()
            try:
                so.intra_op_num_threads = int(getattr(config, "ONNX_INTRA_THREADS", _os.cpu_count() or 4))
                so.inter_op_num_threads = int(getattr(config, "ONNX_INTER_THREADS", 2))
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                so.log_severity_level = 3
            except Exception:
                pass
            # Same pinning as bert NER above: CPU default so the threaded
            # pre-pass never puts concurrent Run on the DML driver.
            try:
                _dev = str(getattr(config, "FAST_EXTRACTOR_DEVICE", "cpu") or "cpu").lower()
            except Exception:
                _dev = "cpu"
            if _dev in ("directml", "dml", "auto") and "DmlExecutionProvider" in ort.get_available_providers():
                providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            elif _dev == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
            self.input_names = [i.name for i in self.session.get_inputs()]
            self.available = True
            print(f"GLiNER loaded ({onnx_path.name}) with providers {providers}.")
        except Exception as e:
            print(f"Failed to load GLiNER ONNX: {e}")

    def _build_inputs(self, words, labels):
        """Official processor arrays -> numpy (no weights needed for this)."""
        import torch
        tin = self.processor.tokenize_inputs([words], labels)
        from gliner.data_processing.utils import prepare_streaming_span_idx
        span_idx_t, span_mask_t = prepare_streaming_span_idx(
            0, len(words), self.max_width, recompute_all=True)

        def _np(v):
            return v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)

        feed = {}
        for name in self.input_names:
            if name == "span_idx":
                feed[name] = span_idx_t.detach().cpu().numpy()[None, :, :].astype(np.int64)
            elif name == "span_mask":
                feed[name] = span_mask_t.detach().cpu().numpy()[None, :].astype(bool)
            elif name == "text_lengths":
                feed[name] = np.array([[len(words)]], dtype=np.int64)
            elif name in tin:
                feed[name] = _np(tin[name]).astype(np.int64)
        return feed, _np(span_idx_t), _np(span_mask_t)

    def extract_entities(self, text, labels=None, threshold=None):
        """Return list of (entity_text, mapped_type, confidence)."""
        if not self.session or not text or not text.strip():
            return []
        labels = labels or self.labels
        thr = self.threshold if threshold is None else float(threshold)
        words = text.split()
        if not words:
            return []
        try:
            feed, span_idx, span_mask = self._build_inputs(words, labels)
            from core.onnx_lock import run as _ort_run
            outputs = _ort_run(self.session, None, feed)
            logits = np.asarray(outputs[0])
            # Layout (batch, start_word, width, class): measured on the
            # quantized checkpoint, not assumed.
            while logits.ndim > 4:
                logits = logits[0]
            if logits.ndim == 4:
                logits = logits[0]
            probs = 1.0 / (1.0 + np.exp(-logits))
            cands = []
            ns, nw, nc = probs.shape[0], probs.shape[1], probs.shape[2]
            for s in range(min(ns, len(words))):
                for w in range(nw):
                    e = s + w
                    if e >= len(words):
                        break
                    for ci in range(nc):
                        p = float(probs[s][w][ci])
                        if p < thr or ci >= len(labels):
                            continue
                        cands.append((p, s, e, labels[ci]))
            # Greedy flat NER: best first, drop word-overlapping losers.
            cands.sort(reverse=True)
            taken, out = set(), []
            for p, s, e, lab in cands:
                if any(i in taken for i in range(s, e + 1)):
                    continue
                taken.update(range(s, e + 1))
                # Strip word-split punctuation artifacts ("Waitangi," -> "Waitangi").
                txt = " ".join(words[s:e + 1]).strip().strip(".,;:!?\"'()[]")
                if not txt:
                    continue
                out.append((txt, _LABEL_MAP.get(lab, "MISC"), p))
            return out
        except Exception as e:
            if config.DEBUG_VERBOSE:
                print(f"GLiNER extract error: {e}")
            logger.warning("GLiNER extract failed", exc_info=True)
            return []
