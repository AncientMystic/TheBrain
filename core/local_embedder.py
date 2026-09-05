
"""
Robust local embedder using ONNX Runtime and transformers tokenizer.
Handles dynamic input shapes and provides clear error messages.
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

class LocalEmbedder:
    def __init__(self):
        self.session = None
        self.tokenizer = None
        self.available = False
        self.input_names = None
        if ort:
            self._load_model()

    def _load_model(self):
        model_dir = Path(config.LOCAL_EMBEDDER_MODEL_DIR)
        if not model_dir.exists():
            print(f"Local embedder model directory not found: {model_dir}")
            return

        onnx_path = model_dir / "model.onnx"
        if not onnx_path.exists():
            print(f"model.onnx not found in {model_dir}")
            return

        try:
            # Load tokenizer
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                print("Loaded tokenizer via transformers.")
            except Exception:
                from tokenizers import Tokenizer
                tokenizer_path = model_dir / "tokenizer.json"
                if tokenizer_path.exists():
                    self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
                    print("Loaded tokenizer via tokenizers.")
                else:
                    print("No tokenizer.json found; cannot tokenize.")
                    return

            # Create ONNX Runtime session (threads/arena/ALL, same as NER — no quality change)
            import os as _os_le
            so = ort.SessionOptions()
            try:
                so.intra_op_num_threads = int(getattr(config, "ONNX_INTRA_THREADS", _os_le.cpu_count() or 4))
                so.inter_op_num_threads = int(getattr(config, "ONNX_INTER_THREADS", 2))
                so.enable_mem_pattern = True
                so.enable_cpu_mem_arena = True
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                so.log_severity_level = 3
            except Exception:
                pass
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"] if "DmlExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.available = True
            print(f"Local embedder loaded via ONNX Runtime with providers {providers}.")
        except Exception as e:
            print(f"Failed to load ONNX embedder: {e}")

    def encode(self, texts):
        if not self.available or not texts:
            return []

        try:
            # Tokenize with padding/truncation
            encodings = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            input_ids = encodings["input_ids"].astype(np.int64)
            attention_mask = encodings["attention_mask"].astype(np.int64)

            # Build inputs dict based on model input names
            onnx_inputs = {}
            if "input_ids" in self.input_names:
                onnx_inputs["input_ids"] = input_ids
            if "attention_mask" in self.input_names:
                onnx_inputs["attention_mask"] = attention_mask
            if "token_type_ids" in self.input_names:
                token_type_ids = encodings.get("token_type_ids", np.zeros_like(input_ids))
                onnx_inputs["token_type_ids"] = token_type_ids.astype(np.int64)

            outputs = self.session.run(None, onnx_inputs)
            hidden = outputs[0]  # shape (batch, seq, hidden)
            if len(hidden.shape) == 3:
                # Mean pooling
                mask = attention_mask[:, :, None]
                summed = (hidden * mask).sum(axis=1)
                counts = mask.sum(axis=1)
                embeddings = summed / counts
            else:
                embeddings = hidden  # already pooled
            return embeddings.tolist()
        except Exception as e:
            if config.DEBUG_VERBOSE:
                print(f"ONNX embedder encode error: {e}")
            logger.warning("Unexpected exception occurred", exc_info=True)
            return []

_local_embedder = None

def get_local_embedder():
    global _local_embedder
    if _local_embedder is None:
        _local_embedder = LocalEmbedder()
    return _local_embedder
