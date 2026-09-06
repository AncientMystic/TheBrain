"""Process-wide ONNX Runtime serialization lock.

Root cause of heap corruption (Windows 0xc0000374): two threads inside
`InferenceSession.run` at once — e.g. the local-embedder worker thread
running the embedding model while an ingestion thread runs the NER model.
ORT's docs promise per-session thread-safety, but the DML execution
provider corrupts the heap under concurrent Run (observed: embedder
session + NER session colliding, plus same-session collisions earlier).
The GIL does not help: Run executes native code concurrently.

Rule: EVERY project-owned `session.run` goes through this lock. Tokenizer
work stays outside (parallel-safe); only the native Run serializes.
Cost is milliseconds per call; correctness is non-negotiable.
"""
import threading

_lock = threading.RLock()


def run(session, output_names, inputs):
    """Drop-in serialized replacement for `session.run(output_names, inputs)`."""
    with _lock:
        return session.run(output_names, inputs)


def resolve_providers():
    """One provider policy for every project-owned session.

    CPU by default: DML sessions corrupt the heap under multi-session and
    multi-threaded use (observed 0xc0000374 both concurrent AND isolated),
    and CPU ORT is already fast for these small models — plus CPU unlocks
    the threaded pre-pass. DML/CUDA only on explicit opt-in via
    ONNX_DEVICE=directml|cuda|auto (auto = try DML first, legacy behavior).
    """
    try:
        import onnxruntime as _ort
        _avail = _ort.get_available_providers()
    except Exception:
        return ["CPUExecutionProvider"]
    try:
        import config as _cfg
        _dev = str(getattr(_cfg, "ONNX_DEVICE", "cpu") or "cpu").lower()
    except Exception:
        _dev = "cpu"
    if _dev in ("directml", "dml") and "DmlExecutionProvider" in _avail:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    if _dev == "cuda" and "CUDAExecutionProvider" in _avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if _dev == "auto":
        if "DmlExecutionProvider" in _avail:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
