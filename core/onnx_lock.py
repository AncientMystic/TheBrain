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
