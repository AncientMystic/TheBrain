"""ONNX lock: concurrent Run calls never overlap (heap-corruption guard)."""
import threading
import time
from core.onnx_lock import run


class _FakeSession:
    def __init__(self):
        self.intervals = []
        self.guard = threading.Lock()

    def run(self, names, inputs):
        t0 = time.perf_counter()
        time.sleep(0.01)
        t1 = time.perf_counter()
        with self.guard:
            self.intervals.append((t0, t1))
        return [inputs]


def test_run_never_overlaps():
    sess = _FakeSession()
    ts = []

    def _worker(i):
        run(sess, None, i)
        ts.append(i)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert sorted(ts) == list(range(8))
    ordered = sorted(sess.intervals)
    for (_, e0), (s1, _) in zip(ordered, ordered[1:]):
        assert s1 >= e0, "overlapping ONNX Run intervals: lock is broken"
