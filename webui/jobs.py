"""
Job registry with SSE streaming (mocked guided run for skeleton batch).

Real workers (process_file/recoll/audit) plug into _run_guided in later batches;
skeleton streams synthetic log/progress/document events so UI terminal + bars
work end-to-end with no backend dependency. Bounded queues, cancel tokens.
"""
import itertools
import queue
import threading
import time
import uuid

_jobs = {}
_lock = threading.Lock()
_seq = itertools.count(1)


def create_job(kind, params=None):
    jid = f"job-{uuid.uuid4().hex[:8]}"
    q = queue.Queue(maxsize=1000)
    cancel = threading.Event()
    job = {"id": jid, "kind": kind, "params": params or {}, "queue": q,
           "cancel": cancel, "done": False, "thread": None}
    with _lock:
        _jobs[jid] = job
    t = threading.Thread(target=_run_mock, args=(job,), daemon=True)
    job["thread"] = t
    t.start()
    return jid


def get_job(jid):
    with _lock:
        return _jobs.get(jid)


def cancel_job(jid):
    job = get_job(jid)
    if job:
        job["cancel"].set()
        return True
    return False


def _emit(job, event):
    try:
        job["queue"].put_nowait(event)
    except queue.Full:
        # Drop debug first, never error/done
        if event.get("type") not in ("error", "done"):
            try:
                job["queue"].get_nowait()
                job["queue"].put_nowait(event)
            except Exception:
                pass


def _run_mock(job):
    # Synthetic guided run: 5 fake documents, 0.3s each, cancellable
    docs = [f"doc-{i}.pdf" for i in range(1, 6)]
    total = len(docs)
    _emit(job, {"type": "log", "level": "info", "msg": f"Starting {job['kind']} (mocked skeleton batch)"})
    for i, d in enumerate(docs, 1):
        if job["cancel"].is_set():
            _emit(job, {"type": "log", "level": "warn", "msg": "Cancelled by user"})
            break
        _emit(job, {"type": "log", "level": "info", "msg": f"Processing {d} ({i}/{total})"})
        time.sleep(0.3)
        _emit(job, {"type": "document", "name": d, "chunks": 12, "facts": 8 + i, "status": "done"})
        _emit(job, {"type": "progress", "done": i, "total": total})
    _emit(job, {"type": "done", "ok": not job["cancel"].is_set()})
    job["done"] = True
