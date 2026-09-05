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
    # Real workers when prerequisites exist; else mocked skeleton stream (offline demo/tests)
    target = _run_mock
    if kind == "guided-learning":
        try:
            from pathlib import Path as _P
            _inp = (params or {}).get("input", "")
            if _inp and _P(str(_inp)).expanduser().exists():
                target = _run_guided_real
        except Exception:
            target = _run_mock
    elif kind == "audit":
        target = _run_audit_real
    elif kind == "research":
        target = _run_research_real
    t = threading.Thread(target=target, args=(job,), daemon=True)
    job["thread"] = t
    t.start()
    return jid


def _run_guided_real(job):
    """Sequential real guided-learning with per-file events (no artificial limits).

    Same pipeline as CLI (single extraction via prepare_next_file, full LLM batches,
    verification, graphs). Emits log/document/progress per file; cancel checked
    between files (not mid-file — process_file is atomic per file by design).
    """
    from pathlib import Path as _P
    params = job.get("params", {})
    raw_inp = str(params.get("input", ""))
    try:
        limit = params.get("limit")
        limit = int(limit) if limit not in (None, "") else None
    except Exception:
        limit = None
    dry = bool(params.get("dry"))
    logic_mode = bool(params.get("logic"))
    verified_flag = bool(params.get("verified"))
    _emit(job, {"type": "log", "level": "info", "msg": f"Starting guided-learning on {raw_inp or '(no input)'}"})
    # Resolve + allow-list (same rules as CLI --input)
    try:
        inp = str(_P(raw_inp).expanduser().resolve())
    except Exception as e:
        _emit(job, {"type": "error", "msg": f"Invalid input path: {e}"})
        _emit(job, {"type": "done", "ok": False})
        job["done"] = True
        return
    import os as _os
    _allowed = [r.strip() for r in _os.environ.get("THEBRAIN_ALLOWED_ROOTS", "").split(",") if r.strip()]
    if _allowed:
        try:
            _ar = [str(_P(r).expanduser().resolve()) for r in _allowed]
            if not any(inp.startswith(a) for a in _ar) and not params.get("allow_outside_root"):
                _emit(job, {"type": "error", "msg": f"Input outside THEBRAIN_ALLOWED_ROOTS (tick allow-outside-root to override)"})
                _emit(job, {"type": "done", "ok": False})
                job["done"] = True
                return
        except Exception:
            pass
    # Scan files (reuse CLI scanner; fallback to rglob)
    try:
        from ingestion.scanner import scan_files
        files = scan_files(inp)
    except Exception:
        try:
            files = [p for p in _P(inp).rglob("*") if p.is_file()]
        except Exception as e:
            _emit(job, {"type": "error", "msg": f"Scan failed: {e}"})
            _emit(job, {"type": "done", "ok": False})
            job["done"] = True
            return
    if limit is not None:
        files = files[:limit]
    total = len(files)
    if dry:
        for f in files:
            _emit(job, {"type": "log", "level": "info", "msg": f"[DRY-RUN] Would process: {getattr(f, 'name', f)}"})
        _emit(job, {"type": "done", "ok": True})
        job["done"] = True
        return
    if not files:
        _emit(job, {"type": "log", "level": "warn", "msg": "No files found"})
        _emit(job, {"type": "done", "ok": True})
        job["done"] = True
        return
    # Lazy imports (avoid circular main<->webui at module load)
    try:
        from main import process_file, prepare_next_file, promote_verified_file
        from core.progress import ProgressTracker
        from core.file_utils import get_file_hash
        from logic import decide_logic_modules
        from core import db as _db
    except Exception as e:
        _emit(job, {"type": "error", "msg": f"Import failed: {e}"})
        _emit(job, {"type": "done", "ok": False})
        job["done"] = True
        return
    tracker = ProgressTracker()
    tracker.total_files = total
    tracker.processed_count = 0
    done = 0
    for f in files:
        if job["cancel"].is_set():
            _emit(job, {"type": "log", "level": "warn", "msg": "Cancelled by user (finishing current file safely)"})
            break
        fname = getattr(f, "name", str(f))
        _emit(job, {"type": "log", "level": "info", "msg": f"Processing {fname}"})
        try:
            _prep = prepare_next_file(f)
        except Exception as e:
            _emit(job, {"type": "log", "level": "error", "msg": f"Prepare failed for {fname}: {e}"})
            _prep = None
        logic_context = ""
        if logic_mode and _prep and _prep.get("text"):
            try:
                _ft = _prep["text"][:1000]
                _ids = decide_logic_modules(_ft, context=_ft)
                if _ids:
                    _conn = _db.db_connect("logic")
                    _cur = _conn.cursor()
                    for _lid in _ids:
                        _cur.execute("SELECT name, category, summary, content FROM logic_modules WHERE logic_id=?", (_lid,))
                        _row = _cur.fetchone()
                        if _row:
                            logic_context += f"[Logic: {_row[0]} ({_row[1]})]\n{_row[2]}\n{_row[3]}\n\n"
                    _conn.close()
            except Exception as e:
                _emit(job, {"type": "log", "level": "warn", "msg": f"Logic decision failed: {e}"})
        try:
            ok = process_file(f, tracker, logic_context=logic_context, preloaded=_prep)
        except Exception as e:
            _emit(job, {"type": "log", "level": "error", "msg": f"Failed {fname}: {e}"})
            ok = False
        # Counts for this file (cheap COUNTs, same DBs the pipeline just wrote)
        facts_n, chunks_n = -1, -1
        try:
            _fh = get_file_hash(f)
            _c = _db.db_connect("key_facts")
            _cur = _c.cursor()
            _cur.execute("SELECT COUNT(*) AS n FROM key_facts WHERE doc_hash=?", (_fh,))
            _row = _cur.fetchone()
            facts_n = int(_row["n"]) if _row else -1
            _c.close()
            _ci = _db.db_connect("index")
            _curi = _ci.cursor()
            _curi.execute("SELECT COUNT(*) AS n FROM document_chunks WHERE doc_hash=?", (_fh,))
            _rowi = _curi.fetchone()
            chunks_n = int(_rowi["n"]) if _rowi else -1
            _ci.close()
            if verified_flag and ok:
                promote_verified_file(_fh, fname, source_file=f)
        except Exception:
            pass
        tracker.processed_count += 1
        done += 1
        _emit(job, {"type": "document", "name": fname, "chunks": chunks_n, "facts": facts_n,
                    "status": "done" if ok else "failed"})
        _emit(job, {"type": "progress", "done": done, "total": total})
    _emit(job, {"type": "done", "ok": not job["cancel"].is_set()})
    job["done"] = True


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


def _run_audit_real(job):
    """Run audit_all as a job with start/finish events (same function as CLI --audit).

    audit_all is atomic (not mid-run cancellable by design); cancel checked before start.
    Emits log + progress 0/1 + done. Never auto-deletes admin/verified (enforced inside audit).
    """
    if job["cancel"].is_set():
        _emit(job, {"type": "log", "level": "warn", "msg": "Cancelled before audit started"})
        _emit(job, {"type": "done", "ok": False})
        job["done"] = True
        return
    _emit(job, {"type": "log", "level": "info", "msg": "Starting audit (same checks as --audit)"})
    _emit(job, {"type": "progress", "done": 0, "total": 1})
    try:
        from audit.auditor import audit_all
        audit_all()
        _emit(job, {"type": "log", "level": "info", "msg": "Audit finished — see Review table below"})
    except Exception as e:
        _emit(job, {"type": "log", "level": "error", "msg": f"Audit failed: {e}"})
        _emit(job, {"type": "done", "ok": False})
        job["done"] = True
        return
    _emit(job, {"type": "progress", "done": 1, "total": 1})
    _emit(job, {"type": "done", "ok": True})
    job["done"] = True


def _run_research_real(job):
    """Run deep-research coordinator as a job (same function as CLI/chat deep path).

    Long-running (minutes); cancel checked before start (coordinator itself is
    atomic per subtopic batch — safe to let current batch finish, then stop).
    Emits log + progress (indeterminate, pulsed) + done with report path.
    """
    params = job.get("params", {})
    query = str(params.get("query", "")).strip()[:500]
    session_id = params.get("session_id")
    if not query:
        _emit(job, {"type": "error", "msg": "Empty research query"})
        _emit(job, {"type": "done", "ok": False})
        job["done"] = True
        return
    if job["cancel"].is_set():
        _emit(job, {"type": "log", "level": "warn", "msg": "Cancelled before research started"})
        _emit(job, {"type": "done", "ok": False})
        job["done"] = True
        return
    _emit(job, {"type": "log", "level": "info", "msg": f"Deep research started: {query[:120]}"})
    _emit(job, {"type": "progress", "done": 0, "total": 1})
    try:
        from deep_research.coordinator import DeepResearchCoordinator
        coordinator = DeepResearchCoordinator(session_id)
        report_path = coordinator.run(query)
        _emit(job, {"type": "log", "level": "info", "msg": f"Report generated: {report_path}"})
        _emit(job, {"type": "report", "path": str(report_path)})
    except Exception as e:
        _emit(job, {"type": "log", "level": "error", "msg": f"Research failed: {e}"})
        _emit(job, {"type": "done", "ok": False})
        job["done"] = True
        return
    _emit(job, {"type": "progress", "done": 1, "total": 1})
    _emit(job, {"type": "done", "ok": True})
    job["done"] = True


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
