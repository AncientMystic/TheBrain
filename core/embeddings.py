
import random
from collections import OrderedDict
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

import config
from core.logger import get_logger
logger = get_logger(__name__)
from core import db
from core.backends import create_backend

# Local embedding queue and worker thread (Future-based, no busy-wait)
import threading as _threading
import queue as _queue
from concurrent.futures import Future as _Future

_local_embed_queue = None
_local_embed_worker_started = False
_local_embed_lock = _threading.Lock()

def _start_local_embed_worker():
    global _local_embed_queue, _local_embed_worker_started
    if _local_embed_worker_started:
        return
    with _local_embed_lock:
        if _local_embed_worker_started:
            return
        _local_embed_queue = _queue.Queue(maxsize=getattr(config, "LOCAL_EMBED_QUEUE_SIZE", 100))

        def worker():
            from core.local_embedder import get_local_embedder
            local = get_local_embedder()
            while True:
                job = _local_embed_queue.get()
                if job is None:
                    _local_embed_queue.task_done()
                    break
                fut, texts = job
                if fut.cancelled():
                    _local_embed_queue.task_done()
                    continue
                try:
                    embs = local.encode(texts)
                except Exception as e:
                    logger.warning(f"Local embed worker error: {e}", exc_info=True)
                    embs = [None] * len(texts)
                if not fut.done():
                    fut.set_result(embs)
                _local_embed_queue.task_done()

        t = _threading.Thread(target=worker, daemon=True, name="local-embed-worker")
        t.start()
        _local_embed_worker_started = True


# In-memory cache for embeddings (short TTL, hashed key to bound memory)
_embedding_cache = OrderedDict()
_embedding_cache_ttl = 300  # seconds
_embedding_cache_maxsize = 10000  # bounded LRU size

def _cache_key(texts, model, space):
    import hashlib
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return (h.hexdigest(), len(texts), model, space)

def _fetch_batch_with_endpoint(endpoint, batch_texts):
    results = []
    if not batch_texts:
        return results
    if config.DEBUG_VERBOSE:
        logger.debug(f"Embedding batch -> {endpoint['url']} model={endpoint['model']} backend={endpoint.get('backend','lmstudio')}")
    try:
        provider = create_backend(endpoint)
        embeddings = provider.embeddings(batch_texts, model=endpoint.get('model'))
        for i, emb in enumerate(embeddings):
            if emb is not None and i < len(batch_texts):
                results.append((batch_texts[i], emb))
    except Exception as e:
        if config.DEBUG_VERBOSE:
            logger.exception(f"Batch embedding exception: {e}")
    return results


def get_embeddings_batch(texts, model=None, batch_size=None, space='hyperbolic'):
    # Use local embedder ONLY when no explicit model requested, or when model matches local repo
    local_model_name = getattr(config, "LOCAL_EMBEDDER_MODEL_REPO", "smcleod/text-embedding-mxbai-embed-large-v1")
    use_local = (
        getattr(config, "USE_LOCAL_EMBEDDER", False)
        and (model is None or model == local_model_name)
    )
    if use_local:
        _start_local_embed_worker()
        # Split texts into batches of 32 to avoid huge single job
        batch_size = 32
        pending = []
        for i in range(0, len(texts), batch_size):
            sub_texts = texts[i:i+batch_size]
            fut: _Future = _Future()
            try:
                _local_embed_queue.put((fut, sub_texts), timeout=5)
            except _queue.Full:
                logger.warning("Local embed queue full; returning None for batch at %d", i)
                fut.set_result([None] * len(sub_texts))
            pending.append((fut, i))
        # Wait for all results (blocking on Future, no busy-wait)
        result = [None] * len(texts)
        for fut, start in pending:
            try:
                embs = fut.result(timeout=getattr(config, "LOCAL_EMBED_TIMEOUT", 120))
            except Exception as e:
                logger.warning(f"Local embed batch failed: {e}", exc_info=True)
                embs = [None] * min(batch_size, len(texts) - start)
            for offset, emb in enumerate(embs):
                idx = start + offset
                if space == 'hyperbolic' and emb is not None:
                    from core.hyperbolic import exp_map
                    emb = exp_map(np.array(emb, dtype=np.float32)).tolist()
                result[idx] = emb
        return result

    """Return embeddings for a list of texts, preferably from cache.
       space: 'hyperbolic' or 'euclidean'. Default is hyperbolic.
       Uses in-memory cache and persistent SQLite cache.
    """
    if model is None:
        model = config.EMBEDDING_ENDPOINTS[0]["model"]
    if batch_size is None:
        batch_size = config.EMBEDDING_BATCH_SIZE

    if not texts:
        return []

    # --- In-memory cache check (TTL/maxsize from config, not hardcoded) ---
    _ttl = float(getattr(config, "EMBEDDING_CACHE_TTL", _embedding_cache_ttl))
    cache_key = _cache_key(texts, model, space)
    cached = _embedding_cache.get(cache_key)
    if cached is not None:
        age = time.time() - cached['timestamp']
        if age < _ttl:
            if cached.get('quant') == 'f16':
                import numpy as _np3
                return [(_np3.asarray(r, dtype=_np3.float32).tolist() if r is not None else None) for r in cached['result']]
            return cached['result']
        else:
            del _embedding_cache[cache_key]

    # --- Persistent cache check ---
    result = [None] * len(texts)
    uncached_indices = []
    uncached_texts = []

    conn = db.db_connect("embeddings")
    cur = conn.cursor()
    # Ensure space column exists (migration)
    cur.execute("PRAGMA table_info(embedding_cache)")
    cols = [row[1] for row in cur.fetchall()]
    if "space" not in cols:
        cur.execute("ALTER TABLE embedding_cache ADD COLUMN space TEXT DEFAULT 'euclidean'")
        conn.commit()
    # We'll also ensure space column in document_embeddings/chunk_embeddings if needed, but not here.

    BATCH = 500
    for start in range(0, len(texts), BATCH):
        chunk = texts[start:start+BATCH]
        placeholders = ",".join("?" for _ in chunk)
        cur.execute(
            f"SELECT text, embedding, space FROM embedding_cache WHERE model=? AND text IN ({placeholders})",
            (model, *chunk),
        )
        rows_all = cur.fetchall()
        cache_map = {}
        _exp_dim = int(getattr(config, "EMBEDDING_DIM", 1024))
        for row in rows_all:
            try:
                _blob = row["embedding"]
                if _blob is not None and (len(_blob) // 4) != _exp_dim:
                    logger.warning(f"Embedding dim mismatch for model={model}: got {len(_blob)//4}, expected {_exp_dim}. Quarantined (not mixed) — poison risk if endpoint changed. Row text len={len(row['text'])}.")
                    try:
                        from core.metrics import inc_counter as _inc3
                        _inc3("embedding_dim_mismatch_total")
                    except Exception:
                        pass
                    continue
            except Exception:
                pass
            if row["space"] == space:
                cache_map[row["text"]] = row["embedding"]
            elif space == 'hyperbolic' and row["space"] == 'euclidean':
                # Convert Euclidean to hyperbolic on the fly and store hyperbolic copy
                euclid = np.frombuffer(row["embedding"], dtype=np.float32)
                from core.hyperbolic import exp_map
                hyperbolic = exp_map(euclid)
                blob = sqlite3.Binary(hyperbolic.astype(np.float32).tobytes())
                cur.execute(
                    "INSERT OR REPLACE INTO embedding_cache (text, embedding, model, space) VALUES (?, ?, ?, 'hyperbolic')",
                    (row["text"], blob, model)
                )
                cache_map[row["text"]] = blob
        for i in range(len(chunk)):
            idx = start + i
            text = chunk[i]
            if text in cache_map:
                result[idx] = np.frombuffer(cache_map[text], dtype=np.float32).tolist()
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)
    conn.commit()
    conn.close()

    # If all cached, store in memory and return (respect quant)
    if not uncached_texts:
        _quant2 = str(getattr(config, "EMBEDDING_QUANT", "float32")).lower()
        if _quant2 in ("float16", "fp16", "f16"):
            import numpy as _np4
            qres = [(_np4.asarray(r, dtype=_np4.float16) if r is not None else None) for r in result]
            _embedding_cache[cache_key] = {'result': qres, 'timestamp': time.time(), 'quant': 'f16'}
        else:
            _embedding_cache[cache_key] = {'result': result, 'timestamp': time.time(), 'quant': 'f32'}
        _embedding_cache.move_to_end(cache_key)
        # Evict oldest if over maxsize (config-driven, not hardcoded)
        _maxsize = int(getattr(config, "EMBEDDING_CACHE_MAXSIZE", _embedding_cache_maxsize))
        while len(_embedding_cache) > _maxsize:
            _embedding_cache.popitem(last=False)
        return result

    # --- Fetch uncached embeddings from backend ---
    batches = [uncached_texts[i:i+batch_size] for i in range(0, len(uncached_texts), batch_size)]
    n_endpoints = len(config.EMBEDDING_ENDPOINTS)
    max_workers = min(n_endpoints, len(batches))
    if max_workers == 0:
        max_workers = 1

    batch_tasks = []
    for idx, batch in enumerate(batches):
        endpoint = config.EMBEDDING_ENDPOINTS[idx % n_endpoints]
        batch_tasks.append((endpoint, batch))

    retrieved = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_fetch_batch_with_endpoint, ep, batch) for ep, batch in batch_tasks]
        for future in as_completed(futures):
            try:
                batch_result = future.result()
                for text, emb in batch_result:
                    retrieved[text] = emb
            except Exception as e:
                if config.DEBUG_VERBOSE:
                    print(f"    (Embedding batch failed: {e})")
                # Continue with other batches; missing embeddings will be None

    # Store in persistent cache and fill result (single txn executemany, bulk-checkpoint friendly)
    conn = db.db_connect("embeddings")
    try:
        # Raise autocheckpoint during bulk to avoid per-commit WAL stall; generic, restored after
        if len(write_rows if 'write_rows' in locals() else []) > 200 or len(uncached_texts) > 200:
            try:
                conn.execute(f"PRAGMA wal_autocheckpoint={int(getattr(config, 'BULK_WAL_AUTOCHECKPOINT', 4000))}")
            except Exception:
                pass
        write_rows = []
        for i, text in zip(uncached_indices, uncached_texts):
            if text in retrieved:
                emb = retrieved[text]
                if space == 'hyperbolic':
                    from core.hyperbolic import exp_map
                    emb = exp_map(np.array(emb, dtype=np.float32))
                result[i] = emb
                blob = sqlite3.Binary(np.array(emb, dtype=np.float32).tobytes())
                write_rows.append((text, blob, model, space))
        if write_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache (text, embedding, model, space) VALUES (?, ?, ?, ?)",
                write_rows,
            )
        conn.commit()
    finally:
        conn.close()

    # Store in memory cache (optional float16 to halve RAM; dequantized on hit transparently)
    _quant = str(getattr(config, "EMBEDDING_QUANT", "float32")).lower()
    if _quant in ("float16", "fp16", "f16"):
        import numpy as _np2
        qresult = [(_np2.asarray(r, dtype=_np2.float16) if r is not None else None) for r in result]
        _embedding_cache[cache_key] = {'result': qresult, 'timestamp': time.time(), 'quant': 'f16'}
    else:
        _embedding_cache[cache_key] = {'result': result, 'timestamp': time.time(), 'quant': 'f32'}
    # Move to end for LRU
    _embedding_cache.move_to_end(cache_key)
    # Evict oldest if over maxsize (config-driven)
    _maxsize = int(getattr(config, "EMBEDDING_CACHE_MAXSIZE", _embedding_cache_maxsize))
    while len(_embedding_cache) > _maxsize:
        _embedding_cache.popitem(last=False)
    return result


def get_embedding(text, model=None, space='hyperbolic'):
    """Return a single embedding vector (hyperbolic by default)."""
    result = get_embeddings_batch([text], model=model, space=space)
    return result[0] if result else None


def get_embeddings_dict(texts, model=None, space='hyperbolic', batch_size=None):
    """Batch helper returning {text: emb} with single cache/HTTP fan-out.

    Preserves model/space semantics of get_embeddings_batch; dedups inputs
    generically (no doc-specific logic) to avoid duplicate work.
    """
    if not texts:
        return {}
    # Preserve order but dedup for efficiency
    seen = {}
    uniq = []
    for t in texts:
        if t not in seen:
            seen[t] = True
            uniq.append(t)
    embs = get_embeddings_batch(uniq, model=model, batch_size=batch_size, space=space)
    return {t: e for t, e in zip(uniq, embs) if e is not None}


def load_embedding_from_db(text, model=None):
    if model is None:
        model = config.EMBEDDING_ENDPOINTS[0]["model"]
    conn = db.db_connect("embeddings")
    cur = conn.cursor()
    cur.execute(
        "SELECT embedding FROM embedding_cache WHERE text=? AND model=? ORDER BY rowid DESC LIMIT 1",
        (text, model),
    )
    row = cur.fetchone()
    conn.close()
    if row:
        return np.frombuffer(row[0], dtype=np.float32).tolist()
    return None


def decode_embedding_blob(blob, context=""):
    """Decode float32 blob to np array with dim guard (quarantine foreign dims, never mix).

    Returns np array or None (with warning + metric on mismatch). Generic.
    """
    if blob is None:
        return None
    try:
        import numpy as _npd
        exp_dim = int(getattr(config, "EMBEDDING_DIM", 1024))
        arr = _npd.frombuffer(blob, dtype=_npd.float32)
        if len(arr) != exp_dim:
            logger.warning(f"Embedding dim mismatch {context}: got {len(arr)}, expected {exp_dim}. Quarantined.")
            try:
                from core.metrics import inc_counter as _inc4
                _inc4("embedding_dim_mismatch_total")
            except Exception:
                pass
            return None
        return arr.copy()
    except Exception:
        return None


def validate_embedding_config(probe=True):
    """Validate embedding endpoints align to EMBEDDING_DIM (default 1024 mxbai).

    Probes each configured embedding endpoint with one short text (single batch,
    no quality impact), asserts dim matches contract, EXCLUDES mismatched endpoints
    from rotation (never mixes dims = poison prevention), and scans stored blobs
    for foreign dims with loud warnings. Returns (ok_endpoints, warnings).
    Generic, no doc-specific logic.
    """
    import warnings as _w
    exp_dim = int(getattr(config, "EMBEDDING_DIM", 1024))
    ok = []
    warns = []
    # 1. Stored-blob audit (cheap COUNT + sample, not full scan)
    try:
        conn = db.db_connect("embeddings")
        cur = conn.cursor()
        try:
            cur.execute("SELECT embedding, model FROM embedding_cache LIMIT 200")
            foreign = {}
            for r in cur.fetchall():
                try:
                    d = (len(r["embedding"]) // 4) if r["embedding"] else 0
                    if d and d != exp_dim:
                        foreign[r["model"]] = foreign.get(r["model"], 0) + 1
                except Exception:
                    continue
            for m, c in foreign.items():
                msg = (f"POISON RISK: embedding_cache holds {c}/200 sampled rows with dim != {exp_dim} "
                       f"(model={m}). New endpoint with different dims will NOT reuse these rows "
                       f"(quarantined per-row), but re-embedding everything is required for full alignment. "
                       f"Do NOT switch EMBEDDING_MODEL lightly.")
                logger.warning(msg)
                warns.append(msg)
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception as e:
        warns.append(f"Stored-blob audit skipped: {e}")
    # 2. Live endpoint probe (single text each; parallel when many endpoints, same checks)
    if probe:
        try:
            eps = getattr(config, "EMBEDDING_ENDPOINTS", [])

            def _probe_one(ep):
                try:
                    from core.backends import create_backend
                    prov = create_backend(ep)
                    vecs = prov.embeddings(["alignment probe"], model=ep.get("model"))
                    d = len(vecs[0]) if vecs and vecs[0] is not None else 0
                    return (ep, d, None)
                except Exception as e:
                    return (ep, 0, str(e))

            _results = []
            if len(list(eps)) > 3:
                try:
                    from concurrent.futures import ThreadPoolExecutor as _TPE
                    with _TPE(max_workers=min(len(list(eps)), 8)) as _ex:
                        _results = list(_ex.map(_probe_one, list(eps)))
                except Exception:
                    _results = [_probe_one(ep) for ep in list(eps)]
            else:
                _results = [_probe_one(ep) for ep in list(eps)]
            for ep, d, err in _results:
                try:
                    if err:
                        warns.append(f"Endpoint probe failed for {ep.get('url')}: {err}")
                        continue
                    if d != exp_dim:
                        msg = (f"POISON RISK: endpoint {ep.get('url')}:{ep.get('model')} returned dim={d}, "
                               f"expected {exp_dim} (mxbai/1024 contract). EXCLUDED from rotation — "
                               f"fix BACKEND_EMBEDDINGS_MODEL or run full re-embed migration.")
                        logger.warning(msg)
                        warns.append(msg)
                        try:
                            eps.remove(ep)
                        except Exception:
                            pass
                    else:
                        ok.append(ep)
                except Exception as e:
                    warns.append(f"Endpoint probe failed for {ep.get('url')}: {e}")
            for w in warns:
                print(f"  [EMBEDDING WARNING] {w}")
        except Exception as e:
            warns.append(f"Probe loop failed: {e}")
    else:
        ok = list(getattr(config, "EMBEDDING_ENDPOINTS", []))
    return ok, warns


def store_embedding_to_cache(text, embedding, model=None):
    if model is None:
        model = config.EMBEDDING_ENDPOINTS[0]["model"]
    blob = sqlite3.Binary(np.array(embedding, dtype=np.float32).tobytes())
    conn = db.db_connect("embeddings")
    conn.execute(
        "INSERT OR REPLACE INTO embedding_cache (text, embedding, model, space) VALUES (?, ?, ?, 'hyperbolic')",
        (text, blob, model),
    )
    conn.commit()
    conn.close()
