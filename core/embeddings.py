
import random
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

# In-memory cache for embeddings (short TTL)
_embedding_cache = {}
_embedding_cache_ttl = 300  # seconds

def _cache_key(texts, model, space):
    return (tuple(texts), model, space)

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

    # --- In-memory cache check ---
    cache_key = _cache_key(texts, model, space)
    cached = _embedding_cache.get(cache_key)
    if cached is not None:
        age = time.time() - cached['timestamp']
        if age < _embedding_cache_ttl:
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
        for row in rows_all:
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

    # If all cached, store in memory and return
    if not uncached_texts:
        _embedding_cache[cache_key] = {'result': result, 'timestamp': time.time()}
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

    # Store in persistent cache and fill result
    conn = db.db_connect("embeddings")
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
    conn.close()

    # Store in memory cache
    _embedding_cache[cache_key] = {'result': result, 'timestamp': time.time()}
    return result


def get_embedding(text, model=None, space='hyperbolic'):
    """Return a single embedding vector (hyperbolic by default)."""
    result = get_embeddings_batch([text], model=model, space=space)
    return result[0] if result else None


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
