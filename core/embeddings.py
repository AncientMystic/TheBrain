import random
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

import config
from core.logger import get_logger
logger = get_logger(__name__)
from core import db


def _fetch_batch_with_endpoint(endpoint, batch_texts):
    results = []
    if not batch_texts:
        return results
    if config.DEBUG_VERBOSE:
        logger.debug(f"Embedding batch -> {endpoint['url']} model={endpoint['model']}")
    try:
        resp = requests.post(
            f"{endpoint['url']}/embeddings",
            json={"input": batch_texts, "model": endpoint['model']},
            timeout=config.EMBEDDING_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get('data', []):
                idx = item.get('index', 0)
                emb = item.get('embedding')
                if emb is not None and idx < len(batch_texts):
                    results.append((batch_texts[idx], emb))
        else:
            if config.DEBUG_VERBOSE:
                logger.error(f"Batch embedding error {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        if config.DEBUG_VERBOSE:
            logger.exception(f"Batch embedding exception: {e}")
    return results


def get_embeddings_batch(texts, model=None, batch_size=None):
    if model is None:
        model = config.EMBEDDING_ENDPOINTS[0]["model"]
    if batch_size is None:
        batch_size = config.EMBEDDING_BATCH_SIZE

    if not texts:
        return []

    result = [None] * len(texts)
    uncached_indices = []
    uncached_texts = []

    conn = db.db_connect("embeddings")
    cur = conn.cursor()
    # Batch cache lookup for efficiency
    BATCH = 500
    for start in range(0, len(texts), BATCH):
        chunk = texts[start:start+BATCH]
        placeholders = ",".join("?" for _ in chunk)
        cur.execute(
            f"SELECT text, embedding FROM embedding_cache WHERE model=? AND text IN ({placeholders})",
            (model, *chunk),
        )
        rows = cur.fetchall()
        cache_map = {row[0]: row[1] for row in rows}
        for i in range(len(chunk)):
            idx = start + i
            text = chunk[i]
            if text in cache_map:
                result[idx] = np.frombuffer(cache_map[text], dtype=np.float32).tolist()
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)
    conn.close()

    if not uncached_texts:
        return result

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
            for text, emb in future.result():
                retrieved[text] = emb

    conn = db.db_connect("embeddings")
    write_rows = []
    for i, text in zip(uncached_indices, uncached_texts):
        if text in retrieved:
            emb = retrieved[text]
            result[i] = emb
            blob = sqlite3.Binary(np.array(emb, dtype=np.float32).tobytes())
            write_rows.append((text, blob, model))
    if write_rows:
        conn.executemany(
            "INSERT OR REPLACE INTO embedding_cache (text, embedding, model) VALUES (?, ?, ?)",
            write_rows,
        )
    conn.commit()
    conn.close()

    return result


def get_embedding(text, model=None):
    result = get_embeddings_batch([text], model=model)
    return result[0] if result else None


def load_embedding_from_db(text, model=None):
    if model is None:
        model = config.EMBEDDING_ENDPOINTS[0]["model"]
    conn = db.db_connect("embeddings")
    cur = conn.cursor()
    cur.execute(
        "SELECT embedding FROM embedding_cache WHERE text=? AND model=?",
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
        "INSERT OR REPLACE INTO embedding_cache (text, embedding, model) VALUES (?, ?, ?)",
        (text, blob, model),
    )
    conn.commit()
    conn.close()
