from pathlib import Path

import numpy as np
from core import db
from core.embeddings import get_embeddings_batch
from core.hyperbolic import exp_map, hyperbolic_distance, frechet_mean

def get_knowledge_centroid():
    """Compute hyperbolic centroid of all processed chunk embeddings."""
    conn = db.db_connect("embeddings")
    cur = conn.cursor()
    cur.execute("SELECT embedding FROM chunk_embeddings")
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return None
    points = []
    for row in rows:
        emb = np.frombuffer(row["embedding"], dtype=np.float32)
        points.append(exp_map(emb))
    return frechet_mean(points, steps=10)

def score_file_by_distance(file_path, centroid):
    """
    Score file by hyperbolic distance using title + sampled pages.
    Uses local embedder if USE_LOCAL_EMBEDDER is true, otherwise backend.
    """
    import config
    try:
        name = Path(file_path).stem
        sample = _extract_sample_text(file_path)
        combined = name
        if sample:
            combined += " " + sample[:1000]
        if not combined.strip():
            combined = name

        if getattr(config, "USE_LOCAL_EMBEDDER", False):
            from core.local_embedder import get_local_embedder
            local = get_local_embedder()
            if local.available:
                raw = local.encode([combined])[0]
                if raw is None:
                    return float('inf')
                from core.hyperbolic import exp_map
                h = exp_map(np.array(raw, dtype=np.float32))
                dist = hyperbolic_distance(h, centroid)
                # Debug print (temporary)
                if getattr(config, "DEBUG_VERBOSE", False):
                    print(f"    Distance for {name}: {dist:.4f}")
                return dist
            # fallback to backend if local not available

        from core.embeddings import get_embeddings_batch
        emb = get_embeddings_batch([combined], space='hyperbolic', model=config.EMBEDDING_MODEL)[0]
        if emb is None:
            return float('inf')
        h = np.array(emb, dtype=np.float32)
        if np.linalg.norm(h) > 1.0:
            from core.hyperbolic import exp_map
            h = exp_map(h)
        dist = hyperbolic_distance(h, centroid)
        if getattr(config, "DEBUG_VERBOSE", False):
            print(f"    Distance (backend) for {name}: {dist:.4f}")
        return dist
    except Exception as e:
        if getattr(config, "DEBUG_VERBOSE", False):
            print(f"    (score_file_by_distance error: {e})")
        return float('inf')
def _extract_sample_text(file_path, max_pages=5, fallback_pages=20, min_chars=200):
    """
    Extract a lightweight sample of text from a file.
    For PDFs: first max_pages; if too short, expand to fallback_pages.
    For other files: first 2000 chars.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    if suffix == '.pdf':
        try:
            import pymupdf as fitz
            fitz.TOOLS.mupdf_display_errors(False)
            doc = fitz.open(str(file_path))
            total_pages = doc.page_count
            pages_to_try = max_pages if total_pages >= max_pages else total_pages
            sample = ""
            for p in range(pages_to_try):
                sample += doc[p].get_text()
            # If sample too short, try more pages
            if len(sample.strip()) < min_chars and total_pages > pages_to_try:
                pages_to_try = min(fallback_pages, total_pages)
                for p in range(max_pages, pages_to_try):
                    sample += doc[p].get_text()
            doc.close()
            return sample.strip()
        except Exception:
            return ""
    else:
        try:
            from extractors.registry import extract_text_from_file
            result = extract_text_from_file(file_path)
            return result.get("text", "")[:2000].strip()
        except Exception:
            return ""



def order_files_by_curriculum(file_paths, max_files=None):
    """Sort files by increasing hyperbolic distance to knowledge centroid using batched embeddings.
       For this operation, always use the backend embedding model to match centroid dimension.
       Processes embeddings in chunks and shows in-place progress."""
    import config
    import numpy as np
    from core.hyperbolic import hyperbolic_distance, exp_map
    from core.embeddings import get_embeddings_batch

    if not file_paths:
        return []

    print(f"  (Curriculum ordering: computing centroid...)")
    centroid = get_knowledge_centroid()
    if centroid is None:
        print("  (No existing knowledge centroid; keeping original order)")
        return list(file_paths)

    total = len(file_paths)
    print(f"  (Curriculum ordering: scoring {total} files...)")

    # Force backend embeddings for curriculum scoring
    old_local_flag = getattr(config, "USE_LOCAL_EMBEDDER", False)
    config.USE_LOCAL_EMBEDDER = False

    try:
        scored = []
        chunk_size = 20
        for start in range(0, total, chunk_size):
            chunk_files = file_paths[start:start+chunk_size]
            combined_texts = []
            for f in chunk_files:
                name = Path(f).stem
                sample = _extract_sample_text(f)
                combined = name
                if sample:
                    combined += " " + sample[:1000]
                if not combined.strip():
                    combined = name
                combined_texts.append(combined)

            model = getattr(config, "EMBEDDING_MODEL", None)
            embs = get_embeddings_batch(combined_texts, space='hyperbolic', model=model)

            for f, emb in zip(chunk_files, embs):
                if emb is None:
                    scored.append((float('inf'), f))
                    continue
                h = np.array(emb, dtype=np.float32)
                if np.linalg.norm(h) > 1.0:
                    h = exp_map(h)
                dist = hyperbolic_distance(h, centroid)
                scored.append((dist, f))

            # In-place progress
            done = min(start + chunk_size, total)
            print(f"\r    Scored {done}/{total} files", end="", flush=True)
        print()  # newline after progress
    finally:
        config.USE_LOCAL_EMBEDDER = old_local_flag

    scored.sort(key=lambda x: x[0])
    ordered = [f for _, f in scored]
    if max_files:
        ordered = ordered[:max_files]
    print(f"  (Curriculum ordering complete)")
    return ordered
