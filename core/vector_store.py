
import numpy as np
import sqlite3
from pathlib import Path
import config

class ExactVectorStore:
    """Brute-force exact vector search using memory-mapped numpy arrays."""
    def __init__(self, db_path, table_name, id_col, emb_col):
        self.db_path = db_path
        self.table_name = table_name
        self.id_col = id_col
        self.emb_col = emb_col
        self.ids = []
        self.embeddings = None
        self.dim = None
        self._load()

    def _load(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(f"SELECT {self.id_col}, {self.emb_col} FROM {self.table_name} WHERE {self.emb_col} IS NOT NULL")
        rows = cur.fetchall()
        conn.close()
        if not rows:
            return
        self.ids = [row[0] for row in rows]
        self.dim = len(np.frombuffer(rows[0][1], dtype=np.float32))
        # Write embeddings to a memory-mapped file
        mmap_path = Path(self.db_path).parent / f"{self.table_name}_embeddings.dat"
        fp = np.memmap(mmap_path, dtype='float32', mode='w+', shape=(len(self.ids), self.dim))
        for i, (_, blob) in enumerate(rows):
            fp[i] = np.frombuffer(blob, dtype=np.float32)
        fp.flush()
        self.embeddings = fp

    def search(self, query_embedding, top_k=10):
        """Return top_k (id, score) where score is cosine similarity."""
        if self.embeddings is None or self.dim is None:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        # Normalize query
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm
        # Compute dot product in chunks to avoid huge memory
        scores = np.zeros(len(self.ids), dtype=np.float32)
        chunk_size = 10000
        for start in range(0, len(self.ids), chunk_size):
            end = min(start + chunk_size, len(self.ids))
            sub = self.embeddings[start:end]
            norms = np.linalg.norm(sub, axis=1)
            norms[norms == 0] = 1e-8
            dots = sub @ q
            scores[start:end] = dots / norms
        top_idx = np.argsort(scores)[-top_k:][::-1]
        return [(self.ids[i], float(scores[i])) for i in top_idx]

    def add(self, id, embedding):
        """Add a new embedding to the store."""
        if self.dim is None:
            self.dim = len(embedding)
            self.embeddings = np.memmap(
                Path(self.db_path).parent / f"{self.table_name}_embeddings.dat",
                dtype='float32', mode='w+', shape=(1, self.dim))
        else:
            # Extend memory-mapped array
            old_shape = self.embeddings.shape
            new_shape = (old_shape[0] + 1, self.dim)
            self.embeddings = np.memmap(self.embeddings.filename, dtype='float32', mode='r+', shape=new_shape)
        self.embeddings[-1] = np.array(embedding, dtype=np.float32)
        self.ids.append(id)

    def close(self):
        if self.embeddings is not None:
            self.embeddings.flush()
