import numpy as np
from fuzzywuzzy import fuzz
from core.text_utils import normalise_text


def normalize_name(name: str) -> str:
    import re
    name = normalise_text(name or "").lower()
    name = re.sub(r'[^\w\s]', '', name)
    return name


def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def find_matching_global_node(name: str, node_type: str, global_nodes: list[dict],
                              threshold: float = 0.85, name_embedding=None,
                              existing_emb_matrix=None, existing_emb_ids=None) -> int | None:
    """
    Find matching global node using exact name, fuzzy matching, and embedding similarity.
    If existing_emb_matrix and existing_emb_ids are provided, uses vectorized cosine similarity.
    """
    norm_name = normalize_name(name)
    best_id = None
    best_score = 0.0

    # 1. Exact normalized match
    for node in global_nodes:
        if node["node_type"] == node_type and normalize_name(node["canonical_name"]) == norm_name:
            return node["global_node_id"]

    # 2. Fuzzy string match
    for node in global_nodes:
        if node["node_type"] == node_type:
            score = fuzz.token_set_ratio(norm_name, normalize_name(node["canonical_name"])) / 100.0
            if score > best_score:
                best_score = score
                best_id = node["global_node_id"]

    if best_score >= threshold:
        return best_id

    # 3. Embedding similarity (if provided)
    if name_embedding is not None:
        best_sim = 0.0
        best_id_emb = None
        for node in global_nodes:
            if node["node_type"] == node_type and node.get("embedding") is not None:
                emb = np.frombuffer(node["embedding"], dtype=np.float32)
                sim = cosine_similarity(name_embedding, emb)
                if sim > best_sim:
                    best_sim = sim
                    best_id_emb = node["global_node_id"]
        if best_sim >= threshold:
            return best_id_emb

    return None
