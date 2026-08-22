import numpy as np
from fuzzywuzzy import fuzz
from core.text_utils import normalise_text


def normalize_name(name: str) -> str:
    """Normalize node name for cross-document comparison."""
    import re
    name = normalise_text(name).lower()
    name = re.sub(r'[^\w\s]', '', name)
    return name


def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def find_matching_global_node(name: str, node_type: str, global_nodes: list[dict],
                              threshold: float = 0.85, name_embedding=None) -> int | None:
    """
    Find an existing global node by name similarity or embedding similarity.
    If name_embedding is provided, use it instead of calling get_embedding.
    """
    norm_name = normalize_name(name)
    best_id = None
    best_score = 0.0

    # First pass: exact match
    for node in global_nodes:
        if node["node_type"] == node_type and normalize_name(node["canonical_name"]) == norm_name:
            return node["global_node_id"]

    # Fuzzy name match
    for node in global_nodes:
        if node["node_type"] == node_type:
            score = fuzz.token_set_ratio(norm_name, normalize_name(node["canonical_name"])) / 100.0
            if score > best_score:
                best_score = score
                best_id = node["global_node_id"]

    if best_score >= threshold:
        return best_id

    # Embedding similarity
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