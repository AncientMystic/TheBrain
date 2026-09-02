import numpy as np
from fuzzywuzzy import fuzz
from core.text_utils import normalise_text


def normalize_name(name: str) -> str:
    import re
    name = normalise_text(name or "").lower()
    name = re.sub(r'[^\w\s]', '', name)
    return name





def find_matching_global_node(name: str, node_type: str, global_nodes: list[dict],
                              threshold: float = 0.85, name_embedding=None,
                              existing_emb_matrix=None, existing_emb_ids=None) -> int | None:
    """Find matching global node using exact name, fuzzy matching, and hyperbolic embedding similarity.
       Skips nodes with mismatched embedding dimensions."""
    from core.hyperbolic import hyperbolic_distance

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

    # Dynamic threshold based on embedding similarities if available
    dynamic_threshold = threshold
    if name_embedding is not None and len(global_nodes) > 0:
        all_sims = []
        for node in global_nodes:
            if node["node_type"] == node_type and node.get("embedding") is not None:
                emb = np.frombuffer(node["embedding"], dtype=np.float32)
                name_emb_arr = np.array(name_embedding, dtype=np.float32)
                if name_emb_arr.shape != emb.shape:
                    continue
                dist = hyperbolic_distance(name_emb_arr, emb)
                sim = 1.0 / (1.0 + dist)
                all_sims.append(sim)
        if all_sims:
            median_sim = float(np.median(all_sims))
            dynamic_threshold = max(0.7, min(threshold, median_sim))
    if best_score >= dynamic_threshold:
        return best_id

    # 3. Hyperbolic embedding similarity with dimension guard
    if name_embedding is not None:
        best_sim = 0.0
        best_id_emb = None
        for node in global_nodes:
            if node["node_type"] == node_type and node.get("embedding") is not None:
                emb = np.frombuffer(node["embedding"], dtype=np.float32)
                name_emb_arr = np.array(name_embedding, dtype=np.float32)
                if name_emb_arr.shape != emb.shape:
                    continue
                dist = hyperbolic_distance(name_emb_arr, emb)
                sim = 1.0 / (1.0 + dist)
                if sim > best_sim:
                    best_sim = sim
                    best_id_emb = node["global_node_id"]
        if best_sim >= threshold:
            return best_id_emb

    return None
