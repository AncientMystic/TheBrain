
import numpy as np
from core.hyperbolic import exp_map, log_map, hyperbolic_distance, frechet_mean

def cluster_hyperbolic(embeddings, n_clusters=None, max_dist=None):
    embeddings = [np.asarray(e, dtype=np.float32) for e in embeddings]
    if not embeddings:
        return []
    if n_clusters is not None:
        n_clusters = min(n_clusters, len(embeddings))
        X = np.array([log_map(e) for e in embeddings], dtype=np.float32)
        rng = np.random.default_rng(42)
        centroids = X[rng.choice(len(X), n_clusters, replace=False)]
        for _ in range(20):
            dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.array([X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k] for k in range(n_clusters)])
            if np.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids
        clusters = [[] for _ in range(n_clusters)]
        for i, lab in enumerate(labels):
            clusters[lab].append(i)
        return clusters
    elif max_dist is not None:
        clusters = []
        used = set()
        for i, e in enumerate(embeddings):
            if i in used:
                continue
            cluster = [i]
            used.add(i)
            for j in range(i+1, len(embeddings)):
                if j not in used and hyperbolic_distance(e, embeddings[j]) < max_dist:
                    cluster.append(j)
                    used.add(j)
            clusters.append(cluster)
        return clusters
    else:
        return [[i] for i in range(len(embeddings))]

def select_representatives(items, embeddings, clusters):
    reps = []
    for cluster in clusters:
        if not cluster:
            continue
        if len(cluster) == 1:
            reps.append(items[cluster[0]])
        else:
            cluster_embs = [embeddings[i] for i in cluster]
            centroid = frechet_mean(cluster_embs, steps=10)
            best_idx = min(cluster, key=lambda i: hyperbolic_distance(embeddings[i], centroid))
            reps.append(items[best_idx])
    return reps
