
import numpy as np
from core.hyperbolic import exp_map, log_map, hyperbolic_distance, frechet_mean

def cluster_hyperbolic(embeddings, n_clusters=None, max_dist=None, max_iter=20, tol=1e-4):
    embeddings = [np.asarray(e, dtype=np.float32) for e in embeddings]
    if not embeddings:
        return []

    if n_clusters is not None:
        n_clusters = min(n_clusters, len(embeddings))
        rng = np.random.default_rng(42)
        indices = rng.choice(len(embeddings), n_clusters, replace=False)
        centroids = [embeddings[i].copy() for i in indices]

        for iteration in range(max_iter):
            labels = []
            for emb in embeddings:
                dists = [hyperbolic_distance(emb, centroid) for centroid in centroids]
                labels.append(int(np.argmin(dists)))

            new_centroids = []
            for k in range(n_clusters):
                cluster_points = [embeddings[i] for i, lab in enumerate(labels) if lab == k]
                if cluster_points:
                    if len(cluster_points) > 200:
                        sample = rng.choice(cluster_points, 200, replace=False)
                        new_centroid = frechet_mean(sample, steps=20)
                    else:
                        new_centroid = frechet_mean(cluster_points, steps=20)
                else:
                    new_centroid = embeddings[rng.choice(len(embeddings))].copy()
                new_centroids.append(new_centroid)

            max_shift = max(hyperbolic_distance(old, new) for old, new in zip(centroids, new_centroids))
            centroids = new_centroids
            if max_shift < tol:
                break

        clusters = [[] for _ in range(n_clusters)]
        for i, emb in enumerate(embeddings):
            dists = [hyperbolic_distance(emb, centroid) for centroid in centroids]
            clusters[int(np.argmin(dists))].append(i)
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
