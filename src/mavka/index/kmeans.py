import numpy as np


def assign(vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    centroids = np.asarray(centroids, dtype=np.float32)

    vectors_sq = np.sum(vectors**2, axis=1, keepdims=True)
    centroids_sq = np.sum(centroids**2, axis=1)
    cross = vectors @ centroids.T
    dist_sq = vectors_sq + centroids_sq - 2 * cross

    return np.argmin(dist_sq, axis=1)


def _update_centroids(
    vectors: np.ndarray,
    assignments: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    dim = vectors.shape[1]
    n = vectors.shape[0]
    centroids = np.zeros((k, dim), dtype=np.float32)

    for cluster in range(k):
        mask = assignments == cluster
        if np.any(mask):
            centroids[cluster] = vectors[mask].mean(axis=0)
        else:
            centroids[cluster] = vectors[rng.integers(0, n)]

    return centroids


def kmeans(
    vectors: np.ndarray, k: int, n_iters: int = 25, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    # convert the vectors array to array float 32
    vectors = np.asarray(vectors, dtype=np.float32)
    # shape returns the dimensions, and we are taking the number of rows
    n = vectors.shape[0]

    # create range
    rng = np.random.default_rng(seed)
    # create centroids, and the numbers you can pick is from 0 to n and there is gonna be k of them
    # it returns an array of k items
    initial_indices = rng.choice(n, size=k, replace=False)
    # pick the vectors of these indexes and make them centroids
    centroids = vectors[initial_indices].copy().astype(np.float32)

    assignments = assign(vectors, centroids)

    for _ in range(n_iters):
        centroids = _update_centroids(vectors, assignments, k, rng)
        new_assignments = assign(vectors, centroids)
        if np.array_equal(new_assignments, assignments):
            assignments = new_assignments
            break
        assignments = new_assignments

    return centroids, assignments
