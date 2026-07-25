import numpy as np

from mavka.kmeans import assign, kmeans
from mavka.store import VectorStore, normalize


class IVFIndex:
    def __init__(self, dim: int, n_lists: int = 100, seed: int = 0):
        self.dim = dim
        self.n_lists = n_lists
        self.seed = seed
        # how many buckets to check
        self.nprobe = 1

        self._store = VectorStore(dim=dim)
        self._trained = False
        self._centroids: np.ndarray | None = None
        self._n_lists_actual = 0
        self._inverted_lists: list[list[int]] = []

    def train(self, vectors) -> None:
        vectors = normalize(np.asarray(vectors, dtype=np.float32))
        n = vectors.shape[0]
        # Fewer training vectors than n_lists: reduce the effective number of
        # clusters to n rather than raising, so training never fails outright.
        n_lists_actual = min(self.n_lists, n)

        centroids, _ = kmeans(vectors, k=n_lists_actual, seed=self.seed)

        self._centroids = centroids
        self._n_lists_actual = n_lists_actual
        self._inverted_lists = [[] for _ in range(n_lists_actual)]
        self._trained = True

    def _require_trained(self) -> None:
        if not self._trained:
            raise ValueError("IVFIndex must be trained before add/search (call train() first)")

    def add(self, vector) -> int:
        self._require_trained()
        id_ = self._store.add(vector)
        normalized = self._store.get(id_)
        bucket = int(assign(normalized[np.newaxis, :], self._centroids)[0])
        self._inverted_lists[bucket].append(id_)
        return id_

    def add_batch(self, vectors) -> list[int]:
        self._require_trained()
        ids = self._store.add_batch(vectors)
        normalized_batch = np.stack([self._store.get(id_) for id_ in ids])
        buckets = assign(normalized_batch, self._centroids)
        for id_, bucket in zip(ids, buckets):
            self._inverted_lists[int(bucket)].append(id_)
        return ids

    def _nearest_centroids(self, query: np.ndarray, nprobe: int) -> np.ndarray:
        centroids = self._centroids
        if nprobe >= len(centroids):
            return np.arange(len(centroids))

        centroids_sq = np.sum(centroids**2, axis=1)
        cross = centroids @ query
        dist_sq = centroids_sq - 2 * cross

        nearest = np.argpartition(dist_sq, nprobe - 1)[:nprobe]
        order = np.argsort(dist_sq[nearest])
        return nearest[order]

    def search(self, query, k: int, nprobe: int | None = None) -> list[tuple[int, float]]:
        self._require_trained()
        if k <= 0:
            raise ValueError(f"k must be positive, got {k!r}")
        if len(self) == 0:
            return []

        probe = self.nprobe if nprobe is None else nprobe
        probe = max(1, min(probe, self._n_lists_actual))

        query = normalize(np.asarray(query, dtype=np.float32))
        bucket_indices = self._nearest_centroids(query, probe)

        candidate_ids: list[int] = []
        for bucket in bucket_indices:
            candidate_ids.extend(self._inverted_lists[bucket])

        if not candidate_ids:
            return []

        candidate_vectors = np.stack([self._store.get(id_) for id_ in candidate_ids])
        scores = candidate_vectors @ query

        k_eff = min(k, len(candidate_ids))
        if k_eff < len(candidate_ids):
            top_local = np.argpartition(scores, -k_eff)[-k_eff:]
        else:
            top_local = np.arange(len(candidate_ids))

        order = np.argsort(scores[top_local])[::-1]
        top_local = top_local[order]

        return [(candidate_ids[i], float(scores[i])) for i in top_local]

    @property
    def count(self) -> int:
        return len(self._store)

    def __len__(self) -> int:
        return len(self._store)
