import numpy as np


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        # returns the length of the vector using L2
        norm = np.linalg.norm(vectors)
        if norm == 0:
            raise ValueError("cannot normalize a zero vector")
        # divide each dimention by the length of the vector to shrink this vector length to 1
        return (vectors / norm).astype(np.float32)
    if vectors.ndim == 2:
        # returns the list of length of vectors using L2
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms == 0):
            raise ValueError("cannot normalize a zero vector")
        return (vectors / norms[:, np.newaxis]).astype(np.float32)
    raise ValueError(f"vectors must be 1D or 2D, got shape {vectors.shape}")


class VectorStore:
    def __init__(self, dim: int, initial_capacity: int = 1024):
        self._dim = dim
        self._capacity = initial_capacity
        self._count = 0
        self._data = np.zeros((initial_capacity, dim), dtype=np.float32)

    @property
    def count(self) -> int:
        return self._count

    def __len__(self) -> int:
        return self._count

    def add(self, vector) -> int:
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != self._dim:
            raise ValueError(f"vector must have length {self._dim}, got shape {vector.shape}")
        vector = normalize(vector)
        self._ensure_capacity(self._count + 1)
        idx = self._count
        self._data[idx] = vector
        self._count += 1
        return idx

    def add_batch(self, vectors) -> list[int]:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(f"vectors must have shape (n, {self._dim}), got {vectors.shape}")
        vectors = normalize(vectors)
        n = vectors.shape[0]
        self._ensure_capacity(self._count + n)
        start = self._count
        self._data[start : start + n] = vectors
        self._count += n
        return list(range(start, start + n))

    def get(self, id: int) -> np.ndarray:
        if id < 0 or id >= self._count:
            raise ValueError(f"id {id!r} out of range [0, {self._count})")
        return self._data[id].copy()

    def search(self, query, k: int) -> list[tuple[int, float]]:
        query = np.asarray(query, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self._dim:
            raise ValueError(f"query must have length {self._dim}, got shape {query.shape}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k!r}")
        if self._count == 0:
            return []

        query = normalize(query)
        scores = self._data[: self._count] @ query
        k = min(k, self._count)

        if k < self._count:
            # Indices of the k largest scores, in O(n) time but unsorted among themselves.
            top_ids = np.argpartition(scores, -k)[-k:]
        else:
            top_ids = np.arange(self._count)

        # Sort top_ids by score descending (argsort ranks scores, not ids, so we
        # get a permutation of positions and apply it to top_ids, not to the scores).
        order = np.argsort(scores[top_ids])[::-1]
        top_ids = top_ids[order]

        return [(int(idx), float(scores[idx])) for idx in top_ids]

    def _ensure_capacity(self, required: int) -> None:
        if required <= self._capacity:
            return
        new_capacity = max(self._capacity, 1)
        while new_capacity < required:
            new_capacity *= 2
        new_data = np.zeros((new_capacity, self._dim), dtype=np.float32)
        new_data[: self._count] = self._data[: self._count]
        self._data = new_data
        self._capacity = new_capacity
