import numpy as np


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
        self._ensure_capacity(self._count + 1)
        idx = self._count
        self._data[idx] = vector
        self._count += 1
        return idx

    def add_batch(self, vectors) -> list[int]:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(f"vectors must have shape (n, {self._dim}), got {vectors.shape}")
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
