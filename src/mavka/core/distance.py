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
