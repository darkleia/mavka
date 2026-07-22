import numpy as np

from mavka import VectorStore
from mavka.eval import evaluate, recall_at_k


def test_recall_at_k_perfect_overlap():
    assert recall_at_k([1, 2, 3], [1, 2, 3]) == 1.0


def test_recall_at_k_zero_overlap():
    assert recall_at_k([4, 5, 6], [1, 2, 3]) == 0.0


def test_recall_at_k_half_overlap():
    assert recall_at_k([1, 2, 7, 8], [1, 2, 3, 4]) == 0.5


def test_recall_at_k_empty_true_ids():
    assert recall_at_k([1, 2, 3], []) == 1.0


def _build_store(rng, dim=16, n=200):
    store = VectorStore(dim=dim)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    store.add_batch(vectors)
    return store


def test_brute_force_against_itself_has_perfect_recall():
    rng = np.random.default_rng(0)
    store = _build_store(rng)
    queries = rng.standard_normal((10, 16)).astype(np.float32)

    results = evaluate(store, store, queries, k=5)

    assert results["mean_recall"] == 1.0


def test_evaluate_returns_expected_keys_and_positive_latencies():
    rng = np.random.default_rng(1)
    store = _build_store(rng)
    queries = rng.standard_normal((10, 16)).astype(np.float32)

    results = evaluate(store, store, queries, k=5)

    assert set(results.keys()) == {"mean_recall", "p50_ms", "p95_ms", "p99_ms", "mean_ms"}
    for key in ("p50_ms", "p95_ms", "p99_ms", "mean_ms"):
        assert isinstance(results[key], float)
        assert results[key] >= 0.0


class DroppingIndex:
    def __init__(self, store):
        self._store = store

    def search(self, query, k):
        results = self._store.search(query, k)
        return results[:-1]


def test_degraded_index_reports_recall_below_one():
    rng = np.random.default_rng(2)
    store = _build_store(rng)
    queries = rng.standard_normal((10, 16)).astype(np.float32)
    degraded = DroppingIndex(store)

    results = evaluate(degraded, store, queries, k=5)

    assert results["mean_recall"] < 1.0
