import numpy as np
import pytest

from mavka import VectorStore


def test_closest_result_is_itself():
    store = VectorStore(dim=3)
    store.add([1.0, 0.0, 0.0])
    store.add([0.0, 1.0, 0.0])
    store.add([0.0, 0.0, 1.0])
    results = store.search([1.0, 0.0, 0.0], k=1)
    assert results[0][0] == 0
    assert results[0][1] == pytest.approx(1.0, abs=1e-6)


def test_results_sorted_descending():
    store = VectorStore(dim=2)
    store.add([1.0, 0.0])
    store.add([0.9, 0.1])
    store.add([0.0, 1.0])
    results = store.search([1.0, 0.0], k=3)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    assert [id_ for id_, _ in results] == [0, 1, 2]


def test_matches_naive_reference():
    rng = np.random.default_rng(0)
    dim = 16
    n = 200
    store = VectorStore(dim=dim)
    vectors = rng.random((n, dim), dtype=np.float64).astype(np.float32)
    store.add_batch(vectors)
    query = rng.random(dim, dtype=np.float64).astype(np.float32)

    naive_scores = []
    for i in range(n):
        dot = 0.0
        norm_v_sq = 0.0
        norm_q_sq = 0.0
        for j in range(dim):
            v = float(vectors[i][j])
            q = float(query[j])
            dot += v * q
            norm_v_sq += v * v
            norm_q_sq += q * q
        cosine = dot / (norm_v_sq**0.5 * norm_q_sq**0.5)
        naive_scores.append((i, cosine))
    naive_scores.sort(key=lambda pair: pair[1], reverse=True)
    naive_top_k = naive_scores[:10]

    results = store.search(query, k=10)

    assert [id_ for id_, _ in results] == [id_ for id_, _ in naive_top_k]
    for (_, actual_score), (_, expected_score) in zip(results, naive_top_k):
        assert actual_score == pytest.approx(expected_score, rel=1e-4, abs=1e-5)


def test_k_greater_than_count_returns_all_ranked():
    store = VectorStore(dim=2)
    store.add([1.0, 0.0])
    store.add([0.9, 0.1])
    store.add([0.0, 1.0])
    results = store.search([1.0, 0.0], k=100)
    assert len(results) == 3
    assert [id_ for id_, _ in results] == [0, 1, 2]


def test_empty_store_returns_empty_list():
    store = VectorStore(dim=3)
    assert store.search([1.0, 2.0, 3.0], k=5) == []


def test_wrong_length_query_raises():
    store = VectorStore(dim=3)
    store.add([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        store.search([1.0, 2.0], k=1)


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_raises(k):
    store = VectorStore(dim=3)
    store.add([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        store.search([1.0, 2.0, 3.0], k=k)


def test_large_random_matches_full_argsort_reference():
    rng = np.random.default_rng(42)
    dim = 32
    n = 5000
    store = VectorStore(dim=dim)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    store.add_batch(vectors)
    query = rng.standard_normal(dim).astype(np.float32)

    unit_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    unit_query = query / np.linalg.norm(query)
    all_scores = unit_vectors @ unit_query
    expected_top10_ids = list(np.argsort(all_scores)[::-1][:10])

    results = store.search(query, k=10)
    actual_ids = [id_ for id_, _ in results]

    assert actual_ids == [int(i) for i in expected_top10_ids]
