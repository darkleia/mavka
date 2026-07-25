import numpy as np
import pytest

from mavka.eval import evaluate
from mavka.ivf import IVFIndex
from mavka.store import VectorStore


def _random_vectors(n, dim, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def _blob_vectors(n_blobs, n_per_blob, dim, std, seed):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_blobs, dim)) * 5
    points = [center + rng.normal(scale=std, size=(n_per_blob, dim)) for center in centers]
    return np.concatenate(points).astype(np.float32)


def test_search_returns_plausible_neighbors():
    dim = 16
    vectors = _random_vectors(500, dim, seed=0)

    index = IVFIndex(dim=dim, n_lists=20)
    index.train(vectors)
    index.add_batch(vectors)

    query = vectors[0]
    results = index.search(query, k=5, nprobe=20)

    assert results[0][0] == 0
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_nprobe_equals_n_lists_matches_brute_force():
    dim = 16
    n_lists = 10
    vectors = _random_vectors(400, dim, seed=1)

    brute = VectorStore(dim=dim)
    brute.add_batch(vectors)

    ivf = IVFIndex(dim=dim, n_lists=n_lists)
    ivf.train(vectors)
    ivf.add_batch(vectors)

    queries = _random_vectors(10, dim, seed=2)
    for query in queries:
        brute_results = brute.search(query, k=10)
        ivf_results = ivf.search(query, k=10, nprobe=n_lists)
        assert [id_ for id_, _ in ivf_results] == [id_ for id_, _ in brute_results]
        for (_, ivf_score), (_, brute_score) in zip(ivf_results, brute_results):
            assert ivf_score == pytest.approx(brute_score, abs=1e-5)


def test_recall_improves_with_nprobe():
    # Blob-clustered data, since real embeddings have cluster structure that
    # IVF exploits; uniformly random vectors have none, so no nprobe short of
    # "probe everything" would reach high recall.
    dim = 32
    n_lists = 50
    vectors = _blob_vectors(n_blobs=30, n_per_blob=100, dim=dim, std=1.0, seed=3)
    queries = _random_vectors(50, dim, seed=4)

    brute = VectorStore(dim=dim)
    brute.add_batch(vectors)

    ivf = IVFIndex(dim=dim, n_lists=n_lists)
    ivf.train(vectors)
    ivf.add_batch(vectors)

    recalls = {}
    for nprobe in (1, 5, 20):
        ivf.nprobe = nprobe
        result = evaluate(ivf, brute, queries, k=10)
        recalls[nprobe] = result["mean_recall"]

    print(f"\nrecall@10 by nprobe: {recalls}")

    assert recalls[1] <= recalls[5] <= recalls[20]
    assert recalls[20] >= 0.95


def test_every_vector_in_exactly_one_bucket():
    dim = 8
    vectors = _random_vectors(200, dim, seed=5)

    ivf = IVFIndex(dim=dim, n_lists=15)
    ivf.train(vectors)
    ids = ivf.add_batch(vectors)

    all_bucket_ids = []
    for bucket in ivf._inverted_lists:
        all_bucket_ids.extend(bucket)

    assert sorted(all_bucket_ids) == sorted(ids)
    assert len(all_bucket_ids) == len(set(all_bucket_ids))


def test_add_and_search_before_train_raise():
    ivf = IVFIndex(dim=8, n_lists=5)
    with pytest.raises(ValueError):
        ivf.add([1.0] * 8)
    with pytest.raises(ValueError):
        ivf.search([1.0] * 8, k=1)


def test_nprobe_out_of_range_is_clamped_not_crashed():
    dim = 8
    vectors = _random_vectors(100, dim, seed=6)
    ivf = IVFIndex(dim=dim, n_lists=10)
    ivf.train(vectors)
    ivf.add_batch(vectors)

    query = vectors[0]
    result_high = ivf.search(query, k=5, nprobe=1000)
    result_zero = ivf.search(query, k=5, nprobe=0)
    result_negative = ivf.search(query, k=5, nprobe=-5)

    assert len(result_high) == 5
    assert len(result_zero) >= 1
    assert len(result_negative) >= 1


def test_determinism_same_seed_same_results():
    dim = 16
    vectors = _random_vectors(300, dim, seed=7)

    ivf_a = IVFIndex(dim=dim, n_lists=10, seed=42)
    ivf_a.train(vectors)
    ivf_a.add_batch(vectors)

    ivf_b = IVFIndex(dim=dim, n_lists=10, seed=42)
    ivf_b.train(vectors)
    ivf_b.add_batch(vectors)

    np.testing.assert_array_equal(ivf_a._centroids, ivf_b._centroids)

    query = _random_vectors(1, dim, seed=8)[0]
    results_a = ivf_a.search(query, k=5, nprobe=5)
    results_b = ivf_b.search(query, k=5, nprobe=5)
    assert results_a == results_b
