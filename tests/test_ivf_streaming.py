import numpy as np

from mavka.eval.sweep import evaluate
from mavka.index.ivf import IVFIndex
from mavka.index.flat import VectorStore


def _random_vectors(n, dim, seed, center=None):
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    if center is not None:
        vectors = vectors + np.asarray(center, dtype=np.float32)
    return vectors.astype(np.float32)


def _make_multiblob(n_total, dim, n_blobs, offset_scale, seed):
    rng = np.random.default_rng(seed)
    per_blob = n_total // n_blobs
    parts = []
    for _ in range(n_blobs):
        center = rng.standard_normal(dim) * offset_scale
        parts.append(rng.standard_normal((per_blob, dim)).astype(np.float32) + center)
    return np.concatenate(parts).astype(np.float32)


def test_streaming_equals_bulk():
    dim = 16
    n_lists = 20
    train_vectors = _random_vectors(500, dim, seed=0)
    stream_vectors = _random_vectors(10_000, dim, seed=1)
    queries = _random_vectors(20, dim, seed=2)

    bulk = IVFIndex(dim=dim, n_lists=n_lists, seed=7)
    bulk.train(train_vectors)
    bulk.add_batch(stream_vectors)

    streamed = IVFIndex(dim=dim, n_lists=n_lists, seed=7)
    streamed.train(train_vectors)
    for i in range(0, 10_000, 100):
        streamed.add_batch(stream_vectors[i : i + 100])

    assert bulk._inverted_lists == streamed._inverted_lists

    for query in queries:
        bulk_results = bulk.search(query, k=10, nprobe=n_lists)
        streamed_results = streamed.search(query, k=10, nprobe=n_lists)
        assert bulk_results == streamed_results


def test_balance_on_matched_distribution():
    dim = 16
    n_lists = 20
    vectors = _random_vectors(2000, dim, seed=3)

    ivf = IVFIndex(dim=dim, n_lists=n_lists)
    ivf.train(vectors)
    ivf.add_batch(vectors)

    factor = ivf.imbalance_factor()
    print(f"\nmatched-distribution imbalance_factor: {factor:.3f}")
    assert factor < 2.0


def test_drift_is_detected():
    dim = 16
    n_lists = 20
    train_vectors = _random_vectors(2000, dim, seed=4)

    ivf = IVFIndex(dim=dim, n_lists=n_lists)
    ivf.train(train_vectors)
    ivf.add_batch(train_vectors)

    baseline_imbalance = ivf.imbalance_factor()
    baseline_drift = ivf.drift_score()
    assert ivf.needs_retrain() is False

    drifted_vectors = _make_multiblob(2000, dim, n_blobs=5, offset_scale=6.0, seed=8)
    ivf.add_batch(drifted_vectors)

    drifted_imbalance = ivf.imbalance_factor()
    drifted_drift = ivf.drift_score()

    print(f"\nimbalance_factor: {baseline_imbalance:.3f} -> {drifted_imbalance:.3f}")
    print(f"drift_score: {baseline_drift:.3f} -> {drifted_drift:.3f}")

    assert drifted_imbalance > baseline_imbalance * 2
    assert ivf.needs_retrain() is True


def test_recall_degrades_under_drift():
    dim = 16
    n_lists = 20
    nprobe = 3
    k = 10

    train_vectors = _random_vectors(2000, dim, seed=6)

    matched_ivf = IVFIndex(dim=dim, n_lists=n_lists)
    matched_ivf.train(train_vectors)
    matched_ivf.add_batch(train_vectors)
    matched_ivf.nprobe = nprobe
    matched_ground_truth = VectorStore(dim=dim)
    matched_ground_truth.add_batch(train_vectors)
    matched_queries = _random_vectors(50, dim, seed=7)
    matched_result = evaluate(matched_ivf, matched_ground_truth, matched_queries, k=k)

    drifted_ivf = IVFIndex(dim=dim, n_lists=n_lists)
    drifted_ivf.train(train_vectors)
    drifted_vectors = _make_multiblob(2000, dim, n_blobs=5, offset_scale=6.0, seed=8)
    drifted_ivf.add_batch(drifted_vectors)
    drifted_ivf.nprobe = nprobe
    drifted_ground_truth = VectorStore(dim=dim)
    drifted_ground_truth.add_batch(drifted_vectors)
    drifted_queries = _make_multiblob(50, dim, n_blobs=5, offset_scale=6.0, seed=9)
    drifted_result = evaluate(drifted_ivf, drifted_ground_truth, drifted_queries, k=k)

    print(
        f"\nrecall@nprobe={nprobe}: matched={matched_result['mean_recall']:.3f} "
        f"drifted={drifted_result['mean_recall']:.3f}"
    )

    assert drifted_result["mean_recall"] < matched_result["mean_recall"] - 0.05


def test_bucket_sizes_sums_to_count_and_no_duplicates():
    dim = 8
    n_lists = 10
    vectors = _random_vectors(500, dim, seed=10)

    ivf = IVFIndex(dim=dim, n_lists=n_lists)
    ivf.train(vectors)
    ids = ivf.add_batch(vectors)

    sizes = ivf.bucket_sizes()
    assert int(sizes.sum()) == ivf.count
    assert int(sizes.sum()) == len(ids)

    all_bucket_ids = []
    for bucket in ivf._inverted_lists:
        all_bucket_ids.extend(bucket)

    assert sorted(all_bucket_ids) == sorted(ids)
    assert len(all_bucket_ids) == len(set(all_bucket_ids))


def test_needs_retrain_does_not_mutate_index():
    dim = 8
    n_lists = 10
    vectors = _random_vectors(500, dim, seed=11)
    queries = _random_vectors(10, dim, seed=12)

    ivf = IVFIndex(dim=dim, n_lists=n_lists)
    ivf.train(vectors)
    ivf.add_batch(vectors)

    count_before = ivf.count
    buckets_before = [list(bucket) for bucket in ivf._inverted_lists]
    results_before = [ivf.search(q, k=5, nprobe=n_lists) for q in queries]

    for _ in range(5):
        ivf.needs_retrain()

    assert ivf.count == count_before
    assert [list(bucket) for bucket in ivf._inverted_lists] == buckets_before
    assert [ivf.search(q, k=5, nprobe=n_lists) for q in queries] == results_before
