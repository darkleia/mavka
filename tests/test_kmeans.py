import time

import numpy as np

from mavka.index.kmeans import assign, kmeans


def _make_blobs(centers, n_per_blob, std, seed):
    rng = np.random.default_rng(seed)
    points = []
    labels = []
    for i, center in enumerate(centers):
        offsets = rng.normal(scale=std, size=(n_per_blob, len(center)))
        points.append(np.asarray(center, dtype=np.float32) + offsets)
        labels.extend([i] * n_per_blob)
    return np.concatenate(points).astype(np.float32), np.array(labels)


def test_recovers_obvious_clusters():
    centers = [[0, 0], [20, 0], [0, 20], [20, 20]]
    vectors, true_labels = _make_blobs(centers, n_per_blob=50, std=0.5, seed=1)

    _, assignments = kmeans(vectors, k=4, seed=0)

    for blob_id in range(len(centers)):
        blob_assignments = assignments[true_labels == blob_id]
        assert np.all(blob_assignments == blob_assignments[0])

    representative_labels = [assignments[true_labels == b][0] for b in range(len(centers))]
    assert len(set(representative_labels)) == len(centers)


def test_output_shapes_are_correct():
    rng = np.random.default_rng(2)
    vectors = rng.standard_normal((100, 8)).astype(np.float32)

    centroids, assignments = kmeans(vectors, k=5, seed=0)

    assert centroids.shape == (5, 8)
    assert assignments.shape == (100,)
    assert np.all(assignments >= 0)
    assert np.all(assignments < 5)


def test_determinism_same_seed_identical_results():
    rng = np.random.default_rng(3)
    vectors = rng.standard_normal((200, 6)).astype(np.float32)

    centroids_a, assignments_a = kmeans(vectors, k=4, seed=42)
    centroids_b, assignments_b = kmeans(vectors, k=4, seed=42)

    np.testing.assert_array_equal(centroids_a, centroids_b)
    np.testing.assert_array_equal(assignments_a, assignments_b)


def test_different_seeds_may_differ():
    rng = np.random.default_rng(4)
    vectors = rng.standard_normal((200, 6)).astype(np.float32)

    _, assignments_a = kmeans(vectors, k=4, seed=1)
    _, assignments_b = kmeans(vectors, k=4, seed=2)

    assert not np.array_equal(assignments_a, assignments_b)


def test_assign_returns_nearest_centroid():
    centroids = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    points = np.array(
        [
            [0.1, 0.1],
            [9.9, 9.8],
            [-1.0, -1.0],
            [10.5, 10.5],
        ],
        dtype=np.float32,
    )

    result = assign(points, centroids)

    assert result.tolist() == [0, 1, 0, 1]


def test_convergence_extra_iterations_do_not_change_result():
    centers = [[0, 0], [20, 0], [0, 20], [20, 20]]
    vectors, _ = _make_blobs(centers, n_per_blob=40, std=0.5, seed=5)

    centroids_short, assignments_short = kmeans(vectors, k=4, n_iters=25, seed=0)
    centroids_long, assignments_long = kmeans(vectors, k=4, n_iters=200, seed=0)

    np.testing.assert_array_equal(assignments_short, assignments_long)
    np.testing.assert_allclose(centroids_short, centroids_long)


def test_empty_cluster_handling_no_crash_no_nans():
    centers = [[0, 0], [30, 30]]
    vectors, _ = _make_blobs(centers, n_per_blob=25, std=0.5, seed=6)

    centroids, assignments = kmeans(vectors, k=10, seed=0)

    assert not np.any(np.isnan(centroids))
    assert centroids.shape == (10, 2)
    assert assignments.shape == (vectors.shape[0],)
    assert np.all(assignments >= 0)
    assert np.all(assignments < 10)


def test_vectorization_sanity_runs_quickly():
    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((5000, 32)).astype(np.float32)

    start = time.perf_counter()
    kmeans(vectors, k=16, n_iters=25, seed=0)
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0
