import numpy as np
import pytest

from mavka.ivf import IVFIndex, IVFState


def _random_vectors(n, dim, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def test_fresh_index_is_untrained_and_rejects_calls():
    ivf = IVFIndex(dim=8, n_lists=5)

    assert ivf.is_trained is False
    assert ivf.state == IVFState.UNTRAINED

    with pytest.raises(RuntimeError):
        ivf.add([1.0] * 8)
    with pytest.raises(RuntimeError):
        ivf.add_batch([[1.0] * 8])
    with pytest.raises(RuntimeError):
        ivf.search([1.0] * 8, k=1)


def test_train_then_empty_search_returns_empty_list():
    vectors = _random_vectors(50, 8, seed=0)
    ivf = IVFIndex(dim=8, n_lists=5)
    ivf.train(vectors)

    assert ivf.is_trained is True
    assert ivf.state == IVFState.TRAINED
    assert ivf.count == 0

    result = ivf.search(vectors[0], k=5)
    assert result == []


def test_second_train_call_after_adds_raises():
    vectors = _random_vectors(50, 8, seed=1)
    ivf = IVFIndex(dim=8, n_lists=5)
    ivf.train(vectors)
    ivf.add_batch(vectors)

    with pytest.raises(RuntimeError):
        ivf.train(vectors)


def test_second_train_call_without_adds_also_raises():
    vectors = _random_vectors(50, 8, seed=2)
    ivf = IVFIndex(dim=8, n_lists=5)
    ivf.train(vectors)

    with pytest.raises(RuntimeError):
        ivf.train(vectors)


def test_incremental_batches_match_bulk_batch():
    dim = 16
    n_lists = 10
    vectors = _random_vectors(1000, dim, seed=3)
    queries = _random_vectors(20, dim, seed=4)

    bulk = IVFIndex(dim=dim, n_lists=n_lists, seed=7)
    bulk.train(vectors)
    bulk.add_batch(vectors)

    incremental = IVFIndex(dim=dim, n_lists=n_lists, seed=7)
    incremental.train(vectors)
    for i in range(0, 1000, 100):
        incremental.add_batch(vectors[i : i + 100])

    assert bulk._inverted_lists == incremental._inverted_lists

    for query in queries:
        bulk_results = bulk.search(query, k=10, nprobe=n_lists)
        incremental_results = incremental.search(query, k=10, nprobe=n_lists)
        assert bulk_results == incremental_results


def test_interleaved_add_and_search_sees_new_vectors():
    dim = 8
    vectors = _random_vectors(300, dim, seed=5)
    train_sample = vectors[:100]

    ivf = IVFIndex(dim=dim, n_lists=10)
    ivf.train(train_sample)

    first_batch = vectors[100:150]
    ivf.add_batch(first_batch)
    assert ivf.count == 50

    query = first_batch[0]
    results_before = ivf.search(query, k=5, nprobe=10)
    assert results_before[0][0] == 0

    second_batch = vectors[150:200]
    ivf.add_batch(second_batch)
    assert ivf.count == 100

    new_query = second_batch[0]
    results_after = ivf.search(new_query, k=5, nprobe=10)
    assert results_after[0][0] == 50


def test_state_transitions():
    vectors = _random_vectors(50, 8, seed=6)
    ivf = IVFIndex(dim=8, n_lists=5)

    assert ivf.state == IVFState.UNTRAINED
    assert ivf.is_trained is False

    ivf.train(vectors)

    assert ivf.state == IVFState.TRAINED
    assert ivf.is_trained is True

    ivf.add_batch(vectors)

    assert ivf.state == IVFState.TRAINED
    assert ivf.is_trained is True


def test_train_with_fewer_samples_than_n_lists_reduces_cluster_count():
    dim = 8
    n_lists = 20
    vectors = _random_vectors(5, dim, seed=7)

    ivf = IVFIndex(dim=dim, n_lists=n_lists)
    ivf.train(vectors)

    assert ivf.is_trained is True
    assert ivf._n_lists_actual == 5
    assert len(ivf._inverted_lists) == 5
    assert ivf._centroids.shape == (5, dim)
