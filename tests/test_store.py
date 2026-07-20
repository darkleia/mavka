import numpy as np
import pytest

from mavka import VectorStore


def test_empty_store_has_count_zero():
    store = VectorStore(dim=4)
    assert store.count == 0
    assert len(store) == 0


def test_add_returns_sequential_ids():
    store = VectorStore(dim=3)
    assert store.add([1.0, 2.0, 3.0]) == 0
    assert store.add([4.0, 5.0, 6.0]) == 1
    assert store.add([7.0, 8.0, 9.0]) == 2


def test_count_and_len_track_additions():
    store = VectorStore(dim=2)
    for i in range(5):
        store.add([float(i), float(i)])
        assert store.count == i + 1
        assert len(store) == i + 1


def test_get_returns_same_values():
    store = VectorStore(dim=3)
    id0 = store.add([1.0, 2.0, 3.0])
    id1 = store.add(np.array([4.0, 5.0, 6.0]))
    np.testing.assert_array_equal(store.get(id0), np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(store.get(id1), np.array([4.0, 5.0, 6.0], dtype=np.float32))


def test_add_batch_returns_correct_ids_and_count():
    store = VectorStore(dim=2)
    store.add([0.0, 0.0])
    ids = store.add_batch([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    assert ids == [1, 2, 3]
    assert store.count == 4


def test_add_batch_values_retrievable():
    store = VectorStore(dim=2)
    vectors = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    ids = store.add_batch(vectors)
    for id_, expected in zip(ids, vectors):
        np.testing.assert_array_equal(store.get(id_), expected.astype(np.float32))


def test_growth_beyond_initial_capacity():
    store = VectorStore(dim=4, initial_capacity=4)
    n = 10
    for i in range(n):
        store.add([float(i)] * 4)
    assert store.count == n
    for i in range(n):
        np.testing.assert_array_equal(store.get(i), np.array([float(i)] * 4, dtype=np.float32))


def test_growth_via_add_batch():
    store = VectorStore(dim=2, initial_capacity=2)
    vectors = [[float(i), float(i)] for i in range(20)]
    ids = store.add_batch(vectors)
    assert ids == list(range(20))
    assert store.count == 20
    for i in ids:
        np.testing.assert_array_equal(store.get(i), np.array([float(i), float(i)], dtype=np.float32))


def test_wrong_length_vector_raises():
    store = VectorStore(dim=3)
    with pytest.raises(ValueError):
        store.add([1.0, 2.0])


def test_wrong_shape_batch_raises():
    store = VectorStore(dim=3)
    with pytest.raises(ValueError):
        store.add_batch([[1.0, 2.0], [3.0, 4.0]])


def test_out_of_range_get_raises():
    store = VectorStore(dim=3)
    store.add([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        store.get(1)
    with pytest.raises(ValueError):
        store.get(-1)
