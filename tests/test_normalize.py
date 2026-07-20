import numpy as np
import pytest

from mavka import VectorStore
from mavka.store import normalize


def test_add_stores_unit_norm_vector():
    store = VectorStore(dim=3)
    id_ = store.add([3.0, 4.0, 0.0])
    stored = store.get(id_)
    assert np.linalg.norm(stored) == pytest.approx(1.0, abs=1e-6)


def test_add_batch_normalizes_every_row():
    store = VectorStore(dim=3)
    ids = store.add_batch([[3.0, 4.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 5.0]])
    for id_ in ids:
        norm = np.linalg.norm(store.get(id_))
        assert norm == pytest.approx(1.0, abs=1e-6)


def test_scaled_vector_normalizes_to_same_stored_vector():
    store = VectorStore(dim=3)
    id_a = store.add([1.0, 2.0, 3.0])
    id_b = store.add([10.0, 20.0, 30.0])
    np.testing.assert_allclose(store.get(id_a), store.get(id_b), atol=1e-6)


def test_normalize_helper_single_vector():
    result = normalize([3.0, 4.0, 0.0])
    assert result.shape == (3,)
    assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(result, [0.6, 0.8, 0.0], atol=1e-6)


def test_normalize_helper_batch():
    result = normalize([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    assert result.shape == (2, 3)
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)


def test_normalize_zero_vector_raises():
    with pytest.raises(ValueError):
        normalize([0.0, 0.0, 0.0])


def test_normalize_batch_with_zero_row_raises():
    with pytest.raises(ValueError):
        normalize([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])


def test_add_zero_vector_raises():
    store = VectorStore(dim=3)
    with pytest.raises(ValueError):
        store.add([0.0, 0.0, 0.0])


def test_add_batch_with_zero_row_raises():
    store = VectorStore(dim=3)
    with pytest.raises(ValueError):
        store.add_batch([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])


def test_search_with_unnormalized_query_finds_stored_vector():
    store = VectorStore(dim=3)
    store.add([1.0, 0.0, 0.0])
    store.add([0.0, 1.0, 0.0])
    store.add([0.0, 0.0, 1.0])
    results = store.search([50.0, 0.0, 0.0], k=1)
    assert results[0][0] == 0
    assert results[0][1] == pytest.approx(1.0, abs=1e-6)
