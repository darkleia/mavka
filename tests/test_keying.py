import numpy as np
import pytest

from mavka.retrieval.keying import make_key, make_keys_batch
from mavka.config import MavkaConfig
from mavka.memory import Memory
from mavka.index.flat import FlatIndex
from mavka.core.distance import normalize


def test_make_key_shape_dtype_unit_norm():
    dim = 8
    action_dim = 3
    z = np.random.default_rng(0).standard_normal(dim).astype(np.float32)
    action = np.random.default_rng(1).standard_normal(action_dim).astype(np.float32)

    key = make_key(z, action, scale=1.0, action_dim=action_dim)

    assert key.shape == (dim + action_dim,)
    assert key.dtype == np.float32
    assert np.linalg.norm(key) == pytest.approx(1.0, abs=1e-5)


def test_scale_zero_action_part_is_zero():
    dim = 8
    action_dim = 3
    z = normalize(np.random.default_rng(2).standard_normal(dim).astype(np.float32))
    action = np.random.default_rng(3).standard_normal(action_dim).astype(np.float32)

    key = make_key(z, action, scale=0.0, action_dim=action_dim)

    np.testing.assert_array_equal(key[dim:], np.zeros(action_dim, dtype=np.float32))
    np.testing.assert_allclose(key[:dim], normalize(z), atol=1e-6)


def test_action_none_returns_normalized_z():
    dim = 8
    z = np.random.default_rng(4).standard_normal(dim).astype(np.float32)

    key = make_key(z, None, scale=1.0, action_dim=None)

    np.testing.assert_allclose(key, normalize(z), atol=1e-6)
    assert key.shape == (dim,)


def test_make_keys_batch_matches_single_key_per_row():
    dim = 6
    action_dim = 2
    rng = np.random.default_rng(5)
    zs = rng.standard_normal((10, dim)).astype(np.float32)
    actions = rng.standard_normal((10, action_dim)).astype(np.float32)

    batch_keys = make_keys_batch(zs, actions, scale=1.5, action_dim=action_dim)

    for i in range(10):
        expected = make_key(zs[i], actions[i], scale=1.5, action_dim=action_dim)
        np.testing.assert_allclose(batch_keys[i], expected, atol=1e-6)


def test_action_scale_zero_retrieval_matches_appearance_only_search():
    dim = 8
    action_dim = 3
    rng = np.random.default_rng(6)

    zs = rng.standard_normal((20, dim)).astype(np.float32)
    actions = rng.standard_normal((20, action_dim)).astype(np.float32)

    config = MavkaConfig(dim=dim, action_dim=action_dim)
    memory = Memory(config, index=FlatIndex(dim=dim), action_scale=0.0)
    plain_store = FlatIndex(dim=dim)

    for i in range(20):
        memory.observe(z=zs[i], action=actions[i], z_next=None, episode_id=0)
        plain_store.add(zs[i])

    query_z = zs[3]
    query_action = actions[7]  # a different action -- must not matter at action_scale=0

    keyed_results = memory.recall(query_z, action=query_action, k=5)
    plain_results = plain_store.search(query_z, k=5)

    assert [id_ for id_, _ in keyed_results] == [id_ for id_, _ in plain_results]
    for (_, keyed_score), (_, plain_score) in zip(keyed_results, plain_results):
        assert keyed_score == pytest.approx(plain_score, abs=1e-5)
