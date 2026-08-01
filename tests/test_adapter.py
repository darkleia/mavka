import numpy as np
import pytest

from mavka.adapter import SyntheticWorldModel, generate_trajectory, populate_store
from mavka.storage.log import AppendLog
from mavka.index.flat import VectorStore, normalize


def test_synthetic_model_produces_unit_vectors_of_right_shape_and_dtype():
    dim = 16
    model = SyntheticWorldModel(dim=dim, action_dim=4, seed=0)

    obs = model.reset()
    z = model.encode(obs)
    assert z.shape == (dim,)
    assert z.dtype == np.float32
    assert np.linalg.norm(z) == pytest.approx(1.0, abs=1e-5)

    action = model.sample_action()
    z_next = model.step(z, action)
    assert z_next.shape == (dim,)
    assert z_next.dtype == np.float32
    assert np.linalg.norm(z_next) == pytest.approx(1.0, abs=1e-5)


def test_similar_inputs_produce_similar_outputs():
    dim = 32
    action_dim = 4
    model = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=0)

    z = model.encode(model.reset())
    action = model.sample_action()
    z_next = model.step(z, action)

    rng = np.random.default_rng(123)
    z_perturbed = normalize(z + rng.standard_normal(dim).astype(np.float32) * 0.02)
    z_next_perturbed = model.step(z_perturbed, action)
    dist_similar = np.linalg.norm(z_next - z_next_perturbed)

    z_random = normalize(rng.standard_normal(dim).astype(np.float32))
    action_random = rng.standard_normal(action_dim).astype(np.float32)
    z_next_random = model.step(z_random, action_random)
    dist_random = np.linalg.norm(z_next - z_next_random)

    assert dist_similar < dist_random


def test_determinism_same_seed_same_trajectory():
    model_a = SyntheticWorldModel(dim=16, action_dim=4, seed=42)
    model_b = SyntheticWorldModel(dim=16, action_dim=4, seed=42)

    steps_a = generate_trajectory(model_a, length=10, episode_id=1)
    steps_b = generate_trajectory(model_b, length=10, episode_id=1)

    for step_a, step_b in zip(steps_a, steps_b):
        np.testing.assert_array_equal(step_a["z"], step_b["z"])
        np.testing.assert_array_equal(step_a["action"], step_b["action"])
        np.testing.assert_array_equal(step_a["z_next"], step_b["z_next"])
        assert step_a["pred_err"] == step_b["pred_err"]


def test_generate_trajectory_has_correct_seq_no_and_episode_id():
    model = SyntheticWorldModel(dim=8, action_dim=None, seed=0)
    steps = generate_trajectory(model, length=15, episode_id=7)

    assert [s["seq_no"] for s in steps] == list(range(15))
    assert all(s["episode_id"] == 7 for s in steps)
    assert all(s["action"] is None for s in steps)

    for i in range(len(steps) - 1):
        np.testing.assert_array_equal(steps[i]["z_next"], steps[i + 1]["z"])


def test_populate_store_fills_append_log_correctly():
    dim = 8
    action_dim = 3
    episode_length = 5
    n_episodes = 4

    model = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=1)
    log = AppendLog(dim=dim, action_dim=action_dim)

    ids = populate_store(model, log, n_episodes=n_episodes, episode_length=episode_length)

    assert len(ids) == n_episodes * episode_length
    assert log.count == n_episodes * episode_length

    for episode_id in range(n_episodes):
        records = log.get_episode(episode_id)
        assert len(records) == episode_length
        assert [r.seq_no for r in records] == list(range(episode_length))

    # Recompute the same trajectories independently to check stored values
    # against what was actually generated (z, action, and z_next via the
    # next record in the same episode).
    reference_model = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=1)
    id_iter = iter(ids)
    for episode_id in range(n_episodes):
        steps = generate_trajectory(reference_model, length=episode_length, episode_id=episode_id)
        for i, step in enumerate(steps):
            record_id = next(id_iter)
            record = log.get(record_id)
            np.testing.assert_array_equal(record.z, step["z"])
            np.testing.assert_array_equal(record.action, step["action"])
            assert record.episode_id == episode_id
            assert record.seq_no == i

            if i < episode_length - 1:
                next_record = log.next_in_episode(record_id)
                np.testing.assert_array_equal(next_record.z, step["z_next"])


def test_end_to_end_smoke_with_vector_store():
    dim = 16
    action_dim = 4
    episode_length = 20

    model = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=0)
    store = VectorStore(dim=dim)

    ids = populate_store(model, store, n_episodes=3, episode_length=episode_length)
    assert store.count == 3 * episode_length

    query_id = ids[10]
    query_z = store.get(query_id)
    results = store.search(query_z, k=store.count)
    ranked_ids = [id_ for id_, _ in results]

    assert results[0][0] == query_id
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    prev_rank = ranked_ids.index(query_id - 1)
    next_rank = ranked_ids.index(query_id + 1)
    assert prev_rank < store.count / 2
    assert next_rank < store.count / 2
