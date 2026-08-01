import numpy as np
import pytest

from mavka.adapter import SyntheticWorldModel, generate_trajectory
from mavka.config import MavkaConfig
from mavka.core.distance import normalize
from mavka.eval.sweep import evaluate
from mavka.index.flat import FlatIndex
from mavka.memory import Memory
from mavka.storage.segments import SegmentStore


def _build_memory_from_adapter(adapter, n_episodes, episode_length, index=None):
    """Roll the adapter forward n_episodes trajectories and feed every
    experience through memory.observe() -- the Memory-based equivalent of
    the old build_pipeline_from_adapter helper, kept local to this test
    module since it's just test setup glue, not library surface.
    """
    all_steps = []
    for episode_id in range(n_episodes):
        all_steps.extend(generate_trajectory(adapter, episode_length, episode_id=episode_id))

    config = MavkaConfig(dim=adapter.dim, action_dim=adapter.action_dim)
    memory = Memory(config, index=index, action_scale=0.0)

    if hasattr(memory._index, "train") and not getattr(memory._index, "is_trained", True):
        training_sample = np.stack([step["z"] for step in all_steps])
        memory._index.train(training_sample)

    for step in all_steps:
        memory.observe(
            z=step["z"],
            action=step["action"],
            z_next=step["z_next"],
            pred_err=step["pred_err"],
            episode_id=step["episode_id"],
        )

    return memory


def test_end_to_end_write_read():
    dim = 16
    action_dim = 4
    adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=1)
    memory = _build_memory_from_adapter(adapter, n_episodes=5, episode_length=20)

    query_id = 42
    query_record = memory.get(query_id)
    results = memory.recall(query_record.z, k=5)

    assert results[0][0] == query_id
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_id_consistency_between_log_and_index():
    dim = 16
    action_dim = 4
    adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=2)
    memory = _build_memory_from_adapter(adapter, n_episodes=5, episode_length=20)

    for query_id in (5, 30, 77, 99):
        query_z = memory.get(query_id).z
        results = memory.recall(query_z, k=10)

        assert len(results) > 0
        for id_, score in results:
            record = memory.get(id_)
            expected_score = float(np.dot(normalize(query_z), record.z))
            assert score == pytest.approx(expected_score, abs=1e-5)


def test_store_and_index_agree_on_count():
    dim = 16
    action_dim = 4
    n_episodes = 5
    episode_length = 20

    adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=3)
    memory = _build_memory_from_adapter(adapter, n_episodes=n_episodes, episode_length=episode_length)

    n = n_episodes * episode_length
    assert memory.count == n
    assert memory._log.count == n
    assert memory._index.count == n


def test_persistence_round_trip(tmp_path):
    dim = 8
    action_dim = 2
    store_path = tmp_path / "store"

    config = MavkaConfig(dim=dim, action_dim=action_dim)
    memory = Memory(
        config, index=FlatIndex(dim=dim), store_path=str(store_path), action_scale=0.0
    )

    rng = np.random.default_rng(4)
    observed = []
    for _ in range(15):
        z = rng.standard_normal(dim).astype(np.float32)
        action = rng.standard_normal(action_dim).astype(np.float32)
        id_ = memory.observe(z=z, action=action, z_next=None, pred_err=0.1, episode_id=0)
        observed.append((id_, normalize(z)))

    memory.close()

    reopened = SegmentStore.open(str(store_path))
    assert reopened.count == 15

    records = list(reopened.scan())
    assert [r.id for r in records] == [id_ for id_, _ in observed]
    for (_, expected_z), record in zip(observed, records):
        np.testing.assert_allclose(record.z, expected_z, atol=1e-6)

    reopened.close()


def test_real_data_recall_quality():
    dim = 32
    action_dim = 4
    n_episodes = 30
    episode_length = 40
    k = 10
    nprobe = 10

    adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=0)
    memory = _build_memory_from_adapter(adapter, n_episodes=n_episodes, episode_length=episode_length)

    reference_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=0)
    brute_memory = _build_memory_from_adapter(
        reference_adapter,
        n_episodes=n_episodes,
        episode_length=episode_length,
        index=FlatIndex(dim=dim),
    )

    memory._index.nprobe = nprobe
    queries = np.stack([memory.get(i).z for i in range(0, memory.count, 10)])
    result = evaluate(memory._index, brute_memory._index, queries, k=k)

    print(f"\nIVF memory recall@{k} (nprobe={nprobe}): {result['mean_recall']:.4f}")
    assert result["mean_recall"] >= 0.9


def test_swapability_with_brute_force_index():
    dim = 8
    action_dim = 2
    config = MavkaConfig(dim=dim, action_dim=action_dim)
    memory = Memory(config, index=FlatIndex(dim=dim), action_scale=0.0)

    rng = np.random.default_rng(5)
    ids = []
    for _ in range(20):
        z = rng.standard_normal(dim).astype(np.float32)
        action = rng.standard_normal(action_dim).astype(np.float32)
        id_ = memory.observe(z=z, action=action, z_next=None, episode_id=0)
        ids.append(id_)

    assert memory.count == 20

    query_id = ids[7]
    query_z = memory.get(query_id).z
    results = memory.recall(query_z, k=3)

    assert results[0][0] == query_id
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)
