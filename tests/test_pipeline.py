import numpy as np
import pytest

from mavka.adapter import SyntheticWorldModel
from mavka.eval.sweep import evaluate
from mavka.pipeline import Pipeline, build_pipeline_from_adapter
from mavka.storage.segments import SegmentStore
from mavka.index.flat import VectorStore, normalize


def test_end_to_end_write_read():
    dim = 16
    action_dim = 4
    adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=1)
    pipeline = build_pipeline_from_adapter(adapter, n_episodes=5, episode_length=20)

    query_id = 42
    query_record = pipeline.get(query_id)
    results = pipeline.recall(query_record.z, k=5)

    assert results[0][0] == query_id
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_id_consistency_between_log_and_index():
    dim = 16
    action_dim = 4
    adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=2)
    pipeline = build_pipeline_from_adapter(adapter, n_episodes=5, episode_length=20)

    for query_id in (5, 30, 77, 99):
        query_z = pipeline.get(query_id).z
        results = pipeline.recall(query_z, k=10)

        assert len(results) > 0
        for id_, score in results:
            record = pipeline.get(id_)
            expected_score = float(np.dot(normalize(query_z), record.z))
            assert score == pytest.approx(expected_score, abs=1e-5)


def test_store_and_index_agree_on_count():
    dim = 16
    action_dim = 4
    n_episodes = 5
    episode_length = 20

    adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=3)
    pipeline = build_pipeline_from_adapter(
        adapter, n_episodes=n_episodes, episode_length=episode_length
    )

    n = n_episodes * episode_length
    assert pipeline.count == n
    assert pipeline._log.count == n
    assert pipeline._index.count == n


def test_persistence_round_trip(tmp_path):
    dim = 8
    action_dim = 2
    store_path = tmp_path / "store"

    pipeline = Pipeline(
        dim=dim, action_dim=action_dim, index=VectorStore(dim=dim), store_path=str(store_path)
    )

    rng = np.random.default_rng(4)
    observed = []
    for _ in range(15):
        z = rng.standard_normal(dim).astype(np.float32)
        action = rng.standard_normal(action_dim).astype(np.float32)
        id_ = pipeline.observe(z=z, action=action, z_next=None, pred_err=0.1, episode_id=0)
        observed.append((id_, normalize(z)))

    pipeline.close()

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
    pipeline = build_pipeline_from_adapter(
        adapter, n_episodes=n_episodes, episode_length=episode_length
    )

    reference_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=0)
    brute_pipeline = build_pipeline_from_adapter(
        reference_adapter,
        n_episodes=n_episodes,
        episode_length=episode_length,
        index=VectorStore(dim=dim),
    )

    pipeline._index.nprobe = nprobe
    queries = np.stack([pipeline.get(i).z for i in range(0, pipeline.count, 10)])
    result = evaluate(pipeline._index, brute_pipeline._index, queries, k=k)

    print(f"\nIVF pipeline recall@{k} (nprobe={nprobe}): {result['mean_recall']:.4f}")
    assert result["mean_recall"] >= 0.9


def test_swapability_with_brute_force_index():
    dim = 8
    action_dim = 2
    pipeline = Pipeline(dim=dim, action_dim=action_dim, index=VectorStore(dim=dim))

    rng = np.random.default_rng(5)
    ids = []
    for _ in range(20):
        z = rng.standard_normal(dim).astype(np.float32)
        action = rng.standard_normal(action_dim).astype(np.float32)
        id_ = pipeline.observe(z=z, action=action, z_next=None, episode_id=0)
        ids.append(id_)

    assert pipeline.count == 20

    query_id = ids[7]
    query_z = pipeline.get(query_id).z
    results = pipeline.recall(query_z, k=3)

    assert results[0][0] == query_id
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)
