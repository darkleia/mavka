import numpy as np
import pytest

from mavka.config import MavkaConfig
from mavka.index.flat import FlatIndex
from mavka.lifecycle.eviction import EvictionPolicy
from mavka.lifecycle.feedback import FeedbackBuffer
from mavka.lifecycle.maintenance import MaintenanceWorker
from mavka.memory import Memory

NOW_NS = 1_000_000_000_000


def _rand(dim, seed):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def test_maintenance_off_is_byte_identical_to_unwired_memory():
    dim = 8
    config = MavkaConfig(dim=dim)
    rng = np.random.default_rng(0)
    zs = [rng.standard_normal(dim).astype(np.float32) for _ in range(30)]

    plain = Memory(config, index=FlatIndex(dim=dim), action_scale=0.0)
    wired_off = Memory(
        config,
        index=FlatIndex(dim=dim),
        action_scale=0.0,
        eviction_policy=None,
        feedback_buffer=None,
        use_tiering=False,
    )

    for z in zs:
        assert plain.observe(z=z, action=None, z_next=None, episode_id=0) == wired_off.observe(
            z=z, action=None, z_next=None, episode_id=0
        )

    for seed in range(5):
        query = rng.standard_normal(dim).astype(np.float32)
        assert plain.recall(query, k=5) == wired_off.recall(query, k=5)


def test_disabled_worker_touches_nothing():
    dim = 8
    config = MavkaConfig(dim=dim)
    policy = EvictionPolicy()
    buffer = FeedbackBuffer()
    memory = Memory(
        config, index=FlatIndex(dim=dim), eviction_policy=policy, feedback_buffer=buffer
    )
    for i in range(10):
        memory.observe(z=_rand(dim, i), action=None, z_next=None, episode_id=0)

    token = buffer.record_used([0, 1])
    buffer.tag_outcome(token, helped=True)

    worker = MaintenanceWorker(
        memory,
        eviction_policy=policy,
        feedback_buffer=buffer,
        capacity=1,
        compaction_threshold=1,
        enabled=False,
    )
    report = worker.step(now_ns=NOW_NS)

    assert report == {
        "feedback_events_applied": 0,
        "compaction": None,
        "eviction": None,
        "migration": None,
    }
    assert memory.count == 10
    # the buffer's pending event is untouched -- drain still finds it.
    assert len(buffer.drain()) == 2


def test_feedback_loop_closes_via_recall_hook_and_worker():
    dim = 8
    config = MavkaConfig(dim=dim)
    policy = EvictionPolicy()
    buffer = FeedbackBuffer()
    memory = Memory(
        config, index=FlatIndex(dim=dim), eviction_policy=policy, feedback_buffer=buffer
    )

    id_helpful = memory.observe(z=_rand(dim, 0), action=None, z_next=None, episode_id=0)
    id_unhelpful = memory.observe(z=_rand(dim, 1), action=None, z_next=None, episode_id=0)

    # Hot-path hook: recall() appends to the buffer automatically and
    # exposes the token for the harness to tag once the verdict is known.
    query = memory.get(id_helpful).z
    results = memory.recall(query, k=1)
    assert results[0][0] == id_helpful
    assert memory.last_feedback_token is not None
    buffer.tag_outcome(memory.last_feedback_token, helped=True)

    query2 = memory.get(id_unhelpful).z
    results2 = memory.recall(query2, k=1)
    assert results2[0][0] == id_unhelpful
    buffer.tag_outcome(memory.last_feedback_token, helped=False)

    worker = MaintenanceWorker(memory, eviction_policy=policy, feedback_buffer=buffer)
    report = worker.step(now_ns=NOW_NS)

    assert report["feedback_events_applied"] == 2
    scores = policy.compute_keep_scores(
        [memory.get(id_helpful), memory.get(id_unhelpful)], now_ns=NOW_NS
    )
    assert scores[id_helpful] > scores[id_unhelpful]


def test_gated_off_recall_never_calls_record_used():
    # feedback_buffer=None (the default) -- recall must not touch any
    # buffer at all, and must not raise.
    dim = 8
    config = MavkaConfig(dim=dim)
    memory = Memory(config, index=FlatIndex(dim=dim))
    memory.observe(z=_rand(dim, 0), action=None, z_next=None, episode_id=0)

    results = memory.recall(_rand(dim, 0), k=1)
    assert len(results) == 1
    assert memory.last_feedback_token is None


def test_record_retrieval_feedback_only_called_via_drain_feedback():
    calls = []

    class SpyPolicy:
        def record_retrieval_feedback(self, id, helped, now_ns=None):
            calls.append((id, helped))

        def compute_keep_scores(self, records, now_ns=None):
            return {r.id: 0.0 for r in records}

        def evict_to_capacity(self, log, index, index_factory, capacity, graph=None, now_ns=None):
            raise AssertionError("not exercised in this test")

    dim = 8
    config = MavkaConfig(dim=dim)
    spy = SpyPolicy()
    buffer = FeedbackBuffer()
    memory = Memory(config, index=FlatIndex(dim=dim), eviction_policy=spy, feedback_buffer=buffer)

    id_ = memory.observe(z=_rand(dim, 0), action=None, z_next=None, episode_id=0)
    memory.recall(memory.get(id_).z, k=1)
    buffer.tag_outcome(memory.last_feedback_token, helped=True)

    # The hot path (observe + recall) must never have called the policy.
    assert calls == []

    worker = MaintenanceWorker(memory, eviction_policy=spy, feedback_buffer=buffer)
    worker.step()

    # Only drain_feedback (inside worker.step()) may call it.
    assert calls == [(id_, True)]


def test_compaction_and_eviction_produce_nonzero_activity():
    dim = 8
    config = MavkaConfig(dim=dim)
    policy = EvictionPolicy()
    memory = Memory(config, index=FlatIndex(dim=dim), eviction_policy=policy)

    rng = np.random.default_rng(3)
    for i in range(40):
        z = rng.standard_normal(dim).astype(np.float32)
        memory.observe(z=z, action=None, z_next=None, pred_err=float(i % 5) * 0.1, episode_id=0)

    # A near-duplicate of an existing record, for compaction to merge.
    anchor = memory.get(0).z
    dup_z = anchor + rng.standard_normal(dim).astype(np.float32) * 0.0001
    memory.observe(z=dup_z, action=None, z_next=None, pred_err=0.05, episode_id=0)

    assert memory.count == 41

    worker = MaintenanceWorker(memory, eviction_policy=policy, capacity=30, compaction_threshold=40)
    report = worker.step(now_ns=NOW_NS)

    assert report["compaction"] is not None
    assert report["compaction"]["records_merged"] >= 1
    assert report["eviction"] is not None
    assert len(report["eviction"]["evicted_ids"]) > 0
    assert memory.count == 30


def test_compaction_and_eviction_gated_off_below_threshold():
    dim = 8
    config = MavkaConfig(dim=dim)
    policy = EvictionPolicy()
    memory = Memory(config, index=FlatIndex(dim=dim), eviction_policy=policy)
    for i in range(5):
        memory.observe(z=_rand(dim, i), action=None, z_next=None, episode_id=0)

    worker = MaintenanceWorker(memory, eviction_policy=policy, capacity=100, compaction_threshold=100)
    report = worker.step(now_ns=NOW_NS)

    assert report["compaction"] is None
    # eviction_policy.evict_to_capacity itself no-ops under capacity, but
    # since eviction_policy and capacity are both set the stage still
    # "ran" -- report a real (empty) result rather than None.
    assert report["eviction"]["evicted_ids"] == []
    assert memory.count == 5


def test_migration_produces_nonzero_activity():
    dim = 8
    config = MavkaConfig(dim=dim)
    memory = Memory(config, action_scale=0.0, use_tiering=True, hot_capacity=10)

    rng = np.random.default_rng(4)
    for i in range(30):
        z = rng.standard_normal(dim).astype(np.float32)
        memory.observe(z=z, action=None, z_next=None, pred_err=float(i % 5) * 0.1, episode_id=0)

    assert memory._tiered.hot_count == 30
    assert memory._tiered.cold_count == 0

    worker = MaintenanceWorker(memory)
    report = worker.step(now_ns=NOW_NS)

    assert report["migration"] is not None
    assert len(report["migration"]["migrated_ids"]) > 0
    assert memory._tiered.hot_count == 10
    assert memory._tiered.cold_count == 20

    # recall still transparently finds a migrated (now-cold) record.
    migrated_id = report["migration"]["migrated_ids"][0]
    query = memory.get(migrated_id).z
    results = memory.recall(query, k=1)
    assert results[0][0] == migrated_id


def test_compact_refuses_action_conditioned_memory():
    dim = 8
    action_dim = 2
    config = MavkaConfig(dim=dim, action_dim=action_dim)
    memory = Memory(config, index=FlatIndex(dim=dim + action_dim), action_scale=1.0)
    with pytest.raises(ValueError):
        memory.compact()


def test_compact_and_evict_refuse_tiered_memory():
    config = MavkaConfig(dim=8)
    memory = Memory(config, use_tiering=True)
    with pytest.raises(ValueError):
        memory.compact()
    with pytest.raises(ValueError):
        memory.evict(5)


def test_migrate_refuses_non_tiered_memory():
    config = MavkaConfig(dim=8)
    memory = Memory(config, index=FlatIndex(dim=8))
    with pytest.raises(ValueError):
        memory.migrate()


def test_evict_refuses_without_eviction_policy():
    config = MavkaConfig(dim=8)
    memory = Memory(config, index=FlatIndex(dim=8))
    with pytest.raises(ValueError):
        memory.evict(5)


def test_determinism():
    def run():
        dim = 8
        config = MavkaConfig(dim=dim)
        policy = EvictionPolicy()
        buffer = FeedbackBuffer()
        memory = Memory(
            config, index=FlatIndex(dim=dim), eviction_policy=policy, feedback_buffer=buffer
        )
        for i in range(20):
            memory.observe(
                z=_rand(dim, i), action=None, z_next=None, pred_err=float(i % 5) * 0.1, episode_id=0
            )
        for i in range(5):
            memory.recall(memory.get(i).z, k=2)
            buffer.tag_outcome(memory.last_feedback_token, helped=(i % 2 == 0))

        worker = MaintenanceWorker(memory, eviction_policy=policy, capacity=15, compaction_threshold=20)
        report = worker.step(now_ns=NOW_NS)
        return report["feedback_events_applied"], sorted(report["eviction"]["evicted_ids"])

    assert run() == run()
