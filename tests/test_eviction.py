import numpy as np
import pytest

from mavka.lifecycle.eviction import EvictionPolicy
from mavka.graph.adjacency import EDGE_TEMPORAL, AdjacencyStore
from mavka.storage.log import AppendLog
from mavka.index.flat import FlatIndex

NOW_NS = 1_000_000_000_000
SECOND_NS = 10**9


def _rand(dim, seed):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def _build_index(dim, log, ids):
    index = FlatIndex(dim=dim)
    for id_ in ids:
        index.add(log.get(id_).z)
    return index


def test_keep_score_ordering():
    dim = 8
    log = AppendLog(dim=dim)

    id_a = log.append(
        z=_rand(dim, 0), episode_id=0, pred_err=0.9, timestamp_ns=NOW_NS - 100 * SECOND_NS
    )
    id_b = log.append(
        z=_rand(dim, 1), episode_id=0, pred_err=0.01, timestamp_ns=NOW_NS - 1000 * SECOND_NS
    )

    policy = EvictionPolicy(decay_constant_s=3600.0)
    policy.record_retrieval_feedback(id_a, helped=True, now_ns=NOW_NS - 10 * SECOND_NS)

    scores = policy.compute_keep_scores([log.get(id_a), log.get(id_b)], now_ns=NOW_NS)

    assert scores[id_a] > scores[id_b]


def test_eviction_respects_capacity():
    dim = 8
    log = AppendLog(dim=dim)
    ids = [
        log.append(z=_rand(dim, i), episode_id=0, pred_err=float(i) * 0.01, timestamp_ns=NOW_NS)
        for i in range(20)
    ]
    index = _build_index(dim, log, ids)

    policy = EvictionPolicy()
    result = policy.evict_to_capacity(
        log, index, index_factory=lambda: FlatIndex(dim=dim), capacity=10, now_ns=NOW_NS
    )

    assert result["log"].count == 10
    assert set(result["evicted_ids"]) == set(ids[:10])


def test_pinning_protects_rare_surprising_record():
    dim = 8
    log = AppendLog(dim=dim)

    boring_ids = [
        log.append(z=_rand(dim, i), episode_id=0, pred_err=0.01, timestamp_ns=NOW_NS)
        for i in range(15)
    ]
    rare_id = log.append(z=_rand(dim, 999), episode_id=0, pred_err=5.0, timestamp_ns=NOW_NS)

    index = _build_index(dim, log, [*boring_ids, rare_id])

    policy = EvictionPolicy(pin_threshold=1.0)
    # Boost utility for some boring records -- "frequently useful" -- to
    # prove this still doesn't outrank the pinned rare record.
    for id_ in boring_ids[:5]:
        policy.record_retrieval_feedback(id_, helped=True, now_ns=NOW_NS)

    result = policy.evict_to_capacity(
        log, index, index_factory=lambda: FlatIndex(dim=dim), capacity=5, now_ns=NOW_NS
    )

    evicted_ids = set(result["evicted_ids"])
    assert rare_id not in evicted_ids
    rare_new_id = result["id_map"][rare_id]
    assert rare_new_id is not None
    assert result["log"].get(rare_new_id).pred_err == 5.0
    assert len(evicted_ids) > 0


def test_all_pinned_over_capacity_evicts_nothing_and_warns():
    dim = 8
    log = AppendLog(dim=dim)
    ids = [
        log.append(z=_rand(dim, i), episode_id=0, pred_err=0.5, timestamp_ns=NOW_NS)
        for i in range(10)
    ]
    for id_ in ids:
        log.pin(id_)

    index = _build_index(dim, log, ids)

    policy = EvictionPolicy()
    with pytest.warns(UserWarning):
        result = policy.evict_to_capacity(
            log, index, index_factory=lambda: FlatIndex(dim=dim), capacity=5, now_ns=NOW_NS
        )

    assert result["evicted_ids"] == []
    assert result["log"].count == 10
    assert result["stats"]["pinned_over_capacity"] == 5


def test_utility_rescues_rarely_but_usefully_retrieved_record():
    dim = 8
    log = AppendLog(dim=dim)
    old_ts = NOW_NS - 1000 * SECOND_NS

    id_useful = log.append(z=_rand(dim, 0), episode_id=0, pred_err=0.1, timestamp_ns=old_ts)
    id_never_useful = log.append(z=_rand(dim, 1), episode_id=0, pred_err=0.1, timestamp_ns=old_ts)

    policy = EvictionPolicy()
    policy.record_retrieval_feedback(id_useful, helped=True, now_ns=NOW_NS - 10 * SECOND_NS)

    scores = policy.compute_keep_scores(
        [log.get(id_useful), log.get(id_never_useful)], now_ns=NOW_NS
    )

    assert scores[id_useful] > scores[id_never_useful]


def test_reference_integrity_after_eviction():
    dim = 8
    log = AppendLog(dim=dim, action_dim=None)
    ids = [
        log.append(z=_rand(dim, i), episode_id=0, pred_err=float(i) * 0.1, timestamp_ns=NOW_NS)
        for i in range(10)
    ]

    graph = AdjacencyStore(degree=4)
    for _ in ids:
        graph.add_node()
    for i in range(9):
        graph.add_edge(ids[i], ids[i + 1], weight=1.0, edge_type=EDGE_TEMPORAL)

    index = _build_index(dim, log, ids)

    policy = EvictionPolicy()
    result = policy.evict_to_capacity(
        log,
        index,
        index_factory=lambda: FlatIndex(dim=dim),
        capacity=5,
        graph=graph,
        now_ns=NOW_NS,
    )

    new_log = result["log"]
    new_graph = result["graph"]
    for src in range(new_graph.count):
        for dst, _weight in new_graph.neighbors_of_type(src, EDGE_TEMPORAL):
            assert 0 <= dst < new_log.count
    assert result["index"].count == new_log.count

    for evicted_id in result["evicted_ids"]:
        assert result["id_map"][evicted_id] is None


def test_decay_decreases_with_age():
    dim = 8
    log = AppendLog(dim=dim)
    id_ = log.append(z=_rand(dim, 0), episode_id=0, pred_err=0.5, timestamp_ns=NOW_NS)

    policy = EvictionPolicy(decay_constant_s=100.0)
    record = log.get(id_)

    factor_fresh = policy._decay_factor(record, now_ns=NOW_NS)
    factor_later = policy._decay_factor(record, now_ns=NOW_NS + 500 * SECOND_NS)

    assert factor_fresh > factor_later
    assert factor_fresh == pytest.approx(1.0, abs=1e-6)


def test_noop_when_under_capacity():
    dim = 8
    log = AppendLog(dim=dim)
    ids = [log.append(z=_rand(dim, i), episode_id=0) for i in range(3)]
    index = _build_index(dim, log, ids)

    policy = EvictionPolicy()
    result = policy.evict_to_capacity(log, index, index_factory=lambda: FlatIndex(dim=dim), capacity=10)

    assert result["evicted_ids"] == []
    assert result["log"] is log
    assert result["log"].count == 3


def test_noop_when_empty():
    dim = 8
    log = AppendLog(dim=dim)
    index = FlatIndex(dim=dim)

    policy = EvictionPolicy()
    result = policy.evict_to_capacity(log, index, index_factory=lambda: FlatIndex(dim=dim), capacity=10)

    assert result["evicted_ids"] == []
    assert result["log"].count == 0


def test_determinism():
    def run():
        dim = 8
        log = AppendLog(dim=dim)
        ids = [
            log.append(z=_rand(dim, i), episode_id=0, pred_err=float(i % 5) * 0.1, timestamp_ns=NOW_NS)
            for i in range(20)
        ]
        index = _build_index(dim, log, ids)
        policy = EvictionPolicy()
        for id_ in ids[::3]:
            policy.record_retrieval_feedback(id_, helped=True, now_ns=NOW_NS)
        result = policy.evict_to_capacity(
            log, index, index_factory=lambda: FlatIndex(dim=dim), capacity=10, now_ns=NOW_NS
        )
        return sorted(result["evicted_ids"]), result["log"].count

    assert run() == run()
