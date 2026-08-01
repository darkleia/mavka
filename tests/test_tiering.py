import numpy as np
import pytest

from mavka.graph.adjacency import EDGE_TEMPORAL, AdjacencyStore
from mavka.core.record import FLAG_PINNED
from mavka.index.flat import FlatIndex
from mavka.storage.tiered import TieredStore

NOW_NS = 1_000_000_000_000
SECOND_NS = 10**9


def _rand(dim, seed):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def test_migration_moves_without_losing_records():
    dim = 8
    store = TieredStore(dim=dim, hot_capacity=5)
    ids = [
        store.observe(z=_rand(dim, i), pred_err=float(i) * 0.1, episode_id=0) for i in range(12)
    ]

    result = store.migrate_hot_to_cold(now_ns=NOW_NS)

    assert store.hot_count + store.cold_count == len(ids) == store.count
    assert set(result["migrated_ids"]) == store._cold_ids

    # Lowest pred_err (and thus lowest keep_score, all else equal) records
    # are the ones that moved.
    migrated = set(result["migrated_ids"])
    assert migrated == set(ids[:7])

    for id_ in ids:
        z = store.get(id_).z
        hits = store.recall(z, k=1)
        assert hits and hits[0][0] == id_


def test_ids_stable_across_migration():
    dim = 8
    store = TieredStore(dim=dim, hot_capacity=4)
    graph = AdjacencyStore(degree=4)

    ids = []
    for i in range(8):
        id_ = store.observe(z=_rand(dim, i), pred_err=0.1, episode_id=0)
        ids.append(id_)
        graph.add_node()
    for i in range(7):
        graph.add_edge(ids[i], ids[i + 1], weight=1.0, edge_type=EDGE_TEMPORAL)

    before_tiers = {id_: store.tier_of(id_) for id_ in ids}
    result = store.migrate_hot_to_cold(now_ns=NOW_NS)
    assert result["migrated_ids"]

    for id_ in ids:
        # id itself is unchanged, still resolves to the same record.
        assert store.get(id_).id == id_
        # tier assignment moved for migrated ids, stayed for the rest --
        # but the id used to look either up never changed.
        if id_ in result["migrated_ids"]:
            assert store.tier_of(id_) == "cold"
            assert before_tiers[id_] == "hot"
        else:
            assert store.tier_of(id_) == before_tiers[id_]

    # Graph edges reference the same ids and still resolve correctly --
    # no id_map / remapping was ever needed.
    for i in range(7):
        neighbors = dict(graph.neighbors_of_type(ids[i], EDGE_TEMPORAL))
        assert ids[i + 1] in neighbors


def test_pinned_records_never_migrate():
    dim = 8
    store = TieredStore(dim=dim, hot_capacity=3)
    ids = [store.observe(z=_rand(dim, i), pred_err=0.1, episode_id=0) for i in range(6)]

    pinned_id = ids[0]
    store._log.pin(pinned_id)
    assert store._log.get(pinned_id).flags & FLAG_PINNED

    result = store.migrate_hot_to_cold(now_ns=NOW_NS)

    assert pinned_id not in result["migrated_ids"]
    assert store.tier_of(pinned_id) == "hot"


def test_all_pinned_over_capacity_migrates_nothing_and_warns():
    dim = 8
    store = TieredStore(dim=dim, hot_capacity=2)
    ids = [store.observe(z=_rand(dim, i), pred_err=0.1, episode_id=0) for i in range(5)]
    for id_ in ids:
        store._log.pin(id_)

    with pytest.warns(UserWarning):
        result = store.migrate_hot_to_cold(now_ns=NOW_NS)

    assert result["migrated_ids"] == []
    assert result["stats"]["pinned_over_target"] == 3
    assert store.hot_count == 5


def test_two_tier_search_matches_exact_search():
    dim = 8
    n = 40
    k = 5
    store = TieredStore(dim=dim, hot_capacity=15, cold_n_lists=4)

    vectors = [_rand(dim, i) for i in range(n)]
    for i, v in enumerate(vectors):
        store.observe(z=v, pred_err=float(i % 7) * 0.1, episode_id=0)

    store.migrate_hot_to_cold(now_ns=NOW_NS)
    assert store.cold_count > 0 and store.hot_count > 0
    # Near-exhaustive cold search so the approximation tolerance in this
    # test is about correctness of the tiering/merge logic, not IVF recall
    # quality (that is IVFIndex's own concern, tested elsewhere).
    store._cold_index.nprobe = store._cold_index._n_lists_actual

    reference = FlatIndex(dim=dim)
    for v in vectors:
        reference.add(v)

    query = _rand(dim, 999)
    exact_top_k = {id_ for id_, _ in reference.search(query, k)}
    tiered_top_k = {id_ for id_, _ in store.recall(query, k)}

    assert tiered_top_k == exact_top_k


def test_merge_ranks_hot_and_cold_by_score():
    dim = 8
    store = TieredStore(dim=dim, hot_capacity=10, cold_n_lists=2)

    # Migration selection is keep_score-driven (pred_err/decay/utility),
    # entirely independent of vector content -- so pred_err, not vector
    # similarity to the query, is what must deterministically decide which
    # of these ends up hot vs. cold. close_z is made similar to the query
    # so that once tiering is settled, the *merge* (not the migration) is
    # what's under test: does a hot hit and a cold hit get ranked in the
    # correct score order.
    query = _rand(dim, 0)
    close_z = query + 0.001 * _rand(dim, 1)  # near the query, in hot tier
    far_z = _rand(dim, 2)  # unrelated direction, in cold tier

    close_id = store.observe(z=close_z, pred_err=1.0, episode_id=0)  # highest -> stays hot
    for i in range(3, 13):
        store.observe(z=_rand(dim, i), pred_err=0.1, episode_id=0)  # middle
    far_id = store.observe(z=far_z, pred_err=0.0, episode_id=0)  # lowest -> migrated

    store.migrate_hot_to_cold(now_ns=NOW_NS, capacity=1)
    assert store.tier_of(close_id) == "hot"
    assert store.tier_of(far_id) == "cold"
    store._cold_index.nprobe = store._cold_index._n_lists_actual

    results = store.recall(query, k=2)
    result_ids = [id_ for id_, _ in results]
    assert result_ids[0] == close_id
    scores = dict(results)
    if far_id in scores:
        assert scores[close_id] > scores[far_id]


def test_promotion_moves_cold_record_back_to_hot():
    dim = 8
    store = TieredStore(dim=dim, hot_capacity=3)
    ids = [store.observe(z=_rand(dim, i), pred_err=0.1, episode_id=0) for i in range(6)]

    result = store.migrate_hot_to_cold(now_ns=NOW_NS)
    cold_id = result["migrated_ids"][0]
    assert store.tier_of(cold_id) == "cold"

    store.promote_to_hot(cold_id)
    assert store.tier_of(cold_id) == "hot"
    assert cold_id in store._hot_ids
    assert cold_id not in store._cold_ids

    # Re-promoting is a no-op, not an error.
    store.promote_to_hot(cold_id)
    assert store.tier_of(cold_id) == "hot"

    # No duplication: the record's stale cold-index copy must never
    # resurface once promoted.
    query = store.get(cold_id).z
    hits = store.recall(query, k=len(ids))
    assert sum(1 for id_, _ in hits if id_ == cold_id) == 1


def test_no_duplication_across_tiers_after_migration():
    dim = 8
    store = TieredStore(dim=dim, hot_capacity=4)
    ids = [store.observe(z=_rand(dim, i), pred_err=float(i) * 0.05, episode_id=0) for i in range(10)]

    store.migrate_hot_to_cold(now_ns=NOW_NS)

    assert store._hot_ids.isdisjoint(store._cold_ids)
    assert store._hot_ids | store._cold_ids == set(ids)
    for id_ in ids:
        assert (id_ in store._hot_ids) != (id_ in store._cold_ids)


def test_determinism():
    def run():
        dim = 8
        store = TieredStore(dim=dim, hot_capacity=5)
        for i in range(15):
            store.observe(z=_rand(dim, i), pred_err=float(i % 5) * 0.1, episode_id=0)
        result = store.migrate_hot_to_cold(now_ns=NOW_NS)
        return sorted(result["migrated_ids"]), store.hot_count, store.cold_count

    assert run() == run()
