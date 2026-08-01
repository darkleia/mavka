import numpy as np

from mavka.lifecycle.compaction import compact
from mavka.graph.adjacency import EDGE_ANALOGOUS, EDGE_TEMPORAL, AdjacencyStore
from mavka.storage.log import AppendLog
from mavka.index.flat import VectorStore


def _rand(dim, seed):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def _build_index(dim, log, ids):
    index = VectorStore(dim=dim)
    for id_ in ids:
        index.add(log.get(id_).z)
    return index


def test_tombstone_reclamation_merge_false():
    dim = 8
    log = AppendLog(dim=dim)
    ids = [log.append(z=_rand(dim, i), episode_id=0) for i in range(10)]
    tombstoned = {2, 5, 7}
    for i in tombstoned:
        log.tombstone(ids[i])

    index = _build_index(dim, log, ids)
    result = compact(log, index, index_factory=lambda: VectorStore(dim=dim), merge=False)

    new_log = result["log"]
    id_map = result["id_map"]

    assert new_log.count == 7
    for i in tombstoned:
        assert id_map[ids[i]] is None

    for i in range(10):
        if i in tombstoned:
            continue
        new_id = id_map[ids[i]]
        assert new_id is not None
        np.testing.assert_array_equal(new_log.get(new_id).z, log.get(ids[i]).z)

    results = result["index"].search(log.get(ids[0]).z, k=1)
    assert results[0][0] == id_map[ids[0]]


def test_near_duplicate_merge_keeps_highest_pred_err():
    dim = 8
    log = AppendLog(dim=dim)
    base_z = _rand(dim, 0)
    id_a = log.append(z=base_z, episode_id=0, pred_err=0.1)
    id_b = log.append(z=base_z + _rand(dim, 1) * 0.0001, episode_id=0, pred_err=0.9)
    id_c = log.append(z=base_z + _rand(dim, 2) * 0.0001, episode_id=0, pred_err=0.3)

    index = _build_index(dim, log, [id_a, id_b, id_c])
    result = compact(
        log, index, index_factory=lambda: VectorStore(dim=dim), merge=True, similarity_threshold=0.999
    )
    id_map = result["id_map"]

    assert id_map[id_a] == id_map[id_b] == id_map[id_c]
    rep_new_id = id_map[id_a]
    assert result["log"].get(rep_new_id).pred_err == 0.9
    assert result["log"].count == 1


def test_reference_integrity_after_compaction():
    dim = 8
    log = AppendLog(dim=dim, action_dim=None)
    base_z = _rand(dim, 0)
    ids = [
        log.append(z=base_z, episode_id=0, pred_err=0.1),
        log.append(z=base_z + _rand(dim, 1) * 0.0001, episode_id=0, pred_err=0.9),
        log.append(z=_rand(dim, 5), episode_id=0, pred_err=0.2),
    ]
    log.tombstone(ids[2])

    graph = AdjacencyStore(degree=4)
    for _ in ids:
        graph.add_node()
    graph.add_edge(ids[0], ids[1], weight=1.0, edge_type=EDGE_TEMPORAL)
    graph.add_edge(ids[0], ids[2], weight=0.5, edge_type=EDGE_ANALOGOUS)  # points at a tombstoned id

    index = _build_index(dim, log, ids)
    result = compact(
        log,
        index,
        index_factory=lambda: VectorStore(dim=dim),
        graph=graph,
        merge=True,
        similarity_threshold=0.999,
    )
    id_map = result["id_map"]
    new_graph = result["graph"]
    new_log = result["log"]

    assert id_map[ids[2]] is None

    for src in range(new_graph.count):
        edges = new_graph.neighbors_of_type(src, EDGE_TEMPORAL) + new_graph.neighbors_of_type(
            src, EDGE_ANALOGOUS
        )
        for dst, _weight in edges:
            assert 0 <= dst < new_log.count

    assert result["index"].count == new_log.count


def test_episode_coherence_after_compaction():
    dim = 8
    log = AppendLog(dim=dim)
    ep0_ids = [log.append(z=_rand(dim, i), episode_id=0, pred_err=float(i)) for i in range(5)]
    log.tombstone(ep0_ids[2])

    index = _build_index(dim, log, ep0_ids)
    result = compact(log, index, index_factory=lambda: VectorStore(dim=dim), merge=False)
    new_log = result["log"]
    id_map = result["id_map"]

    surviving_new_ids = [id_map[old] for old in ep0_ids if id_map[old] is not None]
    seq_nos = [new_log.get(nid).seq_no for nid in surviving_new_ids]
    assert seq_nos == list(range(len(surviving_new_ids)))

    for i in range(len(surviving_new_ids) - 1):
        next_record = new_log.next_in_episode(surviving_new_ids[i])
        assert next_record.id == surviving_new_ids[i + 1]

    assert new_log.next_in_episode(surviving_new_ids[-1]) is None
    assert new_log.prev_in_episode(surviving_new_ids[0]) is None


def test_merge_false_is_lossless_for_live_records():
    dim = 8
    log = AppendLog(dim=dim)
    ids = [log.append(z=_rand(dim, i), episode_id=0) for i in range(8)]
    log.tombstone(ids[3])

    index = _build_index(dim, log, ids)
    result = compact(log, index, index_factory=lambda: VectorStore(dim=dim), merge=False)

    assert result["log"].count == 7
    for i, old_id in enumerate(ids):
        if i == 3:
            assert result["id_map"][old_id] is None
        else:
            assert result["id_map"][old_id] is not None


def test_no_duplicates_case_is_noop_for_merge():
    dim = 8
    log = AppendLog(dim=dim)
    ids = [log.append(z=_rand(dim, i), episode_id=0) for i in range(5)]

    index = _build_index(dim, log, ids)
    result = compact(
        log, index, index_factory=lambda: VectorStore(dim=dim), merge=True, similarity_threshold=0.9999
    )

    assert result["log"].count == 5
    assert result["stats"]["records_merged"] == 0


def test_empty_store():
    dim = 8
    log = AppendLog(dim=dim)
    index = VectorStore(dim=dim)

    result = compact(log, index, index_factory=lambda: VectorStore(dim=dim))

    assert result["log"].count == 0
    assert result["index"].count == 0
    assert result["id_map"] == {}


def test_all_tombstoned_store():
    dim = 8
    log = AppendLog(dim=dim)
    ids = [log.append(z=_rand(dim, i), episode_id=0) for i in range(4)]
    for id_ in ids:
        log.tombstone(id_)

    index = _build_index(dim, log, ids)
    result = compact(log, index, index_factory=lambda: VectorStore(dim=dim))

    assert result["log"].count == 0
    for id_ in ids:
        assert result["id_map"][id_] is None


def test_determinism():
    def run():
        dim = 8
        log = AppendLog(dim=dim)
        base_z = _rand(dim, 0)
        ids = [
            log.append(z=base_z + _rand(dim, i) * 0.0001, episode_id=0, pred_err=float(i))
            for i in range(6)
        ]
        index = _build_index(dim, log, ids)
        result = compact(
            log,
            index,
            index_factory=lambda: VectorStore(dim=dim),
            merge=True,
            similarity_threshold=0.99,
        )
        return result["id_map"], result["log"].count

    assert run() == run()
