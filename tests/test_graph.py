import pytest

from mavka.graph.adjacency import AdjacencyStore


def _make_nodes(store, n):
    return [store.add_node() for _ in range(n)]


def test_new_node_has_no_neighbors():
    store = AdjacencyStore(degree=4)
    (id0,) = _make_nodes(store, 1)

    assert store.degree_of(id0) == 0
    assert store.neighbors(id0) == []
    assert store.neighbor_weights(id0) == []
    assert store.count == 1


def test_add_edge_fills_slots_in_order():
    store = AdjacencyStore(degree=4)
    ids = _make_nodes(store, 4)
    src = ids[0]

    for dst in ids[1:4]:
        assert store.add_edge(src, dst, weight=1.0) is True

    assert store.neighbors(src) == ids[1:4]
    assert store.degree_of(src) == 3


def test_fixed_degree_enforcement_and_replace_weakest_policy():
    store = AdjacencyStore(degree=3)
    ids = _make_nodes(store, 6)
    src = ids[0]

    # Fill all 3 slots.
    assert store.add_edge(src, ids[1], weight=1.0) is True
    assert store.add_edge(src, ids[2], weight=2.0) is True
    assert store.add_edge(src, ids[3], weight=3.0) is True
    assert store.degree_of(src) == 3

    # A lower-weight edge than the current weakest (1.0) is rejected.
    assert store.add_edge(src, ids[4], weight=0.5) is False
    assert store.degree_of(src) == 3
    assert ids[4] not in store.neighbors(src)

    # A higher-weight edge than the current weakest (1.0) evicts it.
    assert store.add_edge(src, ids[5], weight=5.0) is True
    assert store.degree_of(src) == 3
    neighbors = store.neighbors(src)
    assert ids[1] not in neighbors  # the weakest (weight 1.0) got evicted
    assert set(neighbors) == {ids[2], ids[3], ids[5]}


def test_no_duplicate_neighbors_updates_weight_instead():
    store = AdjacencyStore(degree=4)
    ids = _make_nodes(store, 2)
    src, dst = ids

    assert store.add_edge(src, dst, weight=1.0) is True
    assert store.degree_of(src) == 1

    assert store.add_edge(src, dst, weight=9.0) is True
    assert store.degree_of(src) == 1
    assert store.neighbors(src) == [dst]
    assert store.neighbor_weights(src) == [(dst, 9.0)]


def test_self_loop_rejected():
    store = AdjacencyStore(degree=4)
    (id0,) = _make_nodes(store, 1)

    assert store.add_edge(id0, id0, weight=1.0) is False
    assert store.degree_of(id0) == 0


def test_weights_round_trip():
    store = AdjacencyStore(degree=4)
    ids = _make_nodes(store, 4)
    src = ids[0]

    store.add_edge(src, ids[1], weight=0.25)
    store.add_edge(src, ids[2], weight=0.75)
    store.add_edge(src, ids[3], weight=0.5)

    result = dict(store.neighbor_weights(src))
    assert result == {ids[1]: 0.25, ids[2]: 0.75, ids[3]: 0.5}


def test_growth_beyond_initial_capacity():
    store = AdjacencyStore(degree=2, initial_capacity=4)
    n = 20
    ids = _make_nodes(store, n)
    assert store.count == n

    for i in range(n - 1):
        store.add_edge(ids[i], ids[i + 1], weight=1.0)

    for i in range(n - 1):
        assert store.neighbors(ids[i]) == [ids[i + 1]]
    assert store.neighbors(ids[n - 1]) == []


def test_has_edge_and_degree_of():
    store = AdjacencyStore(degree=4)
    ids = _make_nodes(store, 3)
    a, b, c = ids

    store.add_edge(a, b, weight=1.0)

    assert store.has_edge(a, b) is True
    assert store.has_edge(a, c) is False
    assert store.has_edge(b, a) is False  # directed, not symmetric
    assert store.degree_of(a) == 1
    assert store.degree_of(b) == 0


def test_unknown_id_operations_raise_value_error():
    store = AdjacencyStore(degree=4)
    (id0,) = _make_nodes(store, 1)
    unknown = 999

    with pytest.raises(ValueError):
        store.add_edge(unknown, id0)
    with pytest.raises(ValueError):
        store.add_edge(id0, unknown)
    with pytest.raises(ValueError):
        store.neighbors(unknown)
    with pytest.raises(ValueError):
        store.neighbor_weights(unknown)
    with pytest.raises(ValueError):
        store.has_edge(unknown, id0)
    with pytest.raises(ValueError):
        store.has_edge(id0, unknown)
    with pytest.raises(ValueError):
        store.degree_of(unknown)


def test_determinism():
    def run():
        store = AdjacencyStore(degree=3, initial_capacity=2)
        ids = _make_nodes(store, 10)
        for i, src in enumerate(ids):
            for j in range(i + 1, min(i + 5, len(ids))):
                store.add_edge(src, ids[j], weight=float(j - i))
        return [store.neighbor_weights(id_) for id_ in ids]

    assert run() == run()
