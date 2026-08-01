import functools

import numpy as np

from mavka.config import MavkaConfig
from mavka.graph.builder import EdgeBuilder
from mavka.graph.expand import decay_for_depth, expand
from mavka.graph.adjacency import EDGE_ANALOGOUS, EDGE_TEMPORAL, AdjacencyStore
from mavka.memory import Memory
from mavka.retrieval.scorer import FixedWeightScorer
from mavka.index.flat import FlatIndex


def _rand(dim, seed):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def _linear_graph(n, degree=4):
    """0 -> 1 -> 2 -> ... -> n-1, all temporal edges, weight 1.0."""
    graph = AdjacencyStore(degree=degree)
    ids = [graph.add_node() for _ in range(n)]
    for i in range(n - 1):
        graph.add_edge(ids[i], ids[i + 1], weight=1.0, edge_type=EDGE_TEMPORAL)
    return graph, ids


def test_depth_zero_is_identity():
    graph, ids = _linear_graph(5)

    result = expand([ids[0]], graph, depth=0, max_nodes=100)

    assert result == [(ids[0], {"is_seed": True, "depth": 0, "edge_type": None, "weight": None})]


def test_depth_one_and_two_membership_on_known_graph():
    graph, ids = _linear_graph(5)

    depth1 = expand([ids[0]], graph, depth=1, max_nodes=100)
    depth2 = expand([ids[0]], graph, depth=2, max_nodes=100)

    assert {id_ for id_, _ in depth1} == {ids[0], ids[1]}
    assert {id_ for id_, _ in depth2} == {ids[0], ids[1], ids[2]}


def test_provenance_correct():
    graph, ids = _linear_graph(4)

    result = dict(expand([ids[0]], graph, depth=2, max_nodes=100))

    assert result[ids[0]]["is_seed"] is True
    assert result[ids[0]]["depth"] == 0

    assert result[ids[1]]["is_seed"] is False
    assert result[ids[1]]["depth"] == 1
    assert result[ids[1]]["edge_type"] == EDGE_TEMPORAL
    assert result[ids[1]]["weight"] == 1.0

    assert result[ids[2]]["is_seed"] is False
    assert result[ids[2]]["depth"] == 2


def test_dedup_keeps_shallowest_provenance():
    # A diamond: 0 -> 1 -> 3 and 0 -> 2 -> 3. Node 3 is reachable at depth 2
    # via either path -- must appear once.
    graph = AdjacencyStore(degree=4)
    ids = [graph.add_node() for _ in range(4)]
    graph.add_edge(ids[0], ids[1], weight=1.0, edge_type=EDGE_TEMPORAL)
    graph.add_edge(ids[0], ids[2], weight=1.0, edge_type=EDGE_ANALOGOUS)
    graph.add_edge(ids[1], ids[3], weight=1.0, edge_type=EDGE_TEMPORAL)
    graph.add_edge(ids[2], ids[3], weight=1.0, edge_type=EDGE_ANALOGOUS)

    result = expand([ids[0]], graph, depth=2, max_nodes=100)

    result_ids = [id_ for id_, _ in result]
    assert result_ids.count(ids[3]) == 1
    provenance = dict(result)
    assert provenance[ids[3]]["depth"] == 2


def test_max_nodes_cap_respected_and_seeds_never_dropped():
    graph, ids = _linear_graph(10)

    # Cap smaller than the seed count: seed still returned in full.
    seeds = [ids[0], ids[1], ids[2]]
    result = expand(seeds, graph, depth=1, max_nodes=1)
    result_ids = {id_ for id_, _ in result}
    assert set(seeds) <= result_ids

    # Cap that only allows a little expansion beyond the seeds.
    result2 = expand([ids[0]], graph, depth=5, max_nodes=3)
    assert len(result2) <= 3
    assert ids[0] in [id_ for id_, _ in result2]


def test_cycle_safety():
    graph = AdjacencyStore(degree=4)
    ids = [graph.add_node() for _ in range(3)]
    graph.add_edge(ids[0], ids[1], weight=1.0, edge_type=EDGE_TEMPORAL)
    graph.add_edge(ids[1], ids[2], weight=1.0, edge_type=EDGE_TEMPORAL)
    graph.add_edge(ids[2], ids[0], weight=1.0, edge_type=EDGE_TEMPORAL)  # cycle back to seed

    result = expand([ids[0]], graph, depth=10, max_nodes=100)

    assert {id_ for id_, _ in result} == set(ids)
    assert len(result) == 3


def test_edge_types_filter_gives_different_results():
    graph = AdjacencyStore(degree=4)
    ids = [graph.add_node() for _ in range(3)]
    graph.add_edge(ids[0], ids[1], weight=1.0, edge_type=EDGE_TEMPORAL)
    graph.add_edge(ids[0], ids[2], weight=0.8, edge_type=EDGE_ANALOGOUS)

    temporal_only = {id_ for id_, _ in expand([ids[0]], graph, depth=1, max_nodes=100, edge_types=[EDGE_TEMPORAL])}
    analogous_only = {
        id_ for id_, _ in expand([ids[0]], graph, depth=1, max_nodes=100, edge_types=[EDGE_ANALOGOUS])
    }
    both = {id_ for id_, _ in expand([ids[0]], graph, depth=1, max_nodes=100)}

    assert temporal_only == {ids[0], ids[1]}
    assert analogous_only == {ids[0], ids[2]}
    assert both == {ids[0], ids[1], ids[2]}


def test_decay_for_depth_values():
    assert decay_for_depth(0) == 1.0
    assert decay_for_depth(1) == 0.5
    assert decay_for_depth(2) == 0.25


def test_determinism():
    graph, ids = _linear_graph(8)
    assert expand([ids[0]], graph, depth=3, max_nodes=100) == expand(
        [ids[0]], graph, depth=3, max_nodes=100
    )


def test_off_switch_end_to_end_matches_no_expansion():
    dim = 8
    action_dim = 2
    config = MavkaConfig(dim=dim, action_dim=action_dim)

    rng = np.random.default_rng(0)
    zs = [rng.standard_normal(dim).astype(np.float32) for _ in range(20)]
    actions = [rng.standard_normal(action_dim).astype(np.float32) for _ in range(20)]

    # A graph is configured but expansion_depth=0 -- must behave exactly
    # like no graph being configured at all: depth is the true off-switch.
    graph = AdjacencyStore(degree=4)

    memory_no_graph = Memory(config, index=FlatIndex(dim=dim + action_dim), action_scale=1.0)
    memory_no_graph.scorer = FixedWeightScorer(memory_no_graph._log)

    memory_graph_off = Memory(
        config, index=FlatIndex(dim=dim + action_dim), action_scale=1.0, graph=graph, expansion_depth=0
    )
    memory_graph_off.scorer = FixedWeightScorer(memory_graph_off._log)

    for z, action in zip(zs, actions):
        memory_no_graph.observe(z=z, action=action, z_next=None, episode_id=0)
        memory_graph_off.observe(z=z, action=action, z_next=None, episode_id=0)

    query_z = rng.standard_normal(dim).astype(np.float32)
    query_action = rng.standard_normal(action_dim).astype(np.float32)

    without_expand = memory_no_graph.recall(query_z, action=query_action, k=5)
    with_expand_off = memory_graph_off.recall(query_z, action=query_action, k=5)

    assert without_expand == with_expand_off


def test_scoring_decay_ranks_seed_above_equal_similarity_depth_two_node():
    dim = 8
    graph = AdjacencyStore(degree=4)
    builder = EdgeBuilder(n_analogous=0, similarity_threshold=1.1, temporal_weight=1.0)
    config = MavkaConfig(dim=dim)
    memory = Memory(config, index=FlatIndex(dim=dim), action_scale=0.0)

    # Memory has no built-in edge-building hook on observe() (unlike old
    # Pipeline's graph=/edge_builder= constructor args) -- drive
    # EdgeBuilder.on_insert manually after each observe(), same pattern
    # used in test_edges.py.
    ids = []
    for i in range(3):
        record_id = memory.observe(z=_rand(dim, i), action=None, z_next=None, episode_id=0)
        graph.add_node()
        record = memory.get(record_id)
        builder.on_insert(
            record_id,
            record.z,
            record.action,
            record.episode_id,
            record.seq_no,
            memory._log,
            memory._index,
            graph,
        )
        ids.append(record_id)
    # ids[0] -> ids[1] -> ids[2], both temporal edges (weight 1.0 each).

    memory.scorer = FixedWeightScorer(memory._log, w_sim=1.0, w_action=0.0, w_recency=0.0)
    memory.fetch_factor = 1
    memory.graph = graph
    memory.expansion_depth = 2
    memory.expander = functools.partial(expand, max_nodes=10, edge_types=None)

    query_z = memory.get(ids[0]).z
    ranked = memory.recall(query_z, action=None, k=3)
    ranked_ids = [id_ for id_, _ in ranked]

    assert ranked_ids.index(ids[0]) < ranked_ids.index(ids[2])
