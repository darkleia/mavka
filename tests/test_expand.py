import numpy as np

from mavka.edges import EdgeBuilder
from mavka.expand import decay_for_depth, expand
from mavka.graph import EDGE_ANALOGOUS, EDGE_TEMPORAL, AdjacencyStore
from mavka.pipeline import ActionConditionedPipeline, Pipeline
from mavka.scorer import FixedWeightScorer
from mavka.store import VectorStore


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
    graph = AdjacencyStore(degree=4)
    pipeline = ActionConditionedPipeline(
        dim=dim, action_dim=action_dim, index=VectorStore(dim=dim + action_dim)
    )
    # ActionConditionedPipeline has no graph wiring in observe(); build the
    # graph independently to test recall_scored's own depth=0 off-switch.
    scorer = FixedWeightScorer(pipeline._log)

    rng = np.random.default_rng(0)
    for i in range(20):
        pipeline.observe(
            z=rng.standard_normal(dim).astype(np.float32),
            action=rng.standard_normal(action_dim).astype(np.float32),
            z_next=None,
            episode_id=0,
        )

    query_z = rng.standard_normal(dim).astype(np.float32)
    query_action = rng.standard_normal(action_dim).astype(np.float32)

    without_expand = pipeline.recall_scored(query_z, query_action, k=5, scorer=scorer)
    with_expand_off = pipeline.recall_scored(
        query_z, query_action, k=5, scorer=scorer, graph=graph, expand_depth=0
    )

    assert without_expand == with_expand_off


def test_scoring_decay_ranks_seed_above_equal_similarity_depth_two_node():
    dim = 8
    graph = AdjacencyStore(degree=4)
    builder = EdgeBuilder(n_analogous=0, similarity_threshold=1.1, temporal_weight=1.0)
    pipeline = Pipeline(dim=dim, index=VectorStore(dim=dim), graph=graph, edge_builder=builder)

    ids = [
        pipeline.observe(z=_rand(dim, i), action=None, z_next=None, episode_id=0) for i in range(3)
    ]
    # ids[0] -> ids[1] -> ids[2], both temporal edges (weight 1.0 each).

    scorer = FixedWeightScorer(pipeline._log, w_sim=1.0, w_action=0.0, w_recency=0.0)
    query_z = pipeline.get(ids[0]).z

    ranked = pipeline.recall_scored(
        query_z, None, k=3, scorer=scorer, fetch_factor=1, graph=graph, expand_depth=2, max_nodes=10
    )
    ranked_ids = [id_ for id_, _ in ranked]

    assert ranked_ids.index(ids[0]) < ranked_ids.index(ids[2])
