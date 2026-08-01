import functools

import numpy as np

from mavka.adapter import SyntheticWorldModel
from mavka.config import MavkaConfig
from mavka.eval.experiment import CONDITION_NAMES, graph_helps, run_memory_experiment
from mavka.graph.adjacency import EDGE_TEMPORAL, AdjacencyStore
from mavka.graph.expand import expand
from mavka.memory import Memory
from mavka.retrieval.scorer import FixedWeightScorer
from mavka.index.flat import FlatIndex


def _rand(dim, seed):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def test_graph_off_is_pre_graph_identity():
    # Memory has no use_graph override -- expansion_depth==0 is the sole
    # off-switch. This proves it holds even with a graph fully wired up:
    # a Memory with graph configured but expansion_depth=0 must match a
    # Memory with no graph at all, exactly.
    dim = 8
    action_dim = 2
    config = MavkaConfig(dim=dim, action_dim=action_dim)

    zs = [_rand(dim, i) for i in range(20)]
    actions = [_rand(action_dim, i + 100) for i in range(20)]

    memory_no_graph = Memory(config, index=FlatIndex(dim=dim + action_dim), action_scale=1.0)
    memory_no_graph.scorer = FixedWeightScorer(memory_no_graph._log)
    ids = [
        memory_no_graph.observe(z=z, action=action, z_next=None, episode_id=0)
        for z, action in zip(zs, actions)
    ]

    graph = AdjacencyStore(degree=4)
    for _ in ids:
        graph.add_node()
    for i in range(len(ids) - 1):
        graph.add_edge(ids[i], ids[i + 1], weight=1.0, edge_type=EDGE_TEMPORAL)

    memory_graph_off = Memory(
        config, index=FlatIndex(dim=dim + action_dim), action_scale=1.0, graph=graph, expansion_depth=0
    )
    memory_graph_off.scorer = FixedWeightScorer(memory_graph_off._log)
    for z, action in zip(zs, actions):
        memory_graph_off.observe(z=z, action=action, z_next=None, episode_id=0)

    query_z = _rand(dim, 999)
    query_action = _rand(action_dim, 998)

    without_graph_param = memory_no_graph.recall(query_z, action=query_action, k=5)
    with_graph_but_off = memory_graph_off.recall(query_z, action=query_action, k=5)

    assert without_graph_param == with_graph_but_off


def test_graph_expansion_differs_from_off_when_graph_has_edges():
    dim = 8
    action_dim = 2
    config = MavkaConfig(dim=dim, action_dim=action_dim)

    zs = [_rand(dim, i) for i in range(10)]
    actions = [_rand(action_dim, i + 50) for i in range(10)]

    memory_off = Memory(config, index=FlatIndex(dim=dim + action_dim), action_scale=1.0, fetch_factor=1)
    memory_off.scorer = FixedWeightScorer(memory_off._log, w_sim=1.0, w_action=0.0, w_recency=0.0)
    ids = [
        memory_off.observe(z=z, action=action, z_next=None, episode_id=0)
        for z, action in zip(zs, actions)
    ]

    graph = AdjacencyStore(degree=4)
    for _ in ids:
        graph.add_node()
    for i in range(len(ids) - 1):
        graph.add_edge(ids[i], ids[i + 1], weight=1.0, edge_type=EDGE_TEMPORAL)

    memory_on = Memory(
        config,
        index=FlatIndex(dim=dim + action_dim),
        action_scale=1.0,
        fetch_factor=1,
        graph=graph,
        expansion_depth=2,
        expander=functools.partial(expand, max_nodes=100, edge_types=None),
    )
    memory_on.scorer = FixedWeightScorer(memory_on._log, w_sim=1.0, w_action=0.0, w_recency=0.0)
    for z, action in zip(zs, actions):
        memory_on.observe(z=z, action=action, z_next=None, episode_id=0)

    query_z = memory_off.get(ids[0]).z
    query_action = memory_off.get(ids[0]).action

    off_result = memory_off.recall(query_z, action=query_action, k=1)
    on_result = memory_on.recall(query_z, action=query_action, k=5)

    off_ids = {id_ for id_, _ in off_result}
    on_ids = {id_ for id_, _ in on_result}
    assert off_ids != on_ids
    assert off_ids <= on_ids


def test_experiment_includes_graph_conditions_with_same_eval_set():
    adapter = SyntheticWorldModel(dim=8, action_dim=2, seed=0)
    seeds = [0, 1]
    results = run_memory_experiment(adapter, n_episodes=10, episode_length=6, k=3, seeds=seeds)

    graph_condition_names = [
        "full_system_no_graph",
        "full_system_graph_depth1",
        "full_system_graph_depth2",
    ]
    for name in graph_condition_names:
        assert name in results["conditions"]
        stats = results["conditions"][name]
        assert set(stats.keys()) >= {
            "mean_error",
            "std_error",
            "n_steps",
            "relative_improvement_pct",
            "per_seed_errors",
        }
        assert len(stats["per_seed_errors"]) == len(seeds)

    n_steps_values = {results["conditions"][name]["n_steps"] for name in CONDITION_NAMES}
    assert len(n_steps_values) == 1


def test_graph_helps_gates_on_significance():
    no_improvement = {
        "conditions": {
            "full_system_no_graph": {"mean_error": 0.001, "std_error": 0.0001},
            "full_system_graph_depth2": {"mean_error": 0.001, "std_error": 0.0001},
        },
    }
    assert graph_helps(no_improvement, depth=2) is False

    noisy = {
        "conditions": {
            "full_system_no_graph": {"mean_error": 0.001, "std_error": 0.0005},
            "full_system_graph_depth2": {"mean_error": 0.0009, "std_error": 0.0005},
        },
    }
    assert graph_helps(noisy, depth=2) is False

    clear = {
        "conditions": {
            "full_system_no_graph": {"mean_error": 0.010, "std_error": 0.0001},
            "full_system_graph_depth2": {"mean_error": 0.002, "std_error": 0.0001},
        },
    }
    assert graph_helps(clear, depth=2) is True
    assert graph_helps(clear, depth=2, require_significance=False) is True


def test_determinism():
    def run():
        adapter = SyntheticWorldModel(dim=8, action_dim=2, seed=5)
        return run_memory_experiment(adapter, n_episodes=10, episode_length=6, k=3, seeds=[0, 1])

    result_a = run()
    result_b = run()

    for name in ("full_system_no_graph", "full_system_graph_depth1", "full_system_graph_depth2"):
        assert (
            result_a["conditions"][name]["mean_error"] == result_b["conditions"][name]["mean_error"]
        )
        assert (
            result_a["conditions"][name]["per_seed_errors"]
            == result_b["conditions"][name]["per_seed_errors"]
        )
