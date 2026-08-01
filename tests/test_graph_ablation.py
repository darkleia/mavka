import numpy as np

from mavka.adapter import SyntheticWorldModel
from mavka.experiment import CONDITION_NAMES, graph_helps, run_memory_experiment
from mavka.graph import EDGE_TEMPORAL, AdjacencyStore
from mavka.pipeline import ActionConditionedPipeline
from mavka.scorer import FixedWeightScorer
from mavka.store import VectorStore


def _rand(dim, seed):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def test_use_graph_false_is_pre_graph_identity():
    dim = 8
    action_dim = 2
    pipeline = ActionConditionedPipeline(
        dim=dim, action_dim=action_dim, index=VectorStore(dim=dim + action_dim)
    )
    scorer = FixedWeightScorer(pipeline._log)

    ids = [
        pipeline.observe(z=_rand(dim, i), action=_rand(action_dim, i + 100), z_next=None, episode_id=0)
        for i in range(20)
    ]

    graph = AdjacencyStore(degree=4)
    for _ in ids:
        graph.add_node()
    for i in range(len(ids) - 1):
        graph.add_edge(ids[i], ids[i + 1], weight=1.0, edge_type=EDGE_TEMPORAL)

    query_z = _rand(dim, 999)
    query_action = _rand(action_dim, 998)

    without_graph_param = pipeline.recall_scored(query_z, query_action, k=5, scorer=scorer)
    with_graph_but_off = pipeline.recall_scored(
        query_z, query_action, k=5, scorer=scorer, graph=graph, expand_depth=2, use_graph=False
    )

    assert without_graph_param == with_graph_but_off


def test_use_graph_true_differs_from_off_when_graph_has_edges():
    dim = 8
    action_dim = 2
    pipeline = ActionConditionedPipeline(
        dim=dim, action_dim=action_dim, index=VectorStore(dim=dim + action_dim)
    )
    ids = [
        pipeline.observe(z=_rand(dim, i), action=_rand(action_dim, i + 50), z_next=None, episode_id=0)
        for i in range(10)
    ]

    graph = AdjacencyStore(degree=4)
    for _ in ids:
        graph.add_node()
    for i in range(len(ids) - 1):
        graph.add_edge(ids[i], ids[i + 1], weight=1.0, edge_type=EDGE_TEMPORAL)

    scorer = FixedWeightScorer(pipeline._log, w_sim=1.0, w_action=0.0, w_recency=0.0)
    query_z = pipeline.get(ids[0]).z
    query_action = pipeline.get(ids[0]).action

    off_result = pipeline.recall_scored(
        query_z, query_action, k=1, scorer=scorer, fetch_factor=1, graph=graph, use_graph=False
    )
    on_result = pipeline.recall_scored(
        query_z,
        query_action,
        k=5,
        scorer=scorer,
        fetch_factor=1,
        graph=graph,
        expand_depth=2,
        use_graph=True,
        max_nodes=100,
    )

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
