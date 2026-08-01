import functools

from mavka.adapter import SyntheticWorldModel, generate_trajectory
from mavka.config import MavkaConfig
from mavka.eval.baseline import evaluate_no_memory, split_episodes
from mavka.eval.retrieval_eval import evaluate_with_retrieval
from mavka.graph.adjacency import AdjacencyStore
from mavka.graph.builder import EdgeBuilder
from mavka.graph.expand import expand
from mavka.index.flat import FlatIndex
from mavka.memory import Memory
from mavka.retrieval.fusion import ConcatFusionPredictor
from mavka.retrieval.scorer import FixedWeightScorer


def _build_graph(dim, action_dim, memory_episodes, edge_builder, graph) -> Memory:
    """Memory has no built-in edge-building hook on observe() (unlike old
    Pipeline's graph=/edge_builder= constructor args) -- drive
    EdgeBuilder.on_insert manually after each observe(), against a
    throwaway memory used only to build the graph (same pattern as
    eval/experiment.py's _build_graph). This throwaway memory is returned
    only for Part 1's own direct recall() calls -- it must never be
    reused as the memory passed to evaluate_with_retrieval, which fills
    its own memory from scratch and would double-insert every record.
    """
    config = MavkaConfig(dim=dim, action_dim=action_dim)
    memory = Memory(config, index=FlatIndex(dim=dim), action_scale=0.0)

    for episode in memory_episodes:
        for step in episode:
            record_id = memory.observe(
                z=step["z"],
                action=step["action"],
                z_next=step["z_next"],
                pred_err=step["pred_err"],
                episode_id=step["episode_id"],
            )
            graph.add_node()  # kept in lockstep with record_id by construction
            record = memory.get(record_id)
            edge_builder.on_insert(
                record_id,
                record.z,
                record.action,
                record.episode_id,
                record.seq_no,
                memory._log,
                memory._index,
                graph,
            )
    return memory


def main() -> None:
    dim = 32
    action_dim = 4
    n_episodes = 100
    episode_length = 40
    k = 10
    fetch_factor = 5
    max_nodes = 200

    gen_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=0)
    all_episodes = [
        generate_trajectory(gen_adapter, episode_length, episode_id=i) for i in range(n_episodes)
    ]
    memory_episodes, eval_episodes = split_episodes(all_episodes, holdout_frac=0.2, seed=0)

    # Part 1: candidate-set size at each depth on a populated graph.
    graph = AdjacencyStore(degree=8)
    builder = EdgeBuilder(n_analogous=4, similarity_threshold=0.3, temporal_weight=1.0)
    demo_memory = _build_graph(dim, action_dim, memory_episodes, builder, graph)

    query_z = demo_memory.get(demo_memory.count // 2).z
    seed_ids = [id_ for id_, _ in demo_memory.recall(query_z, k=k * fetch_factor)]
    print(f"memory count: {demo_memory.count}, seeds before expansion: {len(seed_ids)}\n")

    print("candidate-set size by expansion depth:")
    for depth in (0, 1, 2):
        expanded = expand(seed_ids, graph, depth=depth, max_nodes=max_nodes)
        print(f"  depth={depth}: {len(expanded)} candidates")

    # Part 2: does expansion change prediction error vs. depth 0 on the eval set?
    baseline_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=0)
    baseline_result = evaluate_no_memory(baseline_adapter, eval_episodes)

    print(f"\nno-memory baseline: {baseline_result['mean_error']:.6f}\n")
    print("appearance-only retrieval (pure similarity scorer) mean_error by expansion depth:")
    for depth in (0, 1, 2):
        eval_graph = AdjacencyStore(degree=8)
        eval_builder = EdgeBuilder(n_analogous=4, similarity_threshold=0.3, temporal_weight=1.0)
        # Build the graph via its own throwaway memory; the actual
        # retrieval memory below starts empty and is filled by
        # evaluate_with_retrieval itself, reusing this already-built
        # graph -- ids line up because both observe memory_episodes in
        # the same order.
        _build_graph(dim, action_dim, memory_episodes, eval_builder, eval_graph)

        config = MavkaConfig(dim=dim, action_dim=action_dim)
        eval_memory = Memory(
            config,
            index=FlatIndex(dim=dim),
            action_scale=0.0,
            fetch_factor=fetch_factor,
            graph=eval_graph,
            expansion_depth=depth,
            expander=functools.partial(expand, max_nodes=max_nodes, edge_types=None),
        )
        eval_memory.scorer = FixedWeightScorer(eval_memory._log, w_sim=1.0, w_action=0.0, w_recency=0.0)

        eval_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=0)
        predictor = ConcatFusionPredictor(eval_adapter, alpha=1.0)

        result = evaluate_with_retrieval(memory_episodes, eval_episodes, eval_memory, predictor, k=k)
        print(f"  depth={depth}: mean_error={result['mean_error']:.6f}")


if __name__ == "__main__":
    main()
