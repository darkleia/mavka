from mavka.action_conditioning import evaluate_with_retrieval
from mavka.adapter import SyntheticWorldModel, generate_trajectory
from mavka.eval.baseline import evaluate_no_memory, split_episodes
from mavka.graph.builder import EdgeBuilder
from mavka.graph.expand import expand
from mavka.retrieval.fusion import ConcatFusionPredictor
from mavka.graph.adjacency import AdjacencyStore
from mavka.pipeline import Pipeline
from mavka.retrieval.scorer import FixedWeightScorer
from mavka.index.flat import FlatIndex


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
    demo_pipeline = Pipeline(
        dim=dim, action_dim=action_dim, index=FlatIndex(dim=dim), graph=graph, edge_builder=builder
    )
    for episode in memory_episodes:
        for step in episode:
            demo_pipeline.observe(
                z=step["z"],
                action=step["action"],
                z_next=step["z_next"],
                pred_err=step["pred_err"],
                episode_id=step["episode_id"],
            )

    query_z = demo_pipeline.get(demo_pipeline.count // 2).z
    seed_ids = [id_ for id_, _ in demo_pipeline.recall(query_z, k * fetch_factor)]
    print(f"pipeline count: {demo_pipeline.count}, seeds before expansion: {len(seed_ids)}\n")

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
        eval_pipeline = Pipeline(
            dim=dim,
            action_dim=action_dim,
            index=FlatIndex(dim=dim),
            graph=eval_graph,
            edge_builder=eval_builder,
        )
        scorer = FixedWeightScorer(eval_pipeline._log, w_sim=1.0, w_action=0.0, w_recency=0.0)
        eval_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=0)
        predictor = ConcatFusionPredictor(eval_adapter, alpha=1.0)

        result = evaluate_with_retrieval(
            memory_episodes,
            eval_episodes,
            eval_pipeline,
            predictor,
            k=k,
            scale=1.0,
            scorer=scorer,
            fetch_factor=fetch_factor,
            graph=eval_graph,
            expand_depth=depth,
            max_nodes=max_nodes,
        )
        print(f"  depth={depth}: mean_error={result['mean_error']:.6f}")


if __name__ == "__main__":
    main()
