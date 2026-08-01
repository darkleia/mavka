from mavka.adapter import SyntheticWorldModel
from mavka.eval.experiment import graph_helps, run_memory_experiment


def report_graph_ablation() -> None:
    dim = 32
    action_dim = 4
    n_episodes = 100
    episode_length = 40
    k = 10
    seeds = [0, 1, 2, 3, 4]

    adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=seeds[0])
    results = run_memory_experiment(
        adapter, n_episodes=n_episodes, episode_length=episode_length, k=k, seeds=seeds
    )

    conditions = results["conditions"]
    no_graph = conditions["full_system_no_graph"]

    print(f"seeds: {results['seeds']}  eval steps per seed: {results['eval_set_sizes']}\n")

    header = f"{'condition':28} {'mean_error':>12} {'std_error':>12} {'vs no-graph':>13}"
    print(header)
    print("-" * len(header))
    print(
        f"{'full_system_no_graph':28} {no_graph['mean_error']:12.6f} "
        f"{no_graph['std_error']:12.6f} {'--':>13}"
    )
    for depth in (1, 2):
        name = f"full_system_graph_depth{depth}"
        stats = conditions[name]
        pct_change = 100.0 * (no_graph["mean_error"] - stats["mean_error"]) / no_graph["mean_error"]
        print(
            f"{name:28} {stats['mean_error']:12.6f} {stats['std_error']:12.6f} {pct_change:+12.2f}%"
        )

    print("\ndoes the graph help beyond action-conditioned retrieval + scorer + fusion?")
    for depth in (1, 2):
        verdict = graph_helps(results, depth=depth)
        print(f"  depth={depth}: graph_helps = {verdict}")

    print(
        "\n"
        + "=" * 78
        + "\nSYNTHETIC RESULT -- validates the pipeline and measures the graph's\n"
        "effect on structured data. NOT a claim about real world models; that\n"
        "requires wiring a real model (later).\n"
        + "=" * 78
    )


if __name__ == "__main__":
    report_graph_ablation()
