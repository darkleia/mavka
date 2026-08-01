from mavka.adapter import SyntheticWorldModel
from mavka.experiment import format_experiment_report, memory_helps, run_memory_experiment


def main() -> None:
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

    print(f"seeds: {results['seeds']}  eval steps per seed: {results['eval_set_sizes']}\n")
    print(format_experiment_report(results))

    helps = memory_helps(results)
    print(f"\nmemory_helps (full system reliably beats baseline): {helps}")

    print(
        "\n"
        + "=" * 78
        + "\nSYNTHETIC RESULT -- validates the pipeline and that memory helps on\n"
        "structured data. NOT a claim about real world models; that requires\n"
        "wiring a real model (later).\n"
        + "=" * 78
    )


if __name__ == "__main__":
    main()
