from mavka.adapter import SyntheticWorldModel
from mavka.eval.experiment import format_experiment_report, memory_helps, run_memory_experiment


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

    full = results["conditions"]["full_system"]
    maint = results["conditions"]["full_system_with_maintenance"]
    print("\nfull_system vs full_system_with_maintenance:")
    print(f"  full_system                   mean_error={full['mean_error']:.6f}  std={full['std_error']:.6f}")
    print(f"  full_system_with_maintenance  mean_error={maint['mean_error']:.6f}  std={maint['std_error']:.6f}")
    delta_pct = 100.0 * (maint["mean_error"] - full["mean_error"]) / full["mean_error"]
    print(f"  delta: {delta_pct:+.2f}% error (positive = maintenance condition worse)")
    print(
        "  note: full_system_with_maintenance uses action_scale=0.0, not "
        "full_system's action_scale=2.0 -- see run_memory_experiment's "
        "docstring for the confirmed compact()/eviction incompatibility "
        "with an action-conditioned index that forces this. Some of the "
        "delta above is that keying-strategy difference, not maintenance "
        "itself; see the task summary for a deconfounded estimate."
    )

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
