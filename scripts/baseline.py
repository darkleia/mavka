from mavka.adapter import SyntheticWorldModel, generate_trajectory
from mavka.baseline import evaluate_no_memory, split_episodes


def main() -> None:
    dim = 32
    action_dim = 4
    n_episodes = 100
    episode_length = 40
    holdout_frac = 0.2
    seed = 0

    generation_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=seed)
    all_episodes = [
        generate_trajectory(generation_adapter, episode_length, episode_id=i)
        for i in range(n_episodes)
    ]

    memory_episodes, eval_episodes = split_episodes(
        all_episodes, holdout_frac=holdout_frac, seed=seed
    )
    print(
        f"episodes: {len(all_episodes)} total, "
        f"{len(memory_episodes)} memory, {len(eval_episodes)} eval"
    )

    # Fresh adapter (same seed = same "model weights") so the measured error
    # depends only on the model and the eval episodes, not on incidental RNG
    # draws consumed while generating the memory split.
    eval_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=seed)
    result = evaluate_no_memory(eval_adapter, eval_episodes)

    print("\nNO-MEMORY BASELINE -- the number to beat")
    print(f"  mean_error:   {result['mean_error']:.6f}")
    print(f"  median_error: {result['median_error']:.6f}")
    print(f"  p90_error:    {result['p90_error']:.6f}")
    print(f"  n_steps:      {result['n_steps']}")


if __name__ == "__main__":
    main()
