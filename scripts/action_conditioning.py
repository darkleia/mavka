from mavka.adapter import SyntheticWorldModel, generate_trajectory
from mavka.config import MavkaConfig
from mavka.eval.baseline import evaluate_no_memory, split_episodes
from mavka.eval.retrieval_eval import evaluate_with_retrieval
from mavka.index.flat import FlatIndex
from mavka.memory import Memory
from mavka.retrieval.fusion import ConcatFusionPredictor


def main() -> None:
    dim = 32
    action_dim = 4
    n_episodes = 100
    episode_length = 40
    holdout_frac = 0.2
    seed = 0
    k = 10

    gen_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=seed)
    all_episodes = [
        generate_trajectory(gen_adapter, episode_length, episode_id=i) for i in range(n_episodes)
    ]
    memory_episodes, eval_episodes = split_episodes(
        all_episodes, holdout_frac=holdout_frac, seed=seed
    )
    print(
        f"episodes: {len(all_episodes)} total, "
        f"{len(memory_episodes)} memory, {len(eval_episodes)} eval\n"
    )

    rows = []

    baseline_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=seed)
    baseline_result = evaluate_no_memory(baseline_adapter, eval_episodes)
    rows.append(("no-memory baseline", baseline_result["mean_error"]))

    config = MavkaConfig(dim=dim, action_dim=action_dim)

    for scale in [0.0, 0.5, 1.0, 2.0]:
        eval_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=seed)
        index_dim = dim if scale == 0.0 else dim + action_dim
        memory = Memory(config, index=FlatIndex(dim=index_dim), action_scale=scale)
        predictor = ConcatFusionPredictor(eval_adapter, alpha=1.0)
        result = evaluate_with_retrieval(memory_episodes, eval_episodes, memory, predictor, k=k)
        label = "appearance-only (scale=0.0)" if scale == 0.0 else f"action-conditioned (scale={scale})"
        rows.append((label, result["mean_error"]))

    print(f"{'method':32} {'mean_error':>12}")
    print("-" * 45)
    for label, mean_error in rows:
        print(f"{label:32} {mean_error:12.6f}")


if __name__ == "__main__":
    main()
