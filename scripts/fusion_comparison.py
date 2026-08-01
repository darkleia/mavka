from mavka.action_conditioning import evaluate_with_retrieval
from mavka.adapter import SyntheticWorldModel, generate_trajectory
from mavka.eval.baseline import evaluate_no_memory, split_episodes
from mavka.retrieval.fusion import ConcatFusionPredictor
from mavka.pipeline import ActionConditionedPipeline
from mavka.index.flat import FlatIndex


def main() -> None:
    dim = 32
    action_dim = 4
    n_episodes = 100
    episode_length = 40
    holdout_frac = 0.2
    seed = 0
    k = 10
    scale = 2.0

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

    def new_pipeline():
        return ActionConditionedPipeline(
            dim=dim, action_dim=action_dim, index=FlatIndex(dim=dim + action_dim), scale=scale
        )

    rows = []

    baseline_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=seed)
    baseline_result = evaluate_no_memory(baseline_adapter, eval_episodes)
    rows.append(("no-memory baseline (alpha=0.0)", baseline_result["mean_error"]))

    for alpha in [0.25, 0.5, 0.75, 1.0]:
        fusion_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=seed)
        predictor = ConcatFusionPredictor(fusion_adapter, alpha=alpha)
        result = evaluate_with_retrieval(
            memory_episodes, eval_episodes, new_pipeline(), predictor, k=k, scale=scale
        )
        label = "pure memory (alpha=1.0)" if alpha == 1.0 else f"concat fusion (alpha={alpha})"
        rows.append((label, result["mean_error"]))

    print(f"{'method':32} {'mean_error':>12}")
    print("-" * 45)
    for label, mean_error in rows:
        print(f"{label:32} {mean_error:12.6f}")


if __name__ == "__main__":
    main()
