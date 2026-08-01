from mavka.action_conditioning import evaluate_with_retrieval
from mavka.adapter import SyntheticWorldModel, generate_trajectory
from mavka.baseline import evaluate_no_memory, split_episodes
from mavka.fusion import ConcatFusionPredictor
from mavka.pipeline import ActionConditionedPipeline
from mavka.store import VectorStore
from mavka.trigger import SurpriseTrigger, evaluate_gated


def main() -> None:
    dim = 32
    action_dim = 4
    n_episodes = 100
    episode_length = 40
    holdout_frac = 0.2
    seed = 0
    k = 10
    scale = 1.0

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
            dim=dim, action_dim=action_dim, index=VectorStore(dim=dim + action_dim), scale=scale
        )

    rows = []

    never_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=seed)
    never_result = evaluate_no_memory(never_adapter, eval_episodes)
    rows.append(("never retrieve (baseline)", never_result["mean_error"], 0.0))

    for lam in [0.5, 1.0, 1.5, 2.5, 4.0]:
        gated_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=seed)
        trigger = SurpriseTrigger(smoothing=0.1, lam=lam, warmup=10)
        gated_result = evaluate_gated(
            memory_episodes, eval_episodes, new_pipeline(), gated_adapter, trigger, k=k, scale=scale
        )
        rows.append(
            (f"gated (lambda={lam})", gated_result["mean_error"], gated_result["retrieval_rate"])
        )

    always_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=seed)
    always_predictor = ConcatFusionPredictor(always_adapter, alpha=1.0)
    always_result = evaluate_with_retrieval(
        memory_episodes, eval_episodes, new_pipeline(), always_predictor, k=k, scale=scale
    )
    rows.append(("always retrieve", always_result["mean_error"], 1.0))

    print(f"{'condition':28} {'mean_error':>12} {'retrieval_rate':>16}")
    print("-" * 58)
    for label, mean_error, rate in rows:
        print(f"{label:28} {mean_error:12.6f} {rate:16.2%}")


if __name__ == "__main__":
    main()
