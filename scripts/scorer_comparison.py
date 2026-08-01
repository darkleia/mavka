from mavka.action_conditioning import evaluate_with_retrieval
from mavka.adapter import SyntheticWorldModel, generate_trajectory
from mavka.baseline import evaluate_no_memory, split_episodes
from mavka.pipeline import ActionConditionedPipeline
from mavka.scorer import FixedWeightScorer
from mavka.store import VectorStore


def main() -> None:
    dim = 32
    action_dim = 4
    n_episodes = 100
    episode_length = 40
    holdout_frac = 0.2
    seed = 0
    k = 10
    scale = 2.0
    fetch_factor = 5

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

    baseline_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=seed)
    baseline_result = evaluate_no_memory(baseline_adapter, eval_episodes)
    rows.append(("no-memory baseline", baseline_result["mean_error"]))

    no_scorer_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=seed)
    no_scorer_result = evaluate_with_retrieval(
        no_scorer_adapter, memory_episodes, eval_episodes, new_pipeline(), k=k, scale=scale
    )
    rows.append(("action-conditioned, no scorer", no_scorer_result["mean_error"]))

    scorer_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=seed)
    scorer_pipeline = new_pipeline()
    scorer = FixedWeightScorer(scorer_pipeline._log)
    scorer_result = evaluate_with_retrieval(
        scorer_adapter,
        memory_episodes,
        eval_episodes,
        scorer_pipeline,
        k=k,
        scale=scale,
        scorer=scorer,
        fetch_factor=fetch_factor,
    )
    rows.append(("action-conditioned, with scorer", scorer_result["mean_error"]))

    print(f"{'method':36} {'mean_error':>12}")
    print("-" * 49)
    for label, mean_error in rows:
        print(f"{label:36} {mean_error:12.6f}")

    print("\nablation: each weight isolated (others zeroed)")
    print(f"{'weights':36} {'mean_error':>12}")
    print("-" * 49)
    ablation_weights = [
        ("w_sim only", dict(w_sim=1.0, w_action=0.0, w_recency=0.0)),
        ("w_action only", dict(w_sim=0.0, w_action=1.0, w_recency=0.0)),
        ("w_recency only", dict(w_sim=0.0, w_action=0.0, w_recency=1.0)),
    ]
    for label, weights in ablation_weights:
        ablation_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=seed)
        ablation_pipeline = new_pipeline()
        ablation_scorer = FixedWeightScorer(ablation_pipeline._log, **weights)
        result = evaluate_with_retrieval(
            ablation_adapter,
            memory_episodes,
            eval_episodes,
            ablation_pipeline,
            k=k,
            scale=scale,
            scorer=ablation_scorer,
            fetch_factor=fetch_factor,
        )
        print(f"{label:36} {result['mean_error']:12.6f}")


if __name__ == "__main__":
    main()
