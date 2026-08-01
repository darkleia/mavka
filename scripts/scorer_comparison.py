from mavka.adapter import SyntheticWorldModel, generate_trajectory
from mavka.config import MavkaConfig
from mavka.eval.baseline import evaluate_no_memory, split_episodes
from mavka.eval.retrieval_eval import evaluate_with_retrieval
from mavka.index.flat import FlatIndex
from mavka.memory import Memory
from mavka.retrieval.fusion import ConcatFusionPredictor
from mavka.retrieval.scorer import FixedWeightScorer


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

    config = MavkaConfig(dim=dim, action_dim=action_dim)

    def new_memory(fetch_factor=5):
        return Memory(
            config, index=FlatIndex(dim=dim + action_dim), action_scale=scale, fetch_factor=fetch_factor
        )

    rows = []

    baseline_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=seed)
    baseline_result = evaluate_no_memory(baseline_adapter, eval_episodes)
    rows.append(("no-memory baseline", baseline_result["mean_error"]))

    no_scorer_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=seed)
    no_scorer_predictor = ConcatFusionPredictor(no_scorer_adapter, alpha=1.0)
    no_scorer_result = evaluate_with_retrieval(
        memory_episodes, eval_episodes, new_memory(), no_scorer_predictor, k=k
    )
    rows.append(("action-conditioned, no scorer", no_scorer_result["mean_error"]))

    scorer_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=seed)
    scorer_memory = new_memory(fetch_factor=fetch_factor)
    scorer_memory.scorer = FixedWeightScorer(scorer_memory._log)
    scorer_predictor = ConcatFusionPredictor(scorer_adapter, alpha=1.0)
    scorer_result = evaluate_with_retrieval(
        memory_episodes, eval_episodes, scorer_memory, scorer_predictor, k=k
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
        ablation_memory = new_memory(fetch_factor=fetch_factor)
        ablation_memory.scorer = FixedWeightScorer(ablation_memory._log, **weights)
        ablation_predictor = ConcatFusionPredictor(ablation_adapter, alpha=1.0)
        result = evaluate_with_retrieval(
            memory_episodes, eval_episodes, ablation_memory, ablation_predictor, k=k
        )
        print(f"{label:36} {result['mean_error']:12.6f}")


if __name__ == "__main__":
    main()
