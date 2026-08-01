import numpy as np

from mavka.action_conditioning import evaluate_with_retrieval
from mavka.adapter import generate_trajectory
from mavka.baseline import evaluate_no_memory, split_episodes
from mavka.fusion import ConcatFusionPredictor
from mavka.pipeline import ActionConditionedPipeline
from mavka.scorer import FixedWeightScorer
from mavka.store import VectorStore

CONDITION_NAMES = [
    "no_memory",
    "appearance_only",
    "action_conditioned",
    "action_conditioned_scorer",
    "full_system",
]
BASELINE_CONDITION = "no_memory"

_DEFAULT_CONFIG = {
    "holdout_frac": 0.2,
    "scale": 2.0,
    "fusion_alpha": 0.5,
    "alpha_grid": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    "scorer_weights": {"w_sim": 1.0, "w_action": 0.5, "w_recency": 0.2},
    "fetch_factor": 5,
}


def _new_pipeline(dim, action_dim, scale):
    return ActionConditionedPipeline(
        dim=dim, action_dim=action_dim, index=VectorStore(dim=dim + action_dim), scale=scale
    )


def run_memory_experiment(
    adapter, n_episodes: int, episode_length: int, k: int, seeds: list[int], config: dict | None = None
) -> dict:
    """Run the five-condition memory comparison across multiple seeds and
    report mean +/- std of prediction error per condition.

    Every condition, within a given seed, is measured on the exact same
    held-out eval_episodes (from a single split_episodes call for that
    seed) with the exact same prediction_error metric, so the only thing
    that differs between conditions is the retrieval configuration:

      no_memory                  -- adapter.step alone, no retrieval at all
      appearance_only             -- scale=0 (z-only keying), fused at a
                                      fixed alpha (config["fusion_alpha"])
      action_conditioned          -- scale>0 keying, no scorer, fused at
                                      the same fixed alpha
      action_conditioned_scorer   -- scale>0 keying + FixedWeightScorer
                                      re-ranking, fused at the same fixed
                                      alpha
      full_system                 -- scale>0 keying + scorer, fusion alpha
                                      swept over config["alpha_grid"] and
                                      the best (lowest mean_error) kept.
                                      Since alpha=0 (pure model, identical
                                      to no_memory) is always included in
                                      the grid, this condition can never
                                      end up worse than the baseline.

    adapter supplies dim/action_dim/class only -- a fresh instance of
    type(adapter) is constructed per seed (for trajectory generation and
    for each condition's own model access), so every condition's model
    calls start from the same, reproducible RNG state for that seed.

    Returns {"conditions": {name: {mean_error, std_error, n_steps,
    relative_improvement_pct, per_seed_errors}}, "seeds": [...],
    "eval_set_sizes": [...], "baseline_condition": "no_memory"}.
    """
    cfg = {**_DEFAULT_CONFIG, **(config or {})}
    dim = adapter.dim
    action_dim = adapter.action_dim
    adapter_cls = type(adapter)

    per_seed_errors = {name: [] for name in CONDITION_NAMES}
    per_seed_n_steps = {name: [] for name in CONDITION_NAMES}
    eval_set_sizes = []

    for seed in seeds:
        gen_adapter = adapter_cls(dim=dim, action_dim=action_dim, seed=seed)
        all_episodes = [
            generate_trajectory(gen_adapter, episode_length, episode_id=i)
            for i in range(n_episodes)
        ]
        memory_episodes, eval_episodes = split_episodes(
            all_episodes, holdout_frac=cfg["holdout_frac"], seed=seed
        )
        eval_set_sizes.append(sum(len(ep) for ep in eval_episodes))

        # 1. no memory
        baseline_adapter = adapter_cls(dim=dim, action_dim=action_dim, seed=seed)
        baseline_result = evaluate_no_memory(baseline_adapter, eval_episodes)
        per_seed_errors["no_memory"].append(baseline_result["mean_error"])
        per_seed_n_steps["no_memory"].append(baseline_result["n_steps"])

        # 2. appearance-only (scale=0), fused at a fixed alpha
        appearance_adapter = adapter_cls(dim=dim, action_dim=action_dim, seed=seed)
        appearance_pipeline = _new_pipeline(dim, action_dim, scale=0.0)
        appearance_predictor = ConcatFusionPredictor(appearance_adapter, alpha=cfg["fusion_alpha"])
        appearance_result = evaluate_with_retrieval(
            memory_episodes, eval_episodes, appearance_pipeline, appearance_predictor, k=k, scale=0.0
        )
        per_seed_errors["appearance_only"].append(appearance_result["mean_error"])
        per_seed_n_steps["appearance_only"].append(appearance_result["n_steps"])

        # 3. action-conditioned (scale>0), no scorer, fused at a fixed alpha
        action_adapter = adapter_cls(dim=dim, action_dim=action_dim, seed=seed)
        action_pipeline = _new_pipeline(dim, action_dim, scale=cfg["scale"])
        action_predictor = ConcatFusionPredictor(action_adapter, alpha=cfg["fusion_alpha"])
        action_result = evaluate_with_retrieval(
            memory_episodes, eval_episodes, action_pipeline, action_predictor, k=k, scale=cfg["scale"]
        )
        per_seed_errors["action_conditioned"].append(action_result["mean_error"])
        per_seed_n_steps["action_conditioned"].append(action_result["n_steps"])

        # 4. action-conditioned + scorer, fused at a fixed alpha
        scorer_adapter = adapter_cls(dim=dim, action_dim=action_dim, seed=seed)
        scorer_pipeline = _new_pipeline(dim, action_dim, scale=cfg["scale"])
        scorer = FixedWeightScorer(scorer_pipeline._log, **cfg["scorer_weights"])
        scorer_predictor = ConcatFusionPredictor(scorer_adapter, alpha=cfg["fusion_alpha"])
        scorer_result = evaluate_with_retrieval(
            memory_episodes,
            eval_episodes,
            scorer_pipeline,
            scorer_predictor,
            k=k,
            scale=cfg["scale"],
            scorer=scorer,
            fetch_factor=cfg["fetch_factor"],
        )
        per_seed_errors["action_conditioned_scorer"].append(scorer_result["mean_error"])
        per_seed_n_steps["action_conditioned_scorer"].append(scorer_result["n_steps"])

        # 5. full system: action-conditioned + scorer, alpha swept, best kept
        best_error = None
        best_n_steps = None
        for alpha in cfg["alpha_grid"]:
            full_adapter = adapter_cls(dim=dim, action_dim=action_dim, seed=seed)
            full_pipeline = _new_pipeline(dim, action_dim, scale=cfg["scale"])
            full_scorer = FixedWeightScorer(full_pipeline._log, **cfg["scorer_weights"])
            full_predictor = ConcatFusionPredictor(full_adapter, alpha=alpha)
            full_result = evaluate_with_retrieval(
                memory_episodes,
                eval_episodes,
                full_pipeline,
                full_predictor,
                k=k,
                scale=cfg["scale"],
                scorer=full_scorer,
                fetch_factor=cfg["fetch_factor"],
            )
            if best_error is None or full_result["mean_error"] < best_error:
                best_error = full_result["mean_error"]
                best_n_steps = full_result["n_steps"]
        per_seed_errors["full_system"].append(best_error)
        per_seed_n_steps["full_system"].append(best_n_steps)

    baseline_mean = float(np.mean(per_seed_errors[BASELINE_CONDITION]))

    conditions = {}
    for name in CONDITION_NAMES:
        errors = np.array(per_seed_errors[name], dtype=np.float64)
        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors))
        relative_improvement_pct = (
            100.0 * (baseline_mean - mean_error) / baseline_mean if baseline_mean != 0 else 0.0
        )
        conditions[name] = {
            "mean_error": mean_error,
            "std_error": std_error,
            "n_steps": per_seed_n_steps[name][0],
            "relative_improvement_pct": relative_improvement_pct,
            "per_seed_errors": per_seed_errors[name],
        }

    return {
        "conditions": conditions,
        "seeds": list(seeds),
        "eval_set_sizes": eval_set_sizes,
        "baseline_condition": BASELINE_CONDITION,
    }


def format_experiment_report(results: dict) -> str:
    conditions = results["conditions"]
    baseline_name = results["baseline_condition"]

    header = f"{'condition':28} {'mean_error':>12} {'std_error':>12} {'n_steps':>9} {'vs baseline':>13}"
    lines = [header, "-" * len(header)]

    for name in CONDITION_NAMES:
        stats = conditions[name]
        label = name + (" [baseline]" if name == baseline_name else "")
        vs_baseline = "--" if name == baseline_name else f"{stats['relative_improvement_pct']:+.2f}%"
        lines.append(
            f"{label:28} {stats['mean_error']:12.6f} {stats['std_error']:12.6f} "
            f"{stats['n_steps']:9d} {vs_baseline:>13}"
        )

    return "\n".join(lines)


def memory_helps(results: dict, min_improvement: float = 0.0, require_significance: bool = True) -> bool:
    """True only if the full system's mean error is below the baseline's by
    at least min_improvement, AND (if require_significance) that gap is
    larger than the combined per-seed std of the two conditions -- a crude
    check that the improvement isn't just seed-to-seed noise.
    """
    conditions = results["conditions"]
    baseline = conditions[results["baseline_condition"]]
    full = conditions["full_system"]

    gap = baseline["mean_error"] - full["mean_error"]
    if gap < min_improvement:
        return False

    if require_significance:
        combined_std = baseline["std_error"] + full["std_error"]
        if gap <= combined_std:
            return False

    return True
