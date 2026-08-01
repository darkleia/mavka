import functools

import numpy as np

from mavka.adapter import generate_trajectory
from mavka.config import MavkaConfig
from mavka.eval.baseline import evaluate_no_memory, prediction_error, split_episodes
from mavka.eval.retrieval_eval import evaluate_with_retrieval
from mavka.graph.adjacency import AdjacencyStore
from mavka.graph.builder import EdgeBuilder
from mavka.graph.expand import expand
from mavka.index.flat import FlatIndex
from mavka.lifecycle.eviction import EvictionPolicy
from mavka.lifecycle.feedback import FeedbackBuffer
from mavka.lifecycle.maintenance import MaintenanceWorker
from mavka.memory import Memory
from mavka.retrieval.fusion import ConcatFusionPredictor, build_context
from mavka.retrieval.scorer import FixedWeightScorer

CONDITION_NAMES = [
    "no_memory",
    "appearance_only",
    "action_conditioned",
    "action_conditioned_scorer",
    "full_system",
    "full_system_no_graph",
    "full_system_graph_depth1",
    "full_system_graph_depth2",
    "full_system_with_maintenance",
]
BASELINE_CONDITION = "no_memory"
GRAPH_DEPTHS = (1, 2)

_DEFAULT_CONFIG = {
    "holdout_frac": 0.2,
    "scale": 2.0,
    "fusion_alpha": 0.5,
    "alpha_grid": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    "scorer_weights": {"w_sim": 1.0, "w_action": 0.5, "w_recency": 0.2},
    "fetch_factor": 5,
    "graph_degree": 8,
    "n_analogous": 4,
    "similarity_threshold": 0.3,
    "temporal_weight": 1.0,
    "max_nodes": 200,
    "maintenance_capacity": None,  # set per-experiment; see _run_full_system_with_maintenance
    "maintenance_compaction_threshold": None,
    "maintenance_cadence_episodes": 10,
}


def _new_memory(
    dim, action_dim, scale, fetch_factor=5, graph=None, expansion_depth=0, expander=None
) -> Memory:
    config = MavkaConfig(dim=dim, action_dim=action_dim)
    index_dim = dim if scale == 0.0 else dim + action_dim
    return Memory(
        config,
        index=FlatIndex(dim=index_dim),
        action_scale=scale,
        fetch_factor=fetch_factor,
        graph=graph,
        expansion_depth=expansion_depth,
        expander=expander,
    )


def _build_graph(dim, action_dim, memory_episodes, cfg) -> AdjacencyStore:
    """Build the temporal + analogous edge graph for this seed's memory
    set, against its own throwaway z-only (action_scale=0) Memory
    instance -- analogous search is z-only by design (see
    graph/builder.py), independent of whatever action-conditioned memory
    is used for retrieval elsewhere in this experiment. Node ids still
    line up with the retrieval memory's record ids, since both are built
    by observing the exact same memory_episodes in the exact same order,
    and ids are assigned purely by insertion sequence.

    Memory itself has no built-in edge-building hook on observe() (unlike
    old Pipeline's graph=/edge_builder= constructor args) -- this helper
    drives EdgeBuilder.on_insert itself, once per observed record, using
    that throwaway memory's own log/index directly.
    """
    graph = AdjacencyStore(degree=cfg["graph_degree"])
    edge_builder = EdgeBuilder(
        n_analogous=cfg["n_analogous"],
        similarity_threshold=cfg["similarity_threshold"],
        temporal_weight=cfg["temporal_weight"],
    )
    graph_memory = _new_memory(dim, action_dim, scale=0.0)

    for episode in memory_episodes:
        for step in episode:
            record_id = graph_memory.observe(
                z=step["z"],
                action=step["action"],
                z_next=step["z_next"],
                pred_err=step["pred_err"],
                episode_id=step["episode_id"],
            )
            graph.add_node()  # kept in lockstep with record_id by construction
            record = graph_memory.get(record_id)
            edge_builder.on_insert(
                record_id,
                record.z,
                record.action,
                record.episode_id,
                record.seq_no,
                graph_memory._log,
                graph_memory._index,
                graph,
            )
    return graph


def _accumulate_maintenance(totals: dict, report: dict) -> None:
    totals["feedback_events_applied"] += report["feedback_events_applied"]
    if report["compaction"] is not None:
        totals["records_compacted"] += (
            report["compaction"]["records_merged"] + report["compaction"]["tombstones_dropped"]
        )
    if report["eviction"] is not None:
        totals["records_evicted"] += len(report["eviction"]["evicted_ids"])
    if report["migration"] is not None:
        totals["records_migrated"] += len(report["migration"]["migrated_ids"])


def _run_full_system_with_maintenance(
    memory_episodes,
    eval_episodes,
    adapter,
    k: int,
    cfg: dict,
    best_alpha: float,
    maintenance_capacity: int,
    maintenance_compaction_threshold: int,
) -> tuple[dict, dict]:
    """Same retrieval configuration as full_system (FixedWeightScorer at
    cfg["scorer_weights"], fusion alpha held at full_system's own best,
    not re-swept -- same pattern full_system_no_graph/graph_depth* already
    use) -- but action_scale=0.0 instead of cfg["scale"].

    Why: wiring this surfaced a genuine, confirmed incompatibility in the
    existing (until now unwired) lifecycle code -- compact()'s index
    rebuild always re-inserts plain z vectors (see
    lifecycle/compaction.py), which raises immediately against an
    action-conditioned (dim + action_dim keyed) index; evict_to_capacity
    hits the same wall since it calls compact() internally. This is not
    something this wiring step patches around -- see Memory.compact()'s
    docstring and MaintenanceWorker's module docstring for the full
    writeup. So this condition demonstrates the lifecycle flow correctly
    and honestly at action_scale=0.0 (appearance-only keying, the other
    axis full_system already isolates elsewhere in this experiment),
    rather than forcing an incompatible composition on cfg["scale"].

    An EvictionPolicy + FeedbackBuffer are attached, and
    MaintenanceWorker.step() is called: every
    cfg["maintenance_cadence_episodes"] memory episodes while filling
    memory (a periodic background pass -- not on every single
    observation, which would defeat the point of it being off the hot
    path), once more right after memory is fully filled, and once more
    after the eval pass (so feedback tagged during eval -- see below --
    still gets drained and counted). Each eval step tags that step's own
    retrieval outcome via memory.last_feedback_token, using
    evaluate_gated's own "helped" definition: memory-augmented error <
    model-alone error.

    Returns (result, maintenance_totals): result matches
    evaluate_with_retrieval's own shape (mean_error, median_error,
    p90_error, n_steps, errors); maintenance_totals sums every
    worker.step() report's counters across the whole run:
    {feedback_events_applied, records_compacted, records_evicted,
    records_migrated}.
    """
    config = MavkaConfig(dim=adapter.dim, action_dim=adapter.action_dim)
    eviction_policy = EvictionPolicy()
    feedback_buffer = FeedbackBuffer()
    memory = Memory(
        config,
        index=FlatIndex(dim=adapter.dim),
        action_scale=0.0,
        fetch_factor=cfg["fetch_factor"],
        eviction_policy=eviction_policy,
        feedback_buffer=feedback_buffer,
    )
    memory.scorer = FixedWeightScorer(memory._log, **cfg["scorer_weights"])

    worker = MaintenanceWorker(
        memory,
        eviction_policy=eviction_policy,
        feedback_buffer=feedback_buffer,
        capacity=maintenance_capacity,
        compaction_threshold=maintenance_compaction_threshold,
    )

    totals = {
        "feedback_events_applied": 0,
        "records_compacted": 0,
        "records_evicted": 0,
        "records_migrated": 0,
    }

    cadence = cfg["maintenance_cadence_episodes"]
    for i, episode in enumerate(memory_episodes):
        for step in episode:
            memory.observe(
                z=step["z"],
                action=step["action"],
                z_next=step["z_next"],
                pred_err=step["pred_err"],
                episode_id=step["episode_id"],
            )
        if (i + 1) % cadence == 0:
            _accumulate_maintenance(totals, worker.step())

    _accumulate_maintenance(totals, worker.step())  # once more after filling

    predictor = ConcatFusionPredictor(adapter, alpha=best_alpha)
    errors = []
    for episode in eval_episodes:
        for step in episode:
            base = np.asarray(adapter.step(step["z"], step["action"]), dtype=np.float32)
            base_error = prediction_error(base, step["z_next"])

            results = memory.recall(step["z"], action=step["action"], k=k)
            context = build_context(results, memory._log, k)
            predicted = predictor.predict(step["z"], step["action"], context)
            error = prediction_error(predicted, step["z_next"])

            feedback_buffer.tag_outcome(memory.last_feedback_token, helped=error < base_error)
            errors.append(error)

    _accumulate_maintenance(totals, worker.step())  # drain feedback tagged during eval

    errors_arr = np.array(errors, dtype=np.float64)
    result = {
        "mean_error": float(np.mean(errors_arr)),
        "median_error": float(np.median(errors_arr)),
        "p90_error": float(np.percentile(errors_arr, 90)),
        "n_steps": len(errors),
        "errors": errors_arr.tolist(),
    }
    return result, totals


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
      full_system_no_graph        -- exactly full_system's result (same
                                      config, same alpha), just labeled
                                      separately so the graph-ablation
                                      conditions below have a same-name-
                                      family baseline to compare against
                                      without having to reference
                                      full_system by a different name.
      full_system_graph_depth1,
      full_system_graph_depth2    -- full_system's own best alpha (held
                                      fixed, not re-swept) plus graph
                                      expansion at depth 1 or 2 -- the
                                      graph is the *only* thing that
                                      differs from full_system_no_graph,
                                      isolating its marginal contribution.
      full_system_with_maintenance -- full_system's own scorer weights and
                                      best alpha, but action_scale=0.0 (see
                                      _run_full_system_with_maintenance's
                                      docstring for why -- a genuine,
                                      confirmed incompatibility between
                                      compact()/evict_to_capacity() and an
                                      action-conditioned index), plus an
                                      EvictionPolicy + FeedbackBuffer +
                                      MaintenanceWorker driving the
                                      previously-unwired background flow
                                      (feedback drain, compaction,
                                      eviction) during the run.

    adapter supplies dim/action_dim/class only -- a fresh instance of
    type(adapter) is constructed per seed (for trajectory generation and
    for each condition's own model access), so every condition's model
    calls start from the same, reproducible RNG state for that seed.

    Returns {"conditions": {name: {mean_error, std_error, n_steps,
    relative_improvement_pct, per_seed_errors}}, "seeds": [...],
    "eval_set_sizes": [...], "baseline_condition": "no_memory",
    "maintenance": {stat_name: {mean, std, per_seed}} for
    feedback_events_applied/records_compacted/records_evicted/
    records_migrated across the full_system_with_maintenance runs}.
    """
    cfg = {**_DEFAULT_CONFIG, **(config or {})}
    dim = adapter.dim
    action_dim = adapter.action_dim
    adapter_cls = type(adapter)

    per_seed_errors = {name: [] for name in CONDITION_NAMES}
    per_seed_n_steps = {name: [] for name in CONDITION_NAMES}
    per_seed_maintenance = {
        "feedback_events_applied": [],
        "records_compacted": [],
        "records_evicted": [],
        "records_migrated": [],
    }
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
        appearance_memory = _new_memory(dim, action_dim, scale=0.0)
        appearance_predictor = ConcatFusionPredictor(appearance_adapter, alpha=cfg["fusion_alpha"])
        appearance_result = evaluate_with_retrieval(
            memory_episodes, eval_episodes, appearance_memory, appearance_predictor, k=k
        )
        per_seed_errors["appearance_only"].append(appearance_result["mean_error"])
        per_seed_n_steps["appearance_only"].append(appearance_result["n_steps"])

        # 3. action-conditioned (scale>0), no scorer, fused at a fixed alpha
        action_adapter = adapter_cls(dim=dim, action_dim=action_dim, seed=seed)
        action_memory = _new_memory(dim, action_dim, scale=cfg["scale"])
        action_predictor = ConcatFusionPredictor(action_adapter, alpha=cfg["fusion_alpha"])
        action_result = evaluate_with_retrieval(
            memory_episodes, eval_episodes, action_memory, action_predictor, k=k
        )
        per_seed_errors["action_conditioned"].append(action_result["mean_error"])
        per_seed_n_steps["action_conditioned"].append(action_result["n_steps"])

        # 4. action-conditioned + scorer, fused at a fixed alpha
        scorer_adapter = adapter_cls(dim=dim, action_dim=action_dim, seed=seed)
        scorer_memory = _new_memory(dim, action_dim, scale=cfg["scale"], fetch_factor=cfg["fetch_factor"])
        scorer_memory.scorer = FixedWeightScorer(scorer_memory._log, **cfg["scorer_weights"])
        scorer_predictor = ConcatFusionPredictor(scorer_adapter, alpha=cfg["fusion_alpha"])
        scorer_result = evaluate_with_retrieval(
            memory_episodes, eval_episodes, scorer_memory, scorer_predictor, k=k
        )
        per_seed_errors["action_conditioned_scorer"].append(scorer_result["mean_error"])
        per_seed_n_steps["action_conditioned_scorer"].append(scorer_result["n_steps"])

        # 5. full system: action-conditioned + scorer, alpha swept, best kept
        best_error = None
        best_n_steps = None
        best_alpha = None
        for alpha in cfg["alpha_grid"]:
            full_adapter = adapter_cls(dim=dim, action_dim=action_dim, seed=seed)
            full_memory = _new_memory(dim, action_dim, scale=cfg["scale"], fetch_factor=cfg["fetch_factor"])
            full_memory.scorer = FixedWeightScorer(full_memory._log, **cfg["scorer_weights"])
            full_predictor = ConcatFusionPredictor(full_adapter, alpha=alpha)
            full_result = evaluate_with_retrieval(
                memory_episodes, eval_episodes, full_memory, full_predictor, k=k
            )
            if best_error is None or full_result["mean_error"] < best_error:
                best_error = full_result["mean_error"]
                best_n_steps = full_result["n_steps"]
                best_alpha = alpha
        per_seed_errors["full_system"].append(best_error)
        per_seed_n_steps["full_system"].append(best_n_steps)

        # full_system_no_graph is exactly full_system's own result -- same
        # config, same already-found best alpha -- just labeled separately
        # so the graph-ablation conditions have a same-family name to
        # compare against in the report.
        per_seed_errors["full_system_no_graph"].append(best_error)
        per_seed_n_steps["full_system_no_graph"].append(best_n_steps)

        # 6/7. full system + graph, at full_system's own best alpha (held
        # fixed, not re-swept) -- the graph is the only thing that varies
        # relative to full_system_no_graph.
        graph = _build_graph(dim, action_dim, memory_episodes, cfg)
        for depth in GRAPH_DEPTHS:
            depth_adapter = adapter_cls(dim=dim, action_dim=action_dim, seed=seed)
            depth_memory = _new_memory(
                dim,
                action_dim,
                scale=cfg["scale"],
                fetch_factor=cfg["fetch_factor"],
                graph=graph,
                expansion_depth=depth,
                expander=functools.partial(expand, max_nodes=cfg["max_nodes"], edge_types=None),
            )
            depth_memory.scorer = FixedWeightScorer(depth_memory._log, **cfg["scorer_weights"])
            depth_predictor = ConcatFusionPredictor(depth_adapter, alpha=best_alpha)
            depth_result = evaluate_with_retrieval(
                memory_episodes, eval_episodes, depth_memory, depth_predictor, k=k
            )
            name = f"full_system_graph_depth{depth}"
            per_seed_errors[name].append(depth_result["mean_error"])
            per_seed_n_steps[name].append(depth_result["n_steps"])

        # 8. full system + maintenance: same scorer/best_alpha as
        # full_system, but action_scale=0.0 (see
        # _run_full_system_with_maintenance's docstring for why) plus an
        # eviction policy, feedback buffer, and periodic
        # MaintenanceWorker.step() calls -- the only new variable versus
        # full_system_no_graph is the maintenance flow itself.
        total_memory_steps = sum(len(ep) for ep in memory_episodes)
        maintenance_capacity = cfg["maintenance_capacity"]
        if maintenance_capacity is None:
            maintenance_capacity = max(1, int(total_memory_steps * 0.8))
        maintenance_compaction_threshold = cfg["maintenance_compaction_threshold"]
        if maintenance_compaction_threshold is None:
            maintenance_compaction_threshold = max(1, int(total_memory_steps * 0.5))

        maintenance_adapter = adapter_cls(dim=dim, action_dim=action_dim, seed=seed)
        maintenance_result, maintenance_totals = _run_full_system_with_maintenance(
            memory_episodes,
            eval_episodes,
            maintenance_adapter,
            k,
            cfg,
            best_alpha,
            maintenance_capacity,
            maintenance_compaction_threshold,
        )
        per_seed_errors["full_system_with_maintenance"].append(maintenance_result["mean_error"])
        per_seed_n_steps["full_system_with_maintenance"].append(maintenance_result["n_steps"])
        for stat_name, value in maintenance_totals.items():
            per_seed_maintenance[stat_name].append(value)

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

    maintenance = {
        stat_name: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "per_seed": values,
        }
        for stat_name, values in per_seed_maintenance.items()
    }

    return {
        "conditions": conditions,
        "seeds": list(seeds),
        "eval_set_sizes": eval_set_sizes,
        "baseline_condition": BASELINE_CONDITION,
        "maintenance": maintenance,
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

    if "maintenance" in results:
        lines.append("")
        lines.append("maintenance activity (full_system_with_maintenance, summed across seeds):")
        for stat_name in (
            "feedback_events_applied",
            "records_compacted",
            "records_evicted",
            "records_migrated",
        ):
            stat = results["maintenance"][stat_name]
            lines.append(f"  {stat_name:28} mean={stat['mean']:8.1f}  std={stat['std']:6.1f}")

    return "\n".join(lines)


def _significant_improvement(
    reference: dict, candidate: dict, min_improvement: float, require_significance: bool
) -> bool:
    """Shared significance guard: candidate's mean_error must be below
    reference's by at least min_improvement, and (if require_significance)
    that gap must exceed the two conditions' combined per-seed std -- a
    crude check that the improvement isn't just seed-to-seed noise.
    """
    gap = reference["mean_error"] - candidate["mean_error"]
    if gap < min_improvement:
        return False

    if require_significance:
        combined_std = reference["std_error"] + candidate["std_error"]
        if gap <= combined_std:
            return False

    return True


def memory_helps(results: dict, min_improvement: float = 0.0, require_significance: bool = True) -> bool:
    """True only if the full system's mean error is significantly below
    the no-memory baseline's (see _significant_improvement)."""
    conditions = results["conditions"]
    baseline = conditions[results["baseline_condition"]]
    full = conditions["full_system"]
    return _significant_improvement(baseline, full, min_improvement, require_significance)


def graph_helps(
    results: dict, depth: int = 2, min_improvement: float = 0.0, require_significance: bool = True
) -> bool:
    """True only if full_system_graph_depth{depth}'s mean error is
    significantly below full_system_no_graph's (see
    _significant_improvement) -- isolates the graph's own marginal
    contribution on top of the rest of the full system, which is held
    fixed (same alpha, same scale, same scorer weights) between the two.
    """
    conditions = results["conditions"]
    no_graph = conditions["full_system_no_graph"]
    with_graph = conditions[f"full_system_graph_depth{depth}"]
    return _significant_improvement(no_graph, with_graph, min_improvement, require_significance)
