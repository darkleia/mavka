import numpy as np

from mavka.eval.baseline import prediction_error
from mavka.retrieval.fusion import ConcatFusionPredictor, build_context
from mavka.retrieval.keying import make_keys_batch


def evaluate_with_retrieval(
    memory_episodes,
    eval_episodes,
    pipeline,
    predictor,
    k: int,
    scale: float,
    scorer=None,
    fetch_factor: int = 5,
    graph=None,
    expand_depth: int = 0,
    max_nodes: int = 50,
    edge_types=None,
    use_graph: bool | None = None,
) -> dict:
    """Fill pipeline's memory from memory_episodes (keyed at the given
    scale), then for each held-out eval step, retrieve the k nearest past
    experiences, assemble them into a fixed-shape context (build_context),
    and ask predictor.predict(z, action, context) for a prediction of
    z_next -- keying, the scorer, the held-out split, and prediction_error
    are all unchanged from before; only how memories become a prediction
    is delegated to predictor now, instead of a hardcoded mean.

    If scorer is given, candidates are fetched via the pipeline's two-stage
    recall_scored (over-fetch k * fetch_factor by raw similarity, then
    re-rank with scorer) instead of the plain recall. If graph and
    expand_depth > 0 are also given, recall_scored's graph-expansion stage
    runs too (see Pipeline.recall_scored); expand_depth=0 (the default)
    never touches the graph, reproducing prior behavior exactly. use_graph
    is the clean on/off alias documented on recall_scored -- pass it
    explicitly to force expansion on or off independent of expand_depth's
    numeric value (e.g. to hold expand_depth=2 configured but still
    compare a use_graph=False run against it).
    """
    pipeline.scale = scale

    if hasattr(pipeline._index, "train") and not getattr(pipeline._index, "is_trained", True):
        zs = np.stack([step["z"] for ep in memory_episodes for step in ep])
        actions = (
            np.stack([step["action"] for ep in memory_episodes for step in ep])
            if pipeline.action_dim is not None
            else None
        )
        training_keys = make_keys_batch(zs, actions, scale, pipeline.action_dim)
        pipeline._index.train(training_keys)

    for episode in memory_episodes:
        for step in episode:
            pipeline.observe(
                z=step["z"],
                action=step["action"],
                z_next=step["z_next"],
                pred_err=step["pred_err"],
                episode_id=step["episode_id"],
            )

    errors = []
    for episode in eval_episodes:
        for step in episode:
            if scorer is not None:
                results = pipeline.recall_scored(
                    step["z"],
                    step["action"],
                    k,
                    scorer,
                    fetch_factor=fetch_factor,
                    graph=graph,
                    expand_depth=expand_depth,
                    max_nodes=max_nodes,
                    edge_types=edge_types,
                    use_graph=use_graph,
                )
            else:
                results = pipeline.recall(step["z"], step["action"], k)

            context = build_context(results, pipeline._log, k)
            predicted_z_next = predictor.predict(step["z"], step["action"], context)

            errors.append(prediction_error(predicted_z_next, step["z_next"]))

    errors_arr = np.array(errors, dtype=np.float64)

    return {
        "mean_error": float(np.mean(errors_arr)),
        "median_error": float(np.median(errors_arr)),
        "p90_error": float(np.percentile(errors_arr, 90)),
        "n_steps": len(errors),
        "errors": errors_arr.tolist(),
    }


def fit_scale(adapter, memory_episodes, validation_episodes, pipeline_factory, k: int, scale_grid):
    """Grid search: try each scale in scale_grid, build a fresh pipeline for
    each (via pipeline_factory(), a zero-arg callable so every trial starts
    from an empty index), evaluate with a pure score-weighted-memory
    predictor (alpha=1), and return the scale with the lowest mean_error on
    validation_episodes. Plain grid search, not gradient-based tuning.
    """
    best_scale = None
    best_error = float("inf")

    for scale in scale_grid:
        pipeline = pipeline_factory()
        predictor = ConcatFusionPredictor(adapter, alpha=1.0)
        result = evaluate_with_retrieval(
            memory_episodes, validation_episodes, pipeline, predictor, k, scale
        )
        if result["mean_error"] < best_error:
            best_error = result["mean_error"]
            best_scale = scale

    return best_scale
