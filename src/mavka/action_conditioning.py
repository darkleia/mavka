import numpy as np

from mavka.baseline import prediction_error
from mavka.keying import make_keys_batch


def evaluate_with_retrieval(
    adapter, memory_episodes, eval_episodes, pipeline, k: int, scale: float
) -> dict:
    """Fill pipeline's memory from memory_episodes (keyed at the given
    scale), then for each held-out eval step, retrieve the k nearest past
    experiences and predict z_next as the plain mean of what actually
    followed those experiences (via the log's next_in_episode) -- no
    fusion, no scoring, deliberately dumb. adapter is used only as a
    fallback (adapter.step) for the rare step where none of the k
    retrieved neighbors have a next state at all (e.g. an empty pipeline,
    or every match being the last step of its episode).
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
            results = pipeline.recall(step["z"], step["action"], k)

            next_zs = []
            for id_, _score in results:
                next_record = pipeline._log.next_in_episode(id_)
                if next_record is not None:
                    next_zs.append(next_record.z)

            if next_zs:
                predicted_z_next = np.mean(np.stack(next_zs), axis=0)
            else:
                predicted_z_next = adapter.step(step["z"], step["action"])

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
    from an empty index), and return the scale with the lowest mean_error
    on validation_episodes. Plain grid search, not gradient-based tuning.
    """
    best_scale = None
    best_error = float("inf")

    for scale in scale_grid:
        pipeline = pipeline_factory()
        result = evaluate_with_retrieval(
            adapter, memory_episodes, validation_episodes, pipeline, k, scale
        )
        if result["mean_error"] < best_error:
            best_error = result["mean_error"]
            best_scale = scale

    return best_scale
