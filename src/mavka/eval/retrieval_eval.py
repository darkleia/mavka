import numpy as np

from mavka.eval.baseline import prediction_error
from mavka.retrieval.fusion import ConcatFusionPredictor, build_context
from mavka.retrieval.keying import make_keys_batch
from mavka.retrieval.trigger import SurpriseTrigger


def evaluate_gated(
    memory_episodes,
    eval_episodes,
    pipeline,
    adapter,
    trigger: SurpriseTrigger,
    k: int,
    scale: float,
    scorer=None,
    fetch_factor: int = 5,
    fusion_alpha: float = 1.0,
    feedback_buffer=None,
) -> dict:
    """Fill pipeline's memory, then for each held-out eval step: compute
    the model-alone prediction and its error (adapter.step), ask
    trigger.should_retrieve (offline/oracle mode -- this measures the
    value of gating, it does not simulate a real deployment) using that
    error, and either keep the model-alone prediction (no retrieval) or
    run the full retrieve + build_context + fusion path and use that
    prediction instead. The trigger's running stats are updated with the
    model-alone error on every step, retrieved or not, so the surprise
    baseline reflects the model's underlying difficulty continuously.

    fusion_alpha defaults to 1.0 (pure memory, ignoring the model
    entirely) deliberately: at alpha=1.0 the retrieved prediction never
    depends on a second adapter.step() call, so exactly one adapter.step()
    happens per eval step regardless of the gating decision -- this keeps
    retrieval-rate-100%/0% runs directly comparable to
    evaluate_with_retrieval / evaluate_no_memory on the same episodes.

    feedback_buffer (mavka.feedback.FeedbackBuffer), if given, is fed on
    every retrieved step: the retrieved ids are recorded via
    record_used() right after recall, and once the memory-augmented error
    is known it is compared to base_error (already computed for the
    gating decision, no extra adapter.step() needed) to tag the outcome
    -- helped = error < base_error. Gated-off steps (retrieve=False) never
    touch the buffer at all, since there is no retrieval to report on.
    Left None (the default), nothing changes.

    Returns the usual mean/median/p90/n_steps/errors dict, plus
    retrieval_rate: the fraction of eval steps that triggered retrieval.
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

    fusion_predictor = ConcatFusionPredictor(adapter, alpha=fusion_alpha)

    errors = []
    retrieved_flags = []

    for episode in eval_episodes:
        for step in episode:
            base = np.asarray(adapter.step(step["z"], step["action"]), dtype=np.float32)
            base_error = prediction_error(base, step["z_next"])

            retrieve = trigger.should_retrieve(base_error)
            retrieved_flags.append(retrieve)

            if retrieve:
                if scorer is not None:
                    results = pipeline.recall_scored(
                        step["z"], step["action"], k, scorer, fetch_factor=fetch_factor
                    )
                else:
                    results = pipeline.recall(step["z"], step["action"], k)

                token = None
                if feedback_buffer is not None:
                    token = feedback_buffer.record_used([id_ for id_, _ in results])

                context = build_context(results, pipeline._log, k)
                predicted = fusion_predictor.predict(step["z"], step["action"], context)
                error = prediction_error(predicted, step["z_next"])

                if feedback_buffer is not None:
                    feedback_buffer.tag_outcome(token, helped=error < base_error)
            else:
                error = base_error

            trigger.update(base_error)
            errors.append(error)

    errors_arr = np.array(errors, dtype=np.float64)

    return {
        "mean_error": float(np.mean(errors_arr)),
        "median_error": float(np.median(errors_arr)),
        "p90_error": float(np.percentile(errors_arr, 90)),
        "n_steps": len(errors),
        "errors": errors_arr.tolist(),
        "retrieval_rate": float(np.mean(retrieved_flags)),
    }
