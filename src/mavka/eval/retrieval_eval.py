import numpy as np

from mavka.core.distance import normalize
from mavka.eval.baseline import prediction_error
from mavka.retrieval.fusion import ConcatFusionPredictor, build_context
from mavka.retrieval.keying import make_keys_batch
from mavka.retrieval.trigger import SurpriseTrigger


def _train_index_if_needed(memory, memory_episodes) -> None:
    """Shared by both eval loops below: if memory's index needs training
    (e.g. a fresh IVFIndex) and hasn't been trained yet, train it on the
    same keys observe()/recall() will build -- z alone at action_scale=0,
    or make_keys_batch's concatenated+scaled key otherwise -- mirroring
    memory.action_scale exactly so training data matches what gets
    indexed.
    """
    if not hasattr(memory._index, "train") or getattr(memory._index, "is_trained", True):
        return

    zs = np.stack([step["z"] for ep in memory_episodes for step in ep])
    if memory.action_scale > 0.0:
        actions = (
            np.stack([step["action"] for ep in memory_episodes for step in ep])
            if memory.action_dim is not None
            else None
        )
        training_keys = make_keys_batch(zs, actions, memory.action_scale, memory.action_dim)
    else:
        training_keys = normalize(zs)
    memory._index.train(training_keys)


def evaluate_with_retrieval(memory_episodes, eval_episodes, memory, predictor, k: int) -> dict:
    """Fill memory from memory_episodes, then for each held-out eval step,
    retrieve the k nearest past experiences via memory.recall (which
    internally over-fetches/expands/scores exactly as memory was
    configured to at construction, or returns plain index order if memory
    has no scorer configured), assemble them into a fixed-shape context
    (build_context), and ask predictor.predict(z, action, context) for a
    prediction of z_next.

    Unlike the old Pipeline-based version, action_scale, scorer, graph,
    expansion_depth and fetch_factor are not parameters here -- they are
    all already fixed on memory itself, at construction.
    """
    _train_index_if_needed(memory, memory_episodes)

    for episode in memory_episodes:
        for step in episode:
            memory.observe(
                z=step["z"],
                action=step["action"],
                z_next=step["z_next"],
                pred_err=step["pred_err"],
                episode_id=step["episode_id"],
            )

    errors = []
    for episode in eval_episodes:
        for step in episode:
            results = memory.recall(step["z"], action=step["action"], k=k)
            context = build_context(results, memory._log, k)
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


def evaluate_gated(
    memory_episodes,
    eval_episodes,
    memory,
    adapter,
    trigger: SurpriseTrigger,
    k: int,
    fusion_alpha: float = 1.0,
    feedback_buffer=None,
) -> dict:
    """Fill memory, then for each held-out eval step: compute the
    model-alone prediction and its error (adapter.step), ask
    trigger.should_retrieve (offline/oracle mode -- this measures the
    value of gating, it does not simulate a real deployment) using that
    error, and either keep the model-alone prediction (no retrieval) or
    run the full retrieve + build_context + fusion path and use that
    prediction instead. The trigger's running stats are updated with the
    model-alone error on every step, retrieved or not, so the surprise
    baseline reflects the model's underlying difficulty continuously.

    Unlike the old Pipeline-based version, action_scale, scorer, and
    fetch_factor are not parameters here -- they are already fixed on
    memory itself, at construction.

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
    _train_index_if_needed(memory, memory_episodes)

    for episode in memory_episodes:
        for step in episode:
            memory.observe(
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
                results = memory.recall(step["z"], action=step["action"], k=k)

                token = None
                if feedback_buffer is not None:
                    token = feedback_buffer.record_used([id_ for id_, _ in results])

                context = build_context(results, memory._log, k)
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
