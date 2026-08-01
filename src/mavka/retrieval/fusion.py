from typing import Protocol, runtime_checkable

import numpy as np

from mavka.index.flat import normalize


def build_context(recall_results: list[tuple[int, float]], log, k: int) -> dict:
    """Assemble a fixed-shape retrieval context from scored recall results.

    For each of the first k (id, score) pairs, gathers the retrieved
    experience's z, action, outcome (next_in_episode(id).z -- what actually
    happened after it), and score. A candidate that is the last step of its
    episode has no outcome and is skipped (left masked out, zero-padded)
    rather than given a garbage value.

    Always returns exactly k rows, regardless of how many valid memories
    were found -- fewer than k valid results, or fewer than k results
    given at all, are zero-padded with mask=False. This fixed [k, ...]
    shape is the model-agnostic contract a real world model's concat or
    cross-attention input would consume.

    Returns a dict:
      mem_z:       (k, dim) float32
      mem_action:  (k, action_dim) float32 (action_dim from the log; 0
                    columns if the log has no actions)
      mem_outcome: (k, dim) float32
      mem_score:   (k,) float32
      mask:        (k,) bool -- True where the row is a real, valid memory
    """
    dim = log._dim
    action_dim = log._action_dim

    mem_z = np.zeros((k, dim), dtype=np.float32)
    mem_action = np.zeros((k, action_dim if action_dim is not None else 0), dtype=np.float32)
    mem_outcome = np.zeros((k, dim), dtype=np.float32)
    mem_score = np.zeros(k, dtype=np.float32)
    mask = np.zeros(k, dtype=bool)

    for i, (id_, score) in enumerate(recall_results[:k]):
        next_record = log.next_in_episode(id_)
        if next_record is None:
            continue

        record = log.get(id_)
        mem_z[i] = record.z
        if action_dim is not None and record.action is not None:
            mem_action[i] = record.action
        mem_outcome[i] = next_record.z
        mem_score[i] = score
        mask[i] = True

    return {
        "mem_z": mem_z,
        "mem_action": mem_action,
        "mem_outcome": mem_outcome,
        "mem_score": mem_score,
        "mask": mask,
    }


@runtime_checkable
class FusionPredictor(Protocol):
    """Any predictor -- synthetic now, a real world model later -- that
    turns (current z, action, retrieved context) into a predicted z_next.
    A real model implements this by consuming context inside its own
    forward pass (concatenating mem_z/mem_action/mem_outcome as extra
    input tokens, or attending over them via cross-attention); this
    interface does not change either way, only what happens inside does.
    """

    def predict(self, z, action, context: dict) -> np.ndarray: ...


def _score_weighted_mean(context: dict) -> np.ndarray | None:
    mask = context["mask"]
    if not np.any(mask):
        return None

    scores = context["mem_score"][mask].astype(np.float64)
    outcomes = context["mem_outcome"][mask]

    # Shift to non-negative before normalizing to weights that sum to 1 --
    # raw scores can be negative (e.g. a plain key dot product), and this
    # keeps the combination well-defined regardless of sign or scale. When
    # all valid scores are equal, every weight collapses to the same value,
    # i.e. this degrades gracefully to a plain mean.
    weights = scores - scores.min() + 1e-8
    weights = weights / weights.sum()

    return np.sum(outcomes * weights[:, np.newaxis], axis=0).astype(np.float32)


class ConcatFusionPredictor:
    """Synthetic fusion predictor: blends the model's own prediction with a
    score-weighted, mask-aware mean of the retrieved outcomes.

    z_next_hat = normalize((1 - alpha) * base + alpha * mem_prediction)
    where base = adapter.step(z, action).

    alpha = 0 ignores memory entirely (pure model, matches the no-memory
    baseline exactly). alpha = 1 ignores the model entirely (pure
    score-weighted memory average). Falls back to the model's own
    prediction if no retrieved candidate has a valid outcome at all (e.g.
    empty memory).

    This hand-blend is a stand-in: a real world model would instead
    consume `context` directly inside its own forward pass rather than
    mixing two separate outputs after the fact -- but it would still
    implement the exact same predict(z, action, context) interface.
    """

    def __init__(self, adapter, alpha: float = 0.5):
        self.adapter = adapter
        self.alpha = alpha

    def predict(self, z, action, context: dict) -> np.ndarray:
        if self.alpha == 0.0:
            return np.asarray(self.adapter.step(z, action), dtype=np.float32)

        mem_prediction = _score_weighted_mean(context)

        if self.alpha == 1.0:
            if mem_prediction is None:
                return np.asarray(self.adapter.step(z, action), dtype=np.float32)
            return normalize(mem_prediction)

        base = np.asarray(self.adapter.step(z, action), dtype=np.float32)
        if mem_prediction is None:
            return base

        blended = (1.0 - self.alpha) * base + self.alpha * mem_prediction
        return normalize(blended.astype(np.float32))
