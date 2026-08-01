import numpy as np


def _min_max_normalize(values: np.ndarray) -> np.ndarray:
    lo = values.min()
    hi = values.max()
    if hi == lo:
        return np.full_like(values, 0.5)
    return (values - lo) / (hi - lo)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class FixedWeightScorer:
    """Re-ranks a candidate set returned by the index using a hand-set
    weighted blend of three signals. Each signal is independently min-max
    normalized to [0, 1] across the current candidate set before weighting,
    so no signal dominates purely from its raw numeric scale and the
    weights are directly comparable:

    - s_sim: the index's own similarity score for the candidate (the raw
      key dot product it was retrieved with).
    - s_action: cosine similarity between the query action and the
      candidate's stored action, computed on the action alone (separate
      from the combined key). Skipped, with its weight redistributed over
      the remaining signals, if the query action or any candidate's stored
      action is missing.
    - s_recency: the candidate's id. Ids are assigned in strictly
      increasing insertion order, so a higher id is a more recent
      experience -- this is a simple, always-available recency proxy
      (no separate timestamp handling needed). 1.0 = most recent candidate
      in this set, 0.0 = least recent, after min-max normalization.

    If a signal is constant across the whole candidate set (no
    discriminating information), it normalizes to a neutral 0.5 for every
    candidate rather than dividing by zero.
    """

    def __init__(self, log, w_sim: float = 1.0, w_action: float = 0.5, w_recency: float = 0.2):
        self._log = log
        self.w_sim = w_sim
        self.w_action = w_action
        self.w_recency = w_recency

    def score(self, candidates: list[tuple[int, float]], query_action) -> list[tuple[int, float]]:
        if not candidates:
            return []
        if len(candidates) == 1:
            return list(candidates)

        ids = [id_ for id_, _ in candidates]
        sim_raw = np.array([s for _, s in candidates], dtype=np.float64)
        records = [self._log.get(id_) for id_ in ids]

        weights = {"sim": self.w_sim}
        signals = {"sim": _min_max_normalize(sim_raw)}

        query_action_arr = (
            np.asarray(query_action, dtype=np.float32) if query_action is not None else None
        )
        has_actions = query_action_arr is not None and all(r.action is not None for r in records)
        if has_actions:
            action_raw = np.array(
                [_cosine_similarity(query_action_arr, r.action) for r in records], dtype=np.float64
            )
            signals["action"] = _min_max_normalize(action_raw)
            weights["action"] = self.w_action

        recency_raw = np.array([r.id for r in records], dtype=np.float64)
        signals["recency"] = _min_max_normalize(recency_raw)
        weights["recency"] = self.w_recency

        total_weight = sum(weights.values())
        if total_weight == 0:
            total_weight = len(weights)
            weights = dict.fromkeys(weights, 1.0)

        combined = np.zeros(len(candidates), dtype=np.float64)
        for name, w in weights.items():
            combined += (w / total_weight) * signals[name]

        order = np.argsort(combined)[::-1]
        return [(ids[i], float(combined[i])) for i in order]
