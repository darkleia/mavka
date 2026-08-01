import time
import warnings

import numpy as np

from mavka.compaction import compact
from mavka.record import FLAG_DELETED, FLAG_PINNED


def _min_max_normalize(values: np.ndarray) -> np.ndarray:
    lo = values.min()
    hi = values.max()
    if hi == lo:
        return np.full_like(values, 0.5, dtype=np.float64)
    return (values - lo) / (hi - lo)


def mark_pinned(log, id: int) -> None:
    """Manually pin a record so it is permanently exempt from eviction,
    using the pinned bit already defined on the record layout
    (record.FLAG_PINNED) -- a thin, documented entry point over
    AppendLog.pin().
    """
    log.pin(id)


class EvictionPolicy:
    """Decay + utility + pred_err blended keep-score, plus pinning, used to
    evict the least-valuable records once a store exceeds capacity.

    keep_score = (w_decay * decay + w_utility * utility + w_pred_err * pred_err) / (w_decay + w_utility + w_pred_err)

    Each of the three signals is independently min-max normalized across
    the current candidate set before blending (same rationale as
    FixedWeightScorer: raw scales differ -- decay is in [0, 1], utility is
    an unbounded count, pred_err is an unbounded positive float -- so none
    of them should dominate purely from scale).

    - decay: exp(-age_seconds / decay_constant_s), where age is measured
      from the record's last "useful" retrieval (see
      record_retrieval_feedback), or from its own creation timestamp if
      it has never had positive feedback. 1.0 = just used/created, ->0 as
      it goes unused for longer than decay_constant_s.
    - utility: a running count of how many times record_retrieval_feedback
      was called with helped=True for this id. Deliberately not a
      frequency/access count -- see record_retrieval_feedback's docstring
      for why frequency alone would be exactly the wrong signal here.
    - pred_err: the record's own stored prediction error -- how
      surprising/informative it was when observed.

    Pinning is separate from and takes precedence over keep_score
    entirely: a pinned record (record.flags & FLAG_PINNED) is never a
    candidate for eviction, full stop, even if that means the store stays
    over capacity (evict_to_capacity documents and warns about this, it
    never violates it). pin_threshold, if set, causes evict_to_capacity to
    automatically pin any not-yet-pinned live record whose pred_err
    exceeds it, before deciding what to evict.
    """

    def __init__(
        self,
        w_decay: float = 1.0,
        w_utility: float = 1.0,
        w_pred_err: float = 1.0,
        decay_constant_s: float = 86400.0,
        pin_threshold: float | None = None,
    ):
        self.w_decay = w_decay
        self.w_utility = w_utility
        self.w_pred_err = w_pred_err
        self.decay_constant_s = decay_constant_s
        self.pin_threshold = pin_threshold

        self._utility: dict[int, float] = {}
        self._last_useful_ns: dict[int, int] = {}

    def record_retrieval_feedback(self, id: int, helped: bool, now_ns: int | None = None) -> None:
        """Call this after a retrieval to report whether a specific
        retrieved record helped -- e.g. it was part of the candidate set
        that produced a lower-error prediction, or simply "was retrieved
        this step" if the caller doesn't want to attempt a finer-grained
        signal. This is the lightweight hook an eval/retrieval path calls;
        this module does not itself decide what "helped" means.

        Only positive feedback has an effect: it increments the record's
        utility count and refreshes its "last useful" timestamp (the
        reference point decay is measured from). This is deliberately not
        a plain access/frequency counter: a record retrieved constantly
        but never actually useful gets no boost from being merely looked
        at, while a rare record that helped even once keeps a live
        utility signal and a refreshed decay clock -- exactly the
        protection a frequency-based policy would fail to provide.
        """
        if not helped:
            return
        if now_ns is None:
            now_ns = time.time_ns()
        self._utility[id] = self._utility.get(id, 0.0) + 1.0
        self._last_useful_ns[id] = now_ns

    def _decay_factor(self, record, now_ns: int) -> float:
        reference_ns = self._last_useful_ns.get(record.id, record.timestamp_ns)
        age_s = max(0.0, (now_ns - reference_ns) / 1e9)
        return float(np.exp(-age_s / self.decay_constant_s))

    def compute_keep_scores(self, records: list, now_ns: int | None = None) -> dict[int, float]:
        """Blended, normalized keep_score for each of the given records
        (typically the current non-pinned live records). Higher = more
        worth keeping. Scores are only comparable within one call's batch
        -- normalization is relative to whatever set is passed in.
        """
        if not records:
            return {}
        if now_ns is None:
            now_ns = time.time_ns()

        decay = np.array([self._decay_factor(r, now_ns) for r in records], dtype=np.float64)
        utility = np.array([self._utility.get(r.id, 0.0) for r in records], dtype=np.float64)
        pred_err = np.array([r.pred_err for r in records], dtype=np.float64)

        decay_n = _min_max_normalize(decay)
        utility_n = _min_max_normalize(utility)
        pred_err_n = _min_max_normalize(pred_err)

        total_weight = self.w_decay + self.w_utility + self.w_pred_err
        combined = (
            self.w_decay * decay_n + self.w_utility * utility_n + self.w_pred_err * pred_err_n
        ) / total_weight

        return {record.id: float(score) for record, score in zip(records, combined)}

    def evict_to_capacity(
        self, log, index, index_factory, capacity: int, graph=None, now_ns: int | None = None
    ) -> dict:
        """If log has more than capacity live records, tombstone the
        lowest-keep_score non-pinned ones until at or under capacity, then
        rebuild log/index/graph via compaction's merge=False (tombstone-
        drop-only) pass, which reuses all of compaction's reference-fixing
        machinery -- evicting a record cleans its index entry and graph
        edges exactly like a compaction merge-away does.

        Runs off the hot path on the same snapshot/atomic-swap discipline
        as compact(): the expensive structural rebuild (index and graph)
        never mutates the originals, only tombstone() (a single flag flip
        on one record slot, an already-established, cheap AppendLog
        operation) touches the live log directly, before the rebuild.

        Before deciding what to evict, if pin_threshold is set, any live,
        not-yet-pinned record whose pred_err exceeds it is pinned
        automatically. Pinned records are never eviction candidates: if
        there aren't enough non-pinned records to reach capacity, this
        warns (via the warnings module) and evicts every non-pinned
        record it can rather than ever touching a pinned one -- pins take
        precedence over the cap, always.

        Returns compact()'s own result dict, plus "evicted_ids" (the list
        of ids tombstoned this call) and stats["pinned_over_capacity"]
        (how many records over capacity remain because pinning protected
        them).
        """
        if now_ns is None:
            now_ns = time.time_ns()

        live_ids = [record.id for record in log.scan() if not (record.flags & FLAG_DELETED)]

        if len(live_ids) <= capacity:
            return {
                "log": log,
                "index": index,
                "graph": graph,
                "id_map": {id_: id_ for id_ in range(log.count)},
                "evicted_ids": [],
                "stats": {
                    "old_count": log.count,
                    "new_count": log.count,
                    "tombstones_dropped": 0,
                    "records_merged": 0,
                    "pinned_over_capacity": 0,
                },
            }

        if self.pin_threshold is not None:
            for id_ in live_ids:
                record = log.get(id_)
                if record.pred_err > self.pin_threshold and not (record.flags & FLAG_PINNED):
                    log.pin(id_)

        live_records = [log.get(id_) for id_ in live_ids]
        candidates = [record for record in live_records if not (record.flags & FLAG_PINNED)]

        n_to_evict = len(live_ids) - capacity
        pinned_over_capacity = 0
        if n_to_evict > len(candidates):
            pinned_over_capacity = n_to_evict - len(candidates)
            warnings.warn(
                f"Pinning prevents reaching capacity {capacity}: only {len(candidates)} "
                f"non-pinned record(s) available to evict but {n_to_evict} would be needed; "
                f"store will remain {pinned_over_capacity} record(s) over capacity.",
                stacklevel=2,
            )
            n_to_evict = len(candidates)

        scores = self.compute_keep_scores(candidates, now_ns)
        victims = sorted(candidates, key=lambda r: (scores[r.id], r.id))[:n_to_evict]
        evicted_ids = [record.id for record in victims]

        for id_ in evicted_ids:
            log.tombstone(id_)

        result = compact(log, index, index_factory, graph=graph, merge=False)
        result["evicted_ids"] = evicted_ids
        result["stats"]["pinned_over_capacity"] = pinned_over_capacity
        return result
