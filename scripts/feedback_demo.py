import numpy as np

from mavka.eviction import EvictionPolicy
from mavka.feedback import FeedbackBuffer, drain_feedback
from mavka.log import AppendLog
from mavka.store import VectorStore

NOW_NS = 1_000_000_000_000
SECOND_NS = 10**9


def _rand(dim, seed):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def main() -> None:
    dim = 8
    capacity = 3
    log = AppendLog(dim=dim)
    old_ts = NOW_NS - 1000 * SECOND_NS

    # Two groups with identical pred_err/age -- indistinguishable by
    # keep_score until retrieval feedback comes in. helpful_ids get
    # retrieved rarely but always help; unhelpful_ids get retrieved often
    # but never help (mirrors the frequency-vs-utility distinction
    # EvictionPolicy.record_retrieval_feedback documents).
    helpful_ids = [
        log.append(z=_rand(dim, i), episode_id=0, pred_err=0.1, timestamp_ns=old_ts)
        for i in range(3)
    ]
    unhelpful_ids = [
        log.append(z=_rand(dim, 10 + i), episode_id=0, pred_err=0.1, timestamp_ns=old_ts)
        for i in range(5)
    ]
    all_ids = [*helpful_ids, *unhelpful_ids]

    index = VectorStore(dim=dim)
    for id_ in all_ids:
        index.add(log.get(id_).z)

    policy = EvictionPolicy()

    records = [log.get(id_) for id_ in all_ids]
    scores_before = policy.compute_keep_scores(records, now_ns=NOW_NS)
    print("keep-scores BEFORE draining feedback:")
    for id_ in all_ids:
        tag = "helpful" if id_ in helpful_ids else "unhelpful"
        print(f"  id={id_} ({tag:9s}) score={scores_before[id_]:.4f}")

    # Hot path: retrieval happens, ids get buffered as used (cheap append).
    # Outcome tagging: each step's verdict lands once known.
    buffer = FeedbackBuffer()
    for id_ in helpful_ids:
        token = buffer.record_used([id_])
        buffer.tag_outcome(token, helped=True)
    for id_ in unhelpful_ids:
        for _ in range(4):  # retrieved often
            token = buffer.record_used([id_])
            buffer.tag_outcome(token, helped=False)

    # Background drain: the only place record_retrieval_feedback is called.
    n_applied = drain_feedback(buffer, policy, now_ns=NOW_NS - 10 * SECOND_NS)
    print(f"\ndrained {n_applied} feedback events into the eviction policy")

    scores_after = policy.compute_keep_scores(records, now_ns=NOW_NS)
    print("\nkeep-scores AFTER draining feedback:")
    for id_ in all_ids:
        tag = "helpful" if id_ in helpful_ids else "unhelpful"
        print(f"  id={id_} ({tag:9s}) score={scores_after[id_]:.4f}")

    result = policy.evict_to_capacity(
        log, index, index_factory=lambda: VectorStore(dim=dim), capacity=capacity, now_ns=NOW_NS
    )
    evicted = set(result["evicted_ids"])
    print(f"\nevicted down to capacity {capacity}: {sorted(evicted)}")
    print(f"  helpful survivors:   {sum(1 for id_ in helpful_ids if id_ not in evicted)}/{len(helpful_ids)}")
    print(f"  unhelpful survivors: {sum(1 for id_ in unhelpful_ids if id_ not in evicted)}/{len(unhelpful_ids)}")


if __name__ == "__main__":
    main()
