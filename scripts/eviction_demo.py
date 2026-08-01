import numpy as np

from mavka.lifecycle.eviction import EvictionPolicy
from mavka.storage.log import AppendLog
from mavka.index.flat import FlatIndex

NOW_NS = 1_000_000_000_000


def _rand(dim, seed):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def main() -> None:
    dim = 16
    capacity = 20
    log = AppendLog(dim=dim)

    # A pile of "boring" records: low pred_err, never usefully retrieved.
    boring_ids = [
        log.append(z=_rand(dim, i), episode_id=0, pred_err=0.02, timestamp_ns=NOW_NS)
        for i in range(29)
    ]

    # One rare, highly surprising record -- the model was very wrong here
    # -- that is never retrieved at all (so it earns no utility boost).
    rare_id = log.append(z=_rand(dim, 999), episode_id=0, pred_err=8.0, timestamp_ns=NOW_NS)

    all_ids = [*boring_ids, rare_id]
    index = FlatIndex(dim=dim)
    for id_ in all_ids:
        index.add(log.get(id_).z)

    print(f"before eviction: {log.count} records (capacity {capacity})")
    print(f"  rare record id={rare_id}, pred_err={log.get(rare_id).pred_err}")

    # pin_threshold ensures the rare record is auto-pinned before eviction
    # decides what to drop, even though it has zero retrieval utility.
    policy = EvictionPolicy(pin_threshold=1.0)
    result = policy.evict_to_capacity(
        log, index, index_factory=lambda: FlatIndex(dim=dim), capacity=capacity, now_ns=NOW_NS
    )

    new_log = result["log"]
    rare_new_id = result["id_map"][rare_id]

    print(f"\nafter eviction:  {new_log.count} records")
    print(f"  evicted: {len(result['evicted_ids'])} boring record(s)")
    print(f"  rare record survived: {rare_new_id is not None}")
    if rare_new_id is not None:
        print(f"  rare record's pred_err intact: {new_log.get(rare_new_id).pred_err}")

    boring_survivors = sum(1 for old_id in boring_ids if result["id_map"][old_id] is not None)
    print(f"  boring survivors: {boring_survivors} of {len(boring_ids)}")


if __name__ == "__main__":
    main()
