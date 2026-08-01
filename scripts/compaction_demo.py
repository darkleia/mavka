import numpy as np

from mavka.lifecycle.compaction import compact
from mavka.storage.log import AppendLog
from mavka.index.flat import FlatIndex


def _rand(dim, seed):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def main() -> None:
    dim = 16
    log = AppendLog(dim=dim)

    # A block of genuinely distinct records...
    distinct_ids = [log.append(z=_rand(dim, i), episode_id=0, pred_err=0.1 * i) for i in range(20)]

    # ...three near-duplicate clusters, each with a clear highest-pred_err
    # member that should survive as the representative...
    duplicate_ids = []
    for cluster in range(3):
        base_z = _rand(dim, 1000 + cluster)
        for i in range(4):
            pred_err = 1.0 if i == 2 else 0.1  # member index 2 is the "surprising" one
            duplicate_ids.append(
                log.append(
                    z=base_z + _rand(dim, 2000 + cluster * 10 + i) * 0.0001,
                    episode_id=1,
                    pred_err=pred_err,
                )
            )

    # ...and some records that get tombstoned before compaction.
    tombstoned_ids = [log.append(z=_rand(dim, 3000 + i), episode_id=2) for i in range(5)]
    for id_ in tombstoned_ids:
        log.tombstone(id_)

    all_ids = distinct_ids + duplicate_ids + tombstoned_ids
    index = FlatIndex(dim=dim)
    for id_ in all_ids:
        index.add(log.get(id_).z)

    print(f"before compaction: {log.count} records, index count {index.count}")

    result = compact(
        log, index, index_factory=lambda: FlatIndex(dim=dim), merge=True, similarity_threshold=0.999
    )
    stats = result["stats"]

    print(
        f"after compaction:  {stats['new_count']} records "
        f"({stats['tombstones_dropped']} tombstones dropped, "
        f"{stats['records_merged']} merged away)"
    )

    # Confirm search still finds the survivors correctly by identity.
    new_log = result["log"]
    new_index = result["index"]
    id_map = result["id_map"]

    ok = True
    for old_id in distinct_ids:
        new_id = id_map[old_id]
        hits = new_index.search(new_log.get(new_id).z, k=1)
        if hits[0][0] != new_id:
            ok = False
    print(f"search self-identity preserved for all survivors: {ok}")


if __name__ == "__main__":
    main()
