import numpy as np

from mavka.tiering import TieredStore

NOW_NS = 1_000_000_000_000


def _rand(dim, seed):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def main() -> None:
    dim = 16
    hot_capacity = 20
    store = TieredStore(dim=dim, hot_capacity=hot_capacity, cold_n_lists=8)

    # 50 "old" records with mixed pred_err, all inserted first -- these are
    # the aging population migration should mostly clear out of hot.
    old_ids = [
        store.observe(z=_rand(dim, i), pred_err=float(i % 5) * 0.05, episode_id=0)
        for i in range(50)
    ]

    # One standout, highly-surprising record buried among the old ones --
    # high pred_err should keep it competitive for staying hot.
    standout_id = store.observe(z=_rand(dim, 999), pred_err=5.0, episode_id=0)

    print(f"populated store: {store.count} records")
    print(f"  before migration -- hot: {store.hot_count}, cold: {store.cold_count}")

    result = store.migrate_hot_to_cold(now_ns=NOW_NS)

    print(f"\nmigrated {len(result['migrated_ids'])} record(s) hot -> cold")
    print(f"  after migration  -- hot: {store.hot_count}, cold: {store.cold_count}")
    print(f"  standout record (pred_err=5.0) tier: {store.tier_of(standout_id)}")

    # Sample query: search for whatever is close to one of the old,
    # now-cold records -- two-tier recall must still find it transparently.
    sample_target = old_ids[0]
    query = store.get(sample_target).z
    hits = store.recall(query, k=5)

    print(f"\nsample query -- nearest to id={sample_target} (currently {store.tier_of(sample_target)}):")
    for id_, score in hits:
        print(f"  id={id_:3d} tier={store.tier_of(id_):4s} score={score:.4f}")

    assert hits[0][0] == sample_target, "two-tier search failed to find the record itself"
    print("\ntwo-tier search found the record itself as the top hit -- OK")


if __name__ == "__main__":
    main()
