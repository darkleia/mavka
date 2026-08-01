import time
import warnings

import numpy as np

from mavka.lifecycle.eviction import EvictionPolicy
from mavka.index.ivf import IVFIndex
from mavka.storage.log import AppendLog
from mavka.core.record import FLAG_PINNED
from mavka.index.flat import FlatIndex
from mavka.core.distance import normalize


class TieredStore:
    """Two-tier storage: a small, bounded hot tier with exact search
    (FlatIndex) and a larger cold tier with approximate search
    (IVFIndex). Every record lives in exactly one tier at a time; a
    background pass (migrate_hot_to_cold) moves the least valuable hot
    records to cold once the hot tier exceeds hot_capacity, using the
    same keep-score EvictionPolicy already uses (decay + utility +
    pred_err) -- this reuses that scoring, not eviction's tombstone/
    rebuild machinery, since nothing here is ever deleted, only moved.

    The single AppendLog underneath is the one and only source of truth
    for id -> record; it is never rebuilt and ids are never reassigned,
    unlike compact()/evict_to_capacity(). Moving a record between tiers
    only touches this store's own tier bookkeeping (which index holds its
    vector, and at what internal position within that index) -- the log,
    and anything that references a record by id (a graph, episode
    navigation), needs no remapping at all, because the id never changes.

    Internally, each tier's index (FlatIndex/IVFIndex) hands back its
    own internal, index-local integer when a vector is added -- unrelated
    to the log id. This store keeps a position -> log_id map per tier to
    translate search hits back to log ids (unlike Pipeline, which gets
    away with no such map because its single index's positions stay in
    lockstep with log ids by construction; that trick breaks here since
    hot needs actual removal on migration, and lockstep can't survive
    that).
    """

    def __init__(
        self,
        dim: int,
        action_dim: int | None = None,
        hot_capacity: int = 200,
        eviction_policy: EvictionPolicy | None = None,
        cold_n_lists: int = 100,
        cold_seed: int = 0,
    ):
        self.dim = dim
        self.action_dim = action_dim
        self.hot_capacity = hot_capacity
        self.policy = eviction_policy if eviction_policy is not None else EvictionPolicy()

        self._log = AppendLog(dim=dim, action_dim=action_dim)

        self._hot_index = FlatIndex(dim=dim)
        self._cold_index = IVFIndex(dim=dim, n_lists=cold_n_lists, seed=cold_seed)

        self._log_id_of_hot_pos: dict[int, int] = {}
        self._log_id_of_cold_pos: dict[int, int] = {}

        self._tier_of: dict[int, str] = {}
        self._hot_ids: set[int] = set()
        self._cold_ids: set[int] = set()

    @property
    def count(self) -> int:
        return self._log.count

    @property
    def hot_count(self) -> int:
        return len(self._hot_ids)

    @property
    def cold_count(self) -> int:
        return len(self._cold_ids)

    def tier_of(self, id: int) -> str:
        if id not in self._tier_of:
            raise ValueError(f"unknown id {id!r}")
        return self._tier_of[id]

    def get(self, id: int):
        return self._log.get(id)

    def observe(self, z, action=None, z_next=None, pred_err: float = 0.0, episode_id: int = 0) -> int:
        """New records always enter the hot tier -- they are, by
        definition, the most recent. z_next is accepted only for calling
        convenience against the same step dicts generate_trajectory
        produces (mirrors Pipeline.observe); it is not stored.
        """
        z = normalize(np.asarray(z, dtype=np.float32))
        log_id = self._log.append(z=z, action=action, pred_err=pred_err, episode_id=episode_id)

        pos = self._hot_index.add(z)
        self._log_id_of_hot_pos[pos] = log_id
        self._tier_of[log_id] = "hot"
        self._hot_ids.add(log_id)

        return log_id

    def migrate_hot_to_cold(self, now_ns: int | None = None, capacity: int | None = None) -> dict:
        """Background pass: if the hot tier exceeds capacity (hot_capacity,
        or the override given here), move the lowest-keep-score non-pinned
        hot records to cold until hot is back at or under capacity.
        Pinned hot records are never migrated; if there aren't enough
        non-pinned records to reach capacity, this warns and migrates
        every non-pinned record it can, exactly like
        EvictionPolicy.evict_to_capacity does for its own over-pinned
        case.

        Builds the new hot tier as a fresh FlatIndex snapshot (excluding
        the migrated ids) and swaps it in atomically at the end --
        FlatIndex has no removal primitive, so shrinking hot means
        rebuilding it, on the same snapshot/atomic-swap discipline
        compact() and evict_to_capacity() use. The cold tier only ever
        grows (append, no rebuild needed) since IVFIndex requires no
        removal here.

        Returns {"migrated_ids": [...], "hot_count": int, "cold_count":
        int, "stats": {"pinned_over_target": int}}. No id_map: ids never
        change, by design.
        """
        if now_ns is None:
            now_ns = time.time_ns()
        target_capacity = self.hot_capacity if capacity is None else capacity

        hot_ids = sorted(self._hot_ids)
        if len(hot_ids) <= target_capacity:
            return {
                "migrated_ids": [],
                "hot_count": len(self._hot_ids),
                "cold_count": len(self._cold_ids),
                "stats": {"pinned_over_target": 0},
            }

        hot_records = [self._log.get(id_) for id_ in hot_ids]
        candidates = [record for record in hot_records if not (record.flags & FLAG_PINNED)]

        n_to_migrate = len(hot_ids) - target_capacity
        pinned_over_target = 0
        if n_to_migrate > len(candidates):
            pinned_over_target = n_to_migrate - len(candidates)
            warnings.warn(
                f"Pinning prevents reaching hot_capacity {target_capacity}: only "
                f"{len(candidates)} non-pinned hot record(s) available to migrate but "
                f"{n_to_migrate} would be needed; hot tier will remain "
                f"{pinned_over_target} record(s) over capacity.",
                stacklevel=2,
            )
            n_to_migrate = len(candidates)

        migrated_ids: list[int] = []
        if n_to_migrate > 0:
            scores = self.policy.compute_keep_scores(candidates, now_ns=now_ns)
            victims = sorted(candidates, key=lambda r: (scores[r.id], r.id))[:n_to_migrate]
            migrated_ids = [record.id for record in victims]

            migrated_set = set(migrated_ids)
            remaining_hot_ids = [id_ for id_ in hot_ids if id_ not in migrated_set]

            new_hot_index = FlatIndex(dim=self.dim)
            new_log_id_of_hot_pos: dict[int, int] = {}
            for id_ in remaining_hot_ids:
                pos = new_hot_index.add(self._log.get(id_).z)
                new_log_id_of_hot_pos[pos] = id_

            migrated_vectors = np.stack([self._log.get(id_).z for id_ in migrated_ids])
            if not self._cold_index.is_trained:
                self._cold_index.train(migrated_vectors)
            cold_positions = self._cold_index.add_batch(migrated_vectors)
            for pos, id_ in zip(cold_positions, migrated_ids):
                self._log_id_of_cold_pos[pos] = id_

            self._hot_index = new_hot_index
            self._log_id_of_hot_pos = new_log_id_of_hot_pos
            self._hot_ids = set(remaining_hot_ids)
            self._cold_ids.update(migrated_ids)
            for id_ in migrated_ids:
                self._tier_of[id_] = "cold"

        return {
            "migrated_ids": migrated_ids,
            "hot_count": len(self._hot_ids),
            "cold_count": len(self._cold_ids),
            "stats": {"pinned_over_target": pinned_over_target},
        }

    def promote_to_hot(self, id: int) -> None:
        """Optional resurrection hook: a cold record that proved useful on
        retrieval can be promoted back to hot. Adds its vector into the
        hot index and flips its tier marker; idempotent no-op if it's
        already hot. Deliberately does not remove the record's old vector
        from the cold IVFIndex (no removal primitive exists) -- recall()
        filters cold search hits by current tier_of, so the stale cold
        copy is simply never surfaced again once promoted.
        """
        if id not in self._tier_of:
            raise ValueError(f"unknown id {id!r}")
        if self._tier_of[id] == "hot":
            return

        z = self._log.get(id).z
        pos = self._hot_index.add(z)
        self._log_id_of_hot_pos[pos] = id

        self._tier_of[id] = "hot"
        self._hot_ids.add(id)
        self._cold_ids.discard(id)

    def recall(self, query_z, k: int) -> list[tuple[int, float]]:
        """Search both tiers and merge into one top-k by score. Both
        FlatIndex and IVFIndex score with normalized-vector dot product
        (cosine similarity), so scores from the two tiers are directly
        comparable without any extra normalization step -- gather
        candidates from each, then take the merged top-k, same pattern as
        every other recall path in this codebase.
        """
        query_z = normalize(np.asarray(query_z, dtype=np.float32))

        hot_hits = self._hot_index.search(query_z, k)
        hot_results = [(self._log_id_of_hot_pos[pos], score) for pos, score in hot_hits]

        cold_results = []
        if self._cold_index.count:
            for pos, score in self._cold_index.search(query_z, k):
                id_ = self._log_id_of_cold_pos[pos]
                if self._tier_of.get(id_) == "cold":
                    cold_results.append((id_, score))

        merged = hot_results + cold_results
        merged.sort(key=lambda pair: pair[1], reverse=True)
        return merged[:k]
