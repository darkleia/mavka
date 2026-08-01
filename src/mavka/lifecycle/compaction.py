import numpy as np

from mavka.graph.adjacency import EDGE_ANALOGOUS, EDGE_TEMPORAL, AdjacencyStore
from mavka.storage.log import AppendLog
from mavka.core.record import FLAG_DELETED


class _UnionFind:
    """Minimal disjoint-set structure for grouping near-duplicate ids.
    Deterministic: a union always attaches the larger root under the
    smaller one, so the same sequence of unions always produces the same
    groups regardless of the order records happen to be processed in.
    """

    def __init__(self, ids):
        self._parent = {id_: id_ for id_ in ids}

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if ra < rb:
            self._parent[rb] = ra
        else:
            self._parent[ra] = rb


def _find_duplicate_groups(
    log, index, live_ids: list[int], similarity_threshold: float, allow_cross_episode: bool
) -> _UnionFind:
    """Union each live record with its already-close (score >=
    similarity_threshold) neighbors, found by searching the existing
    index -- the simple, documented grouping method this module uses
    instead of a full clustering pass. A neighbor is ignored if it's the
    record itself, tombstoned, or (unless allow_cross_episode) from a
    different episode.
    """
    uf = _UnionFind(live_ids)
    live_id_set = set(live_ids)
    k = min(len(live_ids), 10)

    for id_ in live_ids:
        record = log.get(id_)
        for match_id, score in index.search(record.z, k=k):
            if match_id == id_ or match_id not in live_id_set:
                continue
            if score < similarity_threshold:
                continue
            if not allow_cross_episode and log.get(match_id).episode_id != record.episode_id:
                continue
            uf.union(id_, match_id)

    return uf


def _choose_representatives(log, uf: _UnionFind, live_ids: list[int]) -> dict[int, int]:
    """For each duplicate group (a union-find component), the member with
    the highest pred_err -- the most "surprising"/informative record --
    is the representative that survives; ties broken by lowest original
    id (first inserted). Returns {old_id: representative_old_id} for
    every live id; a singleton group maps a record to itself.
    """
    groups: dict[int, list[int]] = {}
    for id_ in live_ids:
        groups.setdefault(uf.find(id_), []).append(id_)

    representative_of_root = {
        root: max(members, key=lambda m: (log.get(m).pred_err, -m))
        for root, members in groups.items()
    }

    return {id_: representative_of_root[uf.find(id_)] for id_ in live_ids}


def compact(
    log,
    index,
    index_factory,
    graph=None,
    similarity_threshold: float = 0.98,
    merge: bool = True,
    allow_cross_episode: bool = False,
) -> dict:
    """Produce a compacted copy of log (and, if given, index and graph),
    dropping tombstoned records and (if merge) collapsing near-duplicate
    groups into a single representative each.

    This never mutates log, index, or graph -- it only reads from them
    and returns brand-new structures. That is the concurrency story this
    module provides: build the compacted state as a snapshot, then have
    the caller swap their references over to the result once it's ready
    (e.g. pipeline._log = result["log"]), rather than compaction ever
    touching a structure a concurrent reader might be using. A single
    Python attribute reassignment is atomic under the GIL, which is as
    far as this module goes -- full lock-free concurrency (e.g. readers
    safely mid-traversal across the swap) is out of scope.

    Method, step by step:
    1. Find near-duplicate groups (see _find_duplicate_groups): for each
       live record, search `index` (the existing, already-populated
       index) for close neighbors, and union them via a disjoint-set
       structure. Only live (non-tombstoned) matches count; by default
       only matches from the *same episode* count too (allow_cross_episode
       is an explicit, off-by-default opt-in -- two near-identical
       situations in different episodes still have different pasts and
       futures and are not merged by default). merge=False skips this
       step entirely: every live record is its own singleton group, and
       the pass only drops tombstones -- a conservative, always-lossless
       pass over live data.
    2. Pick a representative per group (see _choose_representatives): the
       highest-pred_err member survives, ties broken by lowest id.
    3. Tombstoned records are dropped outright: never carried into the
       rebuilt log at all.
    4. Rebuild the log: surviving representatives are re-appended, in
       their original id order, into a fresh AppendLog. Since AppendLog
       assigns seq_no per-episode purely by insertion order, a
       merged-away or tombstoned record that was mid-episode simply
       leaves no gap -- seq_no is implicitly renumbered to have none, and
       next_in_episode/prev_in_episode stay coherent for whatever
       survives. This is the documented episode policy: representatives
       inherit their relative position among that episode's survivors.
    5. Rebuild the index (via index_factory, a zero-arg callable
       returning a fresh, appropriately configured index -- e.g. an
       untrained IVFIndex or a plain VectorStore) over the new log, id
       for id, training first if the index needs it.
    6. Rebuild graph, if given: every edge from every surviving-or-merged
       old id is re-added with both endpoints passed through id_map,
       dropping any edge whose destination was tombstoned. Edges from a
       merged-away id land on its representative's new id; edges to a
       merged-away destination land there too -- both naturally use
       AdjacencyStore's own no-duplicate and replace-weakest policies
       (including rejecting a same-record edge that a merge happened to
       turn into a self-loop), so no separate dedup/fixed-degree logic is
       needed here.

    Returns {"log": AppendLog, "index": <same type index_factory makes>,
    "graph": AdjacencyStore | None, "id_map": {old_id: new_id | None},
    "stats": {old_count, new_count, tombstones_dropped, records_merged}}.
    id_map is None for a dropped (tombstoned) id, otherwise the record's
    new id (its own, if it survived standalone, or its representative's,
    if it was merged away).
    """
    dim = log._dim
    action_dim = log._action_dim

    live_ids = [record.id for record in log.scan() if not (record.flags & FLAG_DELETED)]

    if merge and live_ids:
        uf = _find_duplicate_groups(log, index, live_ids, similarity_threshold, allow_cross_episode)
        representative_of = _choose_representatives(log, uf, live_ids)
    else:
        representative_of = {id_: id_ for id_ in live_ids}

    # old_id -> old representative id, or None (missing = tombstoned).
    old_id_to_old_representative = dict(representative_of)

    surviving_old_ids = sorted(set(representative_of.values()))

    new_log = AppendLog(dim=dim, action_dim=action_dim)
    new_id_of_old_representative = {}
    for old_id in surviving_old_ids:
        record = log.get(old_id)
        new_id = new_log.append(
            z=record.z,
            action=record.action,
            pred_err=record.pred_err,
            episode_id=record.episode_id,
            timestamp_ns=record.timestamp_ns,
        )
        new_id_of_old_representative[old_id] = new_id

    id_map: dict[int, int | None] = {}
    for old_id in range(log.count):
        old_representative = old_id_to_old_representative.get(old_id)
        id_map[old_id] = (
            None if old_representative is None else new_id_of_old_representative[old_representative]
        )

    new_index = index_factory()
    if new_log.count:
        zs = np.stack([new_log.get(i).z for i in range(new_log.count)])
        if hasattr(new_index, "train") and not getattr(new_index, "is_trained", True):
            new_index.train(zs)
        for z in zs:
            new_index.add(z)

    new_graph = None
    if graph is not None:
        new_graph = AdjacencyStore(degree=graph.degree)
        for _ in range(new_log.count):
            new_graph.add_node()

        for old_id in range(log.count):
            new_src = id_map[old_id]
            if new_src is None:
                continue
            for edge_type in (EDGE_TEMPORAL, EDGE_ANALOGOUS):
                for dst, weight in graph.neighbors_of_type(old_id, edge_type):
                    new_dst = id_map[dst]
                    if new_dst is None:
                        continue
                    new_graph.add_edge(new_src, new_dst, weight=weight, edge_type=edge_type)

    return {
        "log": new_log,
        "index": new_index,
        "graph": new_graph,
        "id_map": id_map,
        "stats": {
            "old_count": log.count,
            "new_count": new_log.count,
            "tombstones_dropped": sum(1 for v in id_map.values() if v is None),
            "records_merged": len(live_ids) - len(surviving_old_ids),
        },
    }
