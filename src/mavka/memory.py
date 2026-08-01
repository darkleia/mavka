import functools

import numpy as np

from mavka.core.distance import normalize
from mavka.core.record import Experience
from mavka.graph.expand import decay_for_depth
from mavka.graph.expand import expand as _expand
from mavka.index.ivf import IVFIndex
from mavka.lifecycle.compaction import compact as _compact
from mavka.retrieval.keying import make_key
from mavka.storage.log import AppendLog
from mavka.storage.segments import SegmentStore
from mavka.storage.tiered import TieredStore

_DEFAULT_EXPANDER_MAX_NODES = 50


class Memory:
    """Single configurable store + retrieval path, replacing what used to
    be three separate, near-identical implementations of the same
    log+index+retrieve machinery under different fixed configurations --
    a z-only store, an action-in-the-key variant, and a module of
    free-function eval loops layered on top. Every difference between
    those is now a constructor knob instead of a different class or
    module.

    Knob -> old behavior it reproduces:
    - action_scale == 0.0 (default): the index key is z alone (no
      concatenation at all) -- reproduces the old z-only store exactly.
      The default index, if none is given, is sized `dim`.
    - action_scale > 0.0: the index key is
      make_key(z, action, action_scale, action_dim) -- z concatenated
      with scale*action, then normalized -- reproducing the old
      action-in-the-key store exactly. The default index, if none is
      given, is sized `dim + action_dim`.
    - scorer is None (default): recall() returns the index's raw search
      order -- reproduces the plain recall() both old classes had.
    - scorer is not None: recall() over-fetches k * fetch_factor
      candidates, optionally expands them over graph (see below), then
      re-ranks with scorer.score() and keeps the top k -- reproduces
      recall_scored(). There is no separate recall_scored method:
      whether a scorer is configured is what decides this, inside the
      one recall().
    - expansion_depth == 0 (default): graph is never consulted, even if
      one is configured -- matches recall_scored's expand_depth=0
      off-switch exactly. Expansion only ever runs as part of the
      scorer-present path above (exactly as before: only recall_scored
      ever touched graph expansion, never plain recall()).
    - expansion_depth > 0 with graph set: candidates are expanded via
      expander(seed_ids, graph, depth=expansion_depth) before scoring.
      expander defaults to mavka.graph.expand.expand with the same
      max_nodes=50, edge_types=None defaults recall_scored itself used;
      pass a pre-configured expander (e.g.
      functools.partial(expand, max_nodes=200)) to override those.

    z_next (observe's third positional arg) is accepted only for calling
    convenience against the step dicts generate_trajectory produces, as
    in both old classes -- it is never stored; a step's outcome is simply
    the z of the next observe() call in the same episode.

    Lifecycle wiring (all off by default -- see lifecycle/maintenance.py
    for the worker that actually drives these):
    - eviction_policy: an EvictionPolicy instance. If set, recall()
      still never calls it directly (see the hot-path note below); it is
      only ever advanced by MaintenanceWorker via drain_feedback and
      used by compact()/evict()/migrate() below.
    - feedback_buffer: a FeedbackBuffer instance. If set, every recall()
      call appends the ids it returns via feedback_buffer.record_used()
      -- a trivial, negligible-cost append, no scoring or policy work on
      the hot path. Left None (the default), recall() is unchanged.
    - use_tiering: if True, storage is a TieredStore (hot/cold split)
      instead of a single index -- see migrate() below. Only supported
      with action_scale=0.0 (TieredStore indexes on z alone).
    """

    def __init__(
        self,
        config,
        *,
        index=None,
        store_path=None,
        scorer=None,
        graph=None,
        expander=None,
        action_scale: float = 0.0,
        expansion_depth: int = 0,
        fetch_factor: int = 5,
        eviction_policy=None,
        feedback_buffer=None,
        use_tiering: bool = False,
        hot_capacity: int = 200,
    ):
        self.config = config
        self.dim = config.dim
        self.action_dim = config.action_dim
        self.action_scale = action_scale
        self.expansion_depth = expansion_depth
        self.fetch_factor = fetch_factor
        self.scorer = scorer
        self.graph = graph
        self.expander = (
            expander
            if expander is not None
            else functools.partial(_expand, max_nodes=_DEFAULT_EXPANDER_MAX_NODES, edge_types=None)
        )
        self.eviction_policy = eviction_policy
        self.feedback_buffer = feedback_buffer
        self.use_tiering = use_tiering
        self.last_feedback_token = None

        if use_tiering:
            if action_scale > 0.0:
                raise ValueError(
                    "use_tiering=True only supports action_scale=0.0: TieredStore "
                    "indexes on z alone, with no action-conditioned keying."
                )
            self._tiered = TieredStore(
                dim=self.dim,
                action_dim=self.action_dim,
                hot_capacity=hot_capacity,
                eviction_policy=eviction_policy,
            )
            # TieredStore owns the one true log; Memory does not keep a
            # second, separately-appended copy.
            self._log = self._tiered._log
            self._index = None
        else:
            self._tiered = None
            self._log = AppendLog(dim=self.dim, action_dim=self.action_dim)
            if index is not None:
                self._index = index
            else:
                index_dim = self.dim if action_scale == 0.0 else self.dim + self.action_dim
                self._index = IVFIndex(dim=index_dim)

        self._segment_store = (
            SegmentStore(store_path, dim=self.dim, action_dim=self.action_dim)
            if store_path is not None
            else None
        )

    def observe(self, z, action, z_next, pred_err: float = 0.0, episode_id: int = 0) -> int:
        z = normalize(np.asarray(z, dtype=np.float32))

        if self._tiered is not None:
            log_id = self._tiered.observe(
                z=z, action=action, z_next=z_next, pred_err=pred_err, episode_id=episode_id
            )
            if self._segment_store is not None:
                self._segment_store.append_many([self._log.get(log_id)])
            return log_id

        log_id = self._log.append(z=z, action=action, pred_err=pred_err, episode_id=episode_id)

        if self._segment_store is not None:
            self._segment_store.append_many([self._log.get(log_id)])

        if self.action_scale > 0.0:
            key = make_key(z, action, self.action_scale, self.action_dim)
            self._index.add(key)
        else:
            self._index.add(z)

        return log_id

    def recall(self, z, action=None, k: int = 8) -> list[tuple[int, float]]:
        z = normalize(np.asarray(z, dtype=np.float32))

        if self.action_scale > 0.0:
            query_key = make_key(z, action, self.action_scale, self.action_dim)
        else:
            query_key = z

        if self.scorer is None:
            if self._tiered is not None:
                results = self._tiered.recall(query_key, k)
            else:
                results = self._index.search(query_key, k)
        else:
            if self._tiered is not None:
                candidates = self._tiered.recall(query_key, k * self.fetch_factor)
            else:
                candidates = self._index.search(query_key, k * self.fetch_factor)

            if self.graph is not None and self.expansion_depth > 0:
                seed_scores = dict(candidates)
                expanded = self.expander(
                    list(seed_scores.keys()), self.graph, depth=self.expansion_depth
                )
                candidates = [
                    (node_id, seed_scores[node_id])
                    if prov["is_seed"]
                    else (node_id, prov["weight"] * decay_for_depth(prov["depth"]))
                    for node_id, prov in expanded
                ]

            results = self.scorer.score(candidates, action)[:k]

        # Hot-path hook: a trivial append, nothing else -- see
        # FeedbackBuffer's own docstring for the deferred-drain design
        # this participates in. No-op (default) when no buffer is
        # attached. record_used() returns a token that tag_outcome()
        # needs; recall()'s own return type stays exactly
        # list[tuple[int, float]], so the token is exposed here instead,
        # for the harness to read right after calling recall() and pass
        # to self.feedback_buffer.tag_outcome(...) once the step's
        # verdict is known -- call recall() again (or on another step)
        # only after tagging, since this always holds only the latest
        # token.
        if self.feedback_buffer is not None:
            self.last_feedback_token = self.feedback_buffer.record_used(
                [id_ for id_, _ in results]
            )

        return results

    def get(self, id: int) -> Experience:
        return self._log.get(id)

    @property
    def count(self) -> int:
        return self._log.count

    def compact(
        self,
        similarity_threshold: float = 0.98,
        merge: bool = True,
        allow_cross_episode: bool = False,
    ) -> dict:
        """Delegates to lifecycle.compaction.compact against this
        memory's own log/index/graph, then swaps this memory's own
        references over to the compacted result -- never called from the
        hot path, meant to be driven by MaintenanceWorker.

        Only supported for a plain (non-tiered), z-only (action_scale=0.0)
        memory. Two confirmed, genuine incompatibilities in the existing
        lifecycle code (not something this wiring papers over -- see
        lifecycle/maintenance.py's module docstring for the full writeup):
        - compact()'s index rebuild always re-inserts plain z vectors
          (see lifecycle/compaction.py); an action-conditioned index's
          keys are a different length (z concatenated with scaled
          action), so this raises inside compact() itself the moment
          action_scale > 0.
        - compact() always rebuilds the log with new ids, but
          TieredStore's own docstring is explicit that its ids must
          never be reassigned (its internal tier bookkeeping is keyed by
          them) -- composing the two would silently corrupt that
          bookkeeping, so this is refused up front rather than allowed.
        """
        if self._tiered is not None:
            raise ValueError(
                "Memory.compact() is not supported when use_tiering=True: "
                "compact() always reassigns ids, which would corrupt "
                "TieredStore's id-keyed tier bookkeeping. Use migrate() instead."
            )
        if self.action_scale > 0.0:
            raise ValueError(
                "Memory.compact() only supports action_scale=0.0: compact()'s "
                "index rebuild always re-inserts plain z vectors, incompatible "
                f"with this memory's action-conditioned (action_scale={self.action_scale}) index."
            )

        index_factory = functools.partial(type(self._index), dim=self.dim)
        result = _compact(
            self._log,
            self._index,
            index_factory,
            graph=self.graph,
            similarity_threshold=similarity_threshold,
            merge=merge,
            allow_cross_episode=allow_cross_episode,
        )
        self._swap_rebuilt(result)
        return result

    def evict(self, capacity: int, now_ns: int | None = None) -> dict:
        """Delegates to self.eviction_policy.evict_to_capacity against
        this memory's own log/index/graph, then swaps this memory's own
        references over to the result. Same composability constraints as
        compact() (evict_to_capacity calls compact() internally) -- see
        compact()'s docstring.
        """
        if self.eviction_policy is None:
            raise ValueError("Memory.evict() requires an eviction_policy to be configured")
        if self._tiered is not None:
            raise ValueError(
                "Memory.evict() is not supported when use_tiering=True (see compact()'s docstring)"
            )
        if self.action_scale > 0.0:
            raise ValueError(
                "Memory.evict() only supports action_scale=0.0 (see compact()'s docstring)"
            )

        index_factory = functools.partial(type(self._index), dim=self.dim)
        result = self.eviction_policy.evict_to_capacity(
            self._log, self._index, index_factory, capacity, graph=self.graph, now_ns=now_ns
        )
        self._swap_rebuilt(result)
        return result

    def migrate(self, now_ns: int | None = None) -> dict:
        """Delegates to the tiered store's migrate_hot_to_cold. Only
        supported when use_tiering=True; ids never change (TieredStore's
        own invariant), so there is no log/index swap to do here.
        """
        if self._tiered is None:
            raise ValueError("Memory.migrate() requires use_tiering=True")
        return self._tiered.migrate_hot_to_cold(now_ns=now_ns)

    def _swap_rebuilt(self, result: dict) -> None:
        """Both compact() and evict_to_capacity() return a snapshot
        rebuilt from scratch (new ids, new log/index/graph) and
        explicitly document that it's the caller's job to swap every
        reference over -- see compaction.py's own docstring. Memory owns
        log/index/graph directly, but an attached scorer (bound to
        memory's log at construction, since FixedWeightScorer needs a
        real log up front -- see Memory's class docstring) holds its own
        separate reference that would otherwise go stale the moment ids
        are reassigned, so it is rebound here too.
        """
        self._log = result["log"]
        self._index = result["index"]
        self.graph = result["graph"]
        if self.scorer is not None and hasattr(self.scorer, "_log"):
            self.scorer._log = self._log

    def close(self) -> None:
        if self._segment_store is not None:
            self._segment_store.close()
