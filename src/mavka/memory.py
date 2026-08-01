import functools

import numpy as np

from mavka.core.distance import normalize
from mavka.core.record import Experience
from mavka.graph.expand import decay_for_depth
from mavka.graph.expand import expand as _expand
from mavka.index.ivf import IVFIndex
from mavka.retrieval.keying import make_key
from mavka.storage.log import AppendLog
from mavka.storage.segments import SegmentStore

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

        self._log = AppendLog(dim=self.dim, action_dim=self.action_dim)
        self._segment_store = (
            SegmentStore(store_path, dim=self.dim, action_dim=self.action_dim)
            if store_path is not None
            else None
        )

        if index is not None:
            self._index = index
        else:
            index_dim = self.dim if action_scale == 0.0 else self.dim + self.action_dim
            self._index = IVFIndex(dim=index_dim)

    def observe(self, z, action, z_next, pred_err: float = 0.0, episode_id: int = 0) -> int:
        z = normalize(np.asarray(z, dtype=np.float32))

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
            return self._index.search(query_key, k)

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

        ranked = self.scorer.score(candidates, action)
        return ranked[:k]

    def get(self, id: int) -> Experience:
        return self._log.get(id)

    @property
    def count(self) -> int:
        return self._log.count

    def close(self) -> None:
        if self._segment_store is not None:
            self._segment_store.close()
