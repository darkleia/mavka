import numpy as np

from mavka.adapter import generate_trajectory
from mavka.ivf import IVFIndex
from mavka.keying import make_key
from mavka.log import AppendLog
from mavka.record import Experience
from mavka.segments import SegmentStore
from mavka.store import normalize


class Pipeline:
    """Ties the append log, optional disk store, and vector index together
    behind one write/read interface, so all three always agree on what
    experience a given id refers to.
    """

    def __init__(self, dim: int, action_dim: int | None = None, index=None, store_path=None):
        self.dim = dim
        self.action_dim = action_dim

        self._log = AppendLog(dim=dim, action_dim=action_dim)
        self._segment_store = (
            SegmentStore(store_path, dim=dim, action_dim=action_dim)
            if store_path is not None
            else None
        )
        self._index = index if index is not None else IVFIndex(dim=dim)

    def observe(
        self, z, action, z_next, pred_err: float = 0.0, episode_id: int = 0
    ) -> int:
        """The single write entry point: normalize z, append it to the log
        (and the segment store, if configured), and add it to the index
        under the same id.

        z_next is accepted for calling convenience (it mirrors the step
        dicts generate_trajectory produces) but is not stored as its own
        field here -- as elsewhere in Mavka, a step's z_next is simply the z
        of the next observe() call in the same episode, reachable via the
        log's episode navigation (next_in_episode).
        """
        z = normalize(np.asarray(z, dtype=np.float32))

        log_id = self._log.append(z=z, action=action, pred_err=pred_err, episode_id=episode_id)

        if self._segment_store is not None:
            self._segment_store.append_many([self._log.get(log_id)])

        self._index.add(z)

        return log_id

    def recall(self, query_z, k: int, action=None) -> list[tuple[int, float]]:
        """Search the index for the k nearest neighbors of query_z.

        action is a placeholder for a future action-conditioned query path
        (not built here) and is currently ignored.
        """
        query_z = normalize(np.asarray(query_z, dtype=np.float32))
        return self._index.search(query_z, k)

    def get(self, id: int) -> Experience:
        return self._log.get(id)

    @property
    def count(self) -> int:
        return self._log.count

    def close(self) -> None:
        if self._segment_store is not None:
            self._segment_store.close()


class ActionConditionedPipeline:
    """Like Pipeline, but indexes on [z ; scale * action] instead of z
    alone, so retrieval can distinguish two near-identical situations that
    lead to different outcomes depending on the action taken.

    The log still stores z and action separately, unchanged; only the index
    key is the concatenation. The underlying index must be sized for
    dim + action_dim (the key length), not dim -- if you pass a custom
    index, construct it with that dimension.
    """

    def __init__(
        self, dim: int, action_dim: int, index=None, store_path=None, scale: float = 1.0
    ):
        self.dim = dim
        self.action_dim = action_dim
        self.scale = scale

        self._log = AppendLog(dim=dim, action_dim=action_dim)
        self._segment_store = (
            SegmentStore(store_path, dim=dim, action_dim=action_dim)
            if store_path is not None
            else None
        )
        self._index = index if index is not None else IVFIndex(dim=dim + action_dim)

    def observe(
        self, z, action, z_next, pred_err: float = 0.0, episode_id: int = 0
    ) -> int:
        z = normalize(np.asarray(z, dtype=np.float32))
        action = np.asarray(action, dtype=np.float32)

        log_id = self._log.append(z=z, action=action, pred_err=pred_err, episode_id=episode_id)

        if self._segment_store is not None:
            self._segment_store.append_many([self._log.get(log_id)])

        key = make_key(z, action, self.scale, self.action_dim)
        self._index.add(key)

        return log_id

    def recall(self, query_z, action, k: int) -> list[tuple[int, float]]:
        query_z = normalize(np.asarray(query_z, dtype=np.float32))
        query_key = make_key(query_z, action, self.scale, self.action_dim)
        return self._index.search(query_key, k)

    def get(self, id: int) -> Experience:
        return self._log.get(id)

    @property
    def count(self) -> int:
        return self._log.count

    def close(self) -> None:
        if self._segment_store is not None:
            self._segment_store.close()


def build_pipeline_from_adapter(
    adapter, n_episodes: int, episode_length: int, index=None, store_path=None
) -> Pipeline:
    """Roll the adapter forward n_episodes trajectories and feed every
    experience through pipeline.observe(). Trajectories are generated up
    front so that, if the index needs training (e.g. a fresh IVFIndex), it
    can be trained once on a representative sample before any observe()
    call needs it -- observe() itself never trains anything.
    """
    all_steps = []
    for episode_id in range(n_episodes):
        all_steps.extend(generate_trajectory(adapter, episode_length, episode_id=episode_id))

    pipeline = Pipeline(
        dim=adapter.dim, action_dim=adapter.action_dim, index=index, store_path=store_path
    )

    if hasattr(pipeline._index, "train") and not getattr(pipeline._index, "is_trained", True):
        training_sample = np.stack([step["z"] for step in all_steps])
        pipeline._index.train(training_sample)

    for step in all_steps:
        pipeline.observe(
            z=step["z"],
            action=step["action"],
            z_next=step["z_next"],
            pred_err=step["pred_err"],
            episode_id=step["episode_id"],
        )

    return pipeline
