import itertools
import threading
import time
from collections.abc import Iterator
from dataclasses import replace

import numpy as np

from mavka.record import FLAG_DELETED, Experience


class AppendLog:
    def __init__(self, dim: int, action_dim: int | None = None):
        self._dim = dim
        self._action_dim = action_dim
        self._records: list[Experience] = []
        self._episode_seq_counters: dict[int, int] = {}
        self._episode_index: dict[int, list[int]] = {}
        self._id_counter = itertools.count()
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        return len(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def _validate_shapes(self, z: np.ndarray, action: np.ndarray | None) -> None:
        if z.ndim != 1 or z.shape[0] != self._dim:
            raise ValueError(f"z must have length {self._dim}, got shape {z.shape}")
        if self._action_dim is None:
            if action is not None:
                raise ValueError("action must be None when action_dim is None")
        else:
            if action is None:
                raise ValueError(f"action must be provided when action_dim is {self._action_dim}")
            if action.ndim != 1 or action.shape[0] != self._action_dim:
                raise ValueError(
                    f"action must have length {self._action_dim}, got shape {action.shape}"
                )

    def append(
        self,
        z,
        action=None,
        pred_err: float = 0.0,
        episode_id: int = 0,
        timestamp_ns: int | None = None,
    ) -> int:
        z = np.asarray(z, dtype=np.float32)
        if action is not None:
            action = np.asarray(action, dtype=np.float32)
        self._validate_shapes(z, action)

        if timestamp_ns is None:
            timestamp_ns = time.time_ns()

        with self._lock:
            id_ = next(self._id_counter)
            seq_no = self._episode_seq_counters.get(episode_id, 0)
            self._episode_seq_counters[episode_id] = seq_no + 1

            exp = Experience(
                id=id_,
                episode_id=episode_id,
                seq_no=seq_no,
                timestamp_ns=timestamp_ns,
                pred_err=pred_err,
                flags=0,
                z=z,
                action=action,
            )
            self._records.append(exp)
            self._episode_index.setdefault(episode_id, []).append(id_)

        return id_

    def append_many(
        self,
        zs,
        actions=None,
        pred_errs=None,
        episode_ids=None,
        timestamps_ns=None,
    ) -> list[int]:
        zs = np.asarray(zs, dtype=np.float32)
        n = zs.shape[0]

        if actions is None:
            actions = [None] * n
        if pred_errs is None:
            pred_errs = [0.0] * n
        if episode_ids is None:
            episode_ids = [0] * n
        if timestamps_ns is None:
            timestamps_ns = [None] * n

        return [
            self.append(zs[i], actions[i], pred_errs[i], episode_ids[i], timestamps_ns[i])
            for i in range(n)
        ]

    def get(self, id: int) -> Experience:
        if id < 0 or id >= len(self._records):
            raise ValueError(f"id {id!r} out of range [0, {len(self._records)})")
        return self._records[id]

    def scan(self, start_id: int = 0, end_id: int | None = None) -> Iterator[Experience]:
        end = len(self._records) if end_id is None else end_id
        id_ = start_id
        while id_ < end:
            yield self._records[id_]
            id_ += 1

    def tombstone(self, id: int) -> None:
        with self._lock:
            record = self.get(id)
            self._records[id] = replace(record, flags=record.flags | FLAG_DELETED)

    def _episode_member_ids(self, episode_id: int) -> list[int]:
        if episode_id not in self._episode_index:
            raise ValueError(f"unknown episode_id {episode_id!r}")
        return self._episode_index[episode_id]

    def next_in_episode(self, id: int) -> Experience | None:
        record = self.get(id)
        member_ids = self._episode_index[record.episode_id]
        next_seq_no = record.seq_no + 1
        if next_seq_no >= len(member_ids):
            return None
        return self._records[member_ids[next_seq_no]]

    def prev_in_episode(self, id: int) -> Experience | None:
        record = self.get(id)
        if record.seq_no == 0:
            return None
        member_ids = self._episode_index[record.episode_id]
        return self._records[member_ids[record.seq_no - 1]]

    def get_episode(self, episode_id: int) -> list[Experience]:
        member_ids = self._episode_member_ids(episode_id)
        return [self._records[id_] for id_ in member_ids]

    def episode_ids(self) -> list[int]:
        return list(self._episode_index.keys())

    def episode_length(self, episode_id: int) -> int:
        return len(self._episode_member_ids(episode_id))

    def walk_forward(self, id: int, n: int) -> list[Experience]:
        record = self.get(id)
        member_ids = self._episode_index[record.episode_id]
        start = record.seq_no + 1
        end = min(start + n, len(member_ids))
        return [self._records[id_] for id_ in member_ids[start:end]]

    def walk_back(self, id: int, n: int) -> list[Experience]:
        record = self.get(id)
        member_ids = self._episode_index[record.episode_id]
        end = record.seq_no
        start = max(0, end - n)
        return [self._records[id_] for id_ in reversed(member_ids[start:end])]
