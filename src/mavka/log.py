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
