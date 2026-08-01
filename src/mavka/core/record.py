import struct
from dataclasses import dataclass

import numpy as np

FLAG_DELETED = 1 << 0
FLAG_PINNED = 1 << 1

_SCALAR_FIELDS = [
    ("id", "Q"),
    ("episode_id", "Q"),
    ("seq_no", "I"),
    ("timestamp_ns", "Q"),
    ("pred_err", "f"),
    ("flags", "B"),
]
_SCALAR_FORMAT = "<" + "".join(code for _, code in _SCALAR_FIELDS)
_SCALAR_SIZE = struct.calcsize(_SCALAR_FORMAT)
_ALIGNMENT = 4
_F4_LE = np.dtype("<f4")


def _align(size: int, alignment: int) -> int:
    return ((size + alignment - 1) // alignment) * alignment


def _scalar_offsets() -> dict:
    offsets = {}
    offset = 0
    for name, code in _SCALAR_FIELDS:
        offsets[name] = offset
        offset += struct.calcsize("<" + code)
    return offsets


_SCALAR_OFFSETS = _scalar_offsets()
_SCALAR_BLOCK_SIZE = _align(_SCALAR_SIZE, _ALIGNMENT)


@dataclass(frozen=True)
class Experience:
    id: int
    episode_id: int
    seq_no: int
    timestamp_ns: int
    pred_err: float
    flags: int
    z: np.ndarray
    action: np.ndarray | None = None


class RecordLayout:
    def __init__(self, dim: int, action_dim: int | None = None):
        self.dim = dim
        self.action_dim = action_dim

        self.id_offset = _SCALAR_OFFSETS["id"]
        self.episode_id_offset = _SCALAR_OFFSETS["episode_id"]
        self.seq_no_offset = _SCALAR_OFFSETS["seq_no"]
        self.timestamp_ns_offset = _SCALAR_OFFSETS["timestamp_ns"]
        self.pred_err_offset = _SCALAR_OFFSETS["pred_err"]
        self.flags_offset = _SCALAR_OFFSETS["flags"]

        self.z_offset = _SCALAR_BLOCK_SIZE
        self.z_size = dim * 4

        self.action_offset = self.z_offset + self.z_size
        self.action_size = (action_dim * 4) if action_dim is not None else 0

        self.record_size = self.action_offset + self.action_size


def _validate(exp: Experience, dim: int, action_dim: int | None) -> None:
    z = np.asarray(exp.z)
    if z.ndim != 1 or z.shape[0] != dim:
        raise ValueError(f"z must have length {dim}, got shape {z.shape}")

    if action_dim is None:
        if exp.action is not None:
            raise ValueError("action must be None when action_dim is None")
    else:
        if exp.action is None:
            raise ValueError(f"action must be provided when action_dim is {action_dim}")
        action = np.asarray(exp.action)
        if action.ndim != 1 or action.shape[0] != action_dim:
            raise ValueError(f"action must have length {action_dim}, got shape {action.shape}")


def serialize(exp: Experience, dim: int, action_dim: int | None = None) -> bytes:
    _validate(exp, dim, action_dim)
    layout = RecordLayout(dim, action_dim)

    buf = bytearray(layout.record_size)
    struct.pack_into(
        _SCALAR_FORMAT,
        buf,
        0,
        exp.id,
        exp.episode_id,
        exp.seq_no,
        exp.timestamp_ns,
        exp.pred_err,
        exp.flags,
    )

    z = np.asarray(exp.z, dtype=np.float32).astype(_F4_LE, copy=False)
    buf[layout.z_offset : layout.z_offset + layout.z_size] = z.tobytes()

    if action_dim is not None:
        action = np.asarray(exp.action, dtype=np.float32).astype(_F4_LE, copy=False)
        buf[layout.action_offset : layout.action_offset + layout.action_size] = action.tobytes()

    return bytes(buf)


def deserialize(buf: bytes, dim: int, action_dim: int | None = None) -> Experience:
    layout = RecordLayout(dim, action_dim)
    if len(buf) != layout.record_size:
        raise ValueError(f"buffer must be exactly {layout.record_size} bytes, got {len(buf)}")

    id_, episode_id, seq_no, timestamp_ns, pred_err, flags = struct.unpack_from(
        _SCALAR_FORMAT, buf, 0
    )

    z = np.frombuffer(buf, dtype=_F4_LE, count=dim, offset=layout.z_offset).astype(np.float32)

    action = None
    if action_dim is not None:
        action = np.frombuffer(
            buf, dtype=_F4_LE, count=action_dim, offset=layout.action_offset
        ).astype(np.float32)

    return Experience(
        id=id_,
        episode_id=episode_id,
        seq_no=seq_no,
        timestamp_ns=timestamp_ns,
        pred_err=pred_err,
        flags=flags,
        z=z,
        action=action,
    )


def serialize_many(experiences: list[Experience], dim: int, action_dim: int | None = None) -> bytes:
    return b"".join(serialize(exp, dim, action_dim) for exp in experiences)


def deserialize_many(
    buf: bytes, count: int, dim: int, action_dim: int | None = None
) -> list[Experience]:
    layout = RecordLayout(dim, action_dim)
    expected_size = layout.record_size * count
    if len(buf) != expected_size:
        raise ValueError(
            f"buffer must be exactly {expected_size} bytes for {count} records, got {len(buf)}"
        )
    return [
        deserialize(buf[i * layout.record_size : (i + 1) * layout.record_size], dim, action_dim)
        for i in range(count)
    ]
