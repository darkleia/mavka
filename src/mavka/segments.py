import json
import mmap
import os
import struct
from collections.abc import Iterator
from pathlib import Path

from mavka.record import (
    Experience,
    RecordLayout,
    deserialize,
    deserialize_many,
    serialize_many,
)

MAGIC = b"MVKA"
FORMAT_VERSION = 1
HEADER_FORMAT = "<4sIIiQ"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def _pack_header(dim: int, action_dim: int | None, record_count: int) -> bytes:
    action_dim_field = -1 if action_dim is None else action_dim
    return struct.pack(HEADER_FORMAT, MAGIC, FORMAT_VERSION, dim, action_dim_field, record_count)


def _unpack_header(buf: bytes, path: Path) -> tuple[int, int | None, int]:
    if len(buf) < HEADER_SIZE:
        raise ValueError(f"{path}: file is smaller than the {HEADER_SIZE}-byte header")
    magic, version, dim, action_dim_field, record_count = struct.unpack(
        HEADER_FORMAT, buf[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise ValueError(f"{path}: bad magic bytes {magic!r}, expected {MAGIC!r}")
    if version != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported format version {version}, expected {FORMAT_VERSION}")
    action_dim = None if action_dim_field == -1 else action_dim_field
    return dim, action_dim, record_count


class SegmentStore:
    def __init__(self, path, dim: int, action_dim: int | None = None, segment_size: int = 100_000):
        self.path = Path(path)
        self.segment_size = segment_size
        self._wal_path = self.path / "wal.dat"
        self._manifest_path = self.path / "manifest.json"

        if self._manifest_path.exists():
            manifest = self._read_manifest()
            if manifest["dim"] != dim or manifest["action_dim"] != action_dim:
                raise ValueError(
                    f"store at {self.path} was created with dim={manifest['dim']}, "
                    f"action_dim={manifest['action_dim']}; got dim={dim}, action_dim={action_dim}"
                )
            self.dim = dim
            self.action_dim = action_dim
            self._segments = manifest["segments"]
            self._layout = RecordLayout(dim, action_dim)
            self._verify_segments()
            self._wal_tail = self._recover_wal()
        else:
            self.path.mkdir(parents=True, exist_ok=True)
            self.dim = dim
            self.action_dim = action_dim
            self._segments = []
            self._layout = RecordLayout(dim, action_dim)
            self._write_manifest()
            with open(self._wal_path, "wb") as f:
                f.write(_pack_header(dim, action_dim, 0))
                f.flush()
                os.fsync(f.fileno())
            self._fsync_dir()
            self._wal_tail = []

        self._wal_file = open(self._wal_path, "ab")

    @classmethod
    def open(cls, path) -> "SegmentStore":
        path = Path(path)
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"no store found at {path} (missing manifest.json)")
        with open(manifest_path) as f:
            manifest = json.load(f)
        return cls(
            path,
            dim=manifest["dim"],
            action_dim=manifest["action_dim"],
            segment_size=manifest["segment_size"],
        )

    @property
    def count(self) -> int:
        return self._sealed_count() + len(self._wal_tail)

    def _sealed_count(self) -> int:
        return sum(seg["record_count"] for seg in self._segments)

    def append_many(self, records: list[Experience]) -> None:
        if not records:
            return

        payload = serialize_many(records, self.dim, self.action_dim)
        self._wal_file.write(payload)
        self._wal_file.flush()
        os.fsync(self._wal_file.fileno())

        self._wal_tail.extend(records)

        while len(self._wal_tail) >= self.segment_size:
            self._seal_segment()

    def scan(self, start_id: int = 0) -> Iterator[Experience]:
        for seg_meta in self._segments:
            last_id = seg_meta["first_id"] + seg_meta["record_count"]
            if last_id <= start_id:
                continue
            yield from self._iter_segment(seg_meta, start_id)
        for record in self._wal_tail:
            if record.id >= start_id:
                yield record

    def close(self) -> None:
        self._wal_file.close()

    def _iter_segment(self, seg_meta: dict, start_id: int) -> Iterator[Experience]:
        seg_path = self.path / seg_meta["filename"]
        first_id = seg_meta["first_id"]
        record_count = seg_meta["record_count"]
        skip = max(0, start_id - first_id)

        with open(seg_path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for i in range(skip, record_count):
                offset = HEADER_SIZE + i * self._layout.record_size
                buf = mm[offset : offset + self._layout.record_size]
                yield deserialize(buf, self.dim, self.action_dim)

    def _seal_segment(self) -> None:
        to_seal = self._wal_tail[: self.segment_size]
        leftover = self._wal_tail[self.segment_size :]

        first_id = to_seal[0].id
        seg_filename = f"seg_{len(self._segments):05d}.dat"
        seg_path = self.path / seg_filename

        header = _pack_header(self.dim, self.action_dim, len(to_seal))
        payload = serialize_many(to_seal, self.dim, self.action_dim)
        with open(seg_path, "wb") as f:
            f.write(header)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        self._fsync_dir()

        self._segments.append(
            {
                "filename": seg_filename,
                "first_id": first_id,
                "record_count": len(to_seal),
            }
        )
        self._write_manifest()

        self._rotate_wal(leftover)
        self._wal_tail = leftover

    def _rotate_wal(self, leftover_records: list[Experience]) -> None:
        tmp_path = self.path / "wal.dat.tmp"
        header = _pack_header(self.dim, self.action_dim, 0)
        payload = serialize_many(leftover_records, self.dim, self.action_dim)
        with open(tmp_path, "wb") as f:
            f.write(header)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())

        self._wal_file.close()
        os.replace(tmp_path, self._wal_path)
        self._fsync_dir()

        self._wal_file = open(self._wal_path, "ab")

    def _write_manifest(self) -> None:
        manifest = {
            "dim": self.dim,
            "action_dim": self.action_dim,
            "segment_size": self.segment_size,
            "segments": self._segments,
        }
        tmp_path = self.path / "manifest.json.tmp"
        with open(tmp_path, "w") as f:
            json.dump(manifest, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self._manifest_path)
        self._fsync_dir()

    def _read_manifest(self) -> dict:
        with open(self._manifest_path) as f:
            return json.load(f)

    def _fsync_dir(self) -> None:
        fd = os.open(self.path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _check_header_matches(self, dim: int, action_dim: int | None, path: Path) -> None:
        if dim != self.dim or action_dim != self.action_dim:
            raise ValueError(
                f"{path} header (dim={dim}, action_dim={action_dim}) does not match "
                f"store config (dim={self.dim}, action_dim={self.action_dim})"
            )

    def _verify_segments(self) -> None:
        for seg_meta in self._segments:
            seg_path = self.path / seg_meta["filename"]
            with open(seg_path, "rb") as f:
                header_bytes = f.read(HEADER_SIZE)
            dim, action_dim, record_count = _unpack_header(header_bytes, seg_path)
            self._check_header_matches(dim, action_dim, seg_path)
            if record_count != seg_meta["record_count"]:
                raise ValueError(
                    f"{seg_path} header record_count={record_count} does not match "
                    f"manifest record_count={seg_meta['record_count']}"
                )

    def _recover_wal(self) -> list[Experience]:
        with open(self._wal_path, "rb") as f:
            data = f.read()

        dim, action_dim, _ = _unpack_header(data, self._wal_path)
        self._check_header_matches(dim, action_dim, self._wal_path)

        body = data[HEADER_SIZE:]
        usable_len = (len(body) // self._layout.record_size) * self._layout.record_size

        if usable_len != len(body):
            with open(self._wal_path, "r+b") as f:
                f.truncate(HEADER_SIZE + usable_len)
                f.flush()
                os.fsync(f.fileno())
            body = body[:usable_len]

        record_count = usable_len // self._layout.record_size
        records = (
            deserialize_many(body, record_count, self.dim, self.action_dim)
            if record_count
            else []
        )

        sealed_count = self._sealed_count()
        return [record for record in records if record.id >= sealed_count]
