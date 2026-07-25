import json

import numpy as np
import pytest

from mavka.record import Experience, serialize_many
from mavka.segments import SegmentStore


def _make_experience(id_, dim=4, action_dim=None, episode_id=0, seq_no=None):
    rng = np.random.default_rng(id_)
    z = rng.standard_normal(dim).astype(np.float32)
    action = rng.standard_normal(action_dim).astype(np.float32) if action_dim is not None else None
    return Experience(
        id=id_,
        episode_id=episode_id,
        seq_no=seq_no if seq_no is not None else id_,
        timestamp_ns=1_000_000 + id_,
        pred_err=float(np.float32(0.01 * id_)),
        flags=0,
        z=z,
        action=action,
    )


def test_scan_returns_records_in_id_order_with_exact_values(tmp_path):
    dim = 4
    n = 20
    records = [_make_experience(i, dim=dim) for i in range(n)]

    store = SegmentStore(tmp_path, dim=dim, segment_size=1000)
    store.append_many(records)

    result = list(store.scan())

    assert [r.id for r in result] == list(range(n))
    for original, restored in zip(records, result):
        assert restored.episode_id == original.episode_id
        assert restored.seq_no == original.seq_no
        assert restored.timestamp_ns == original.timestamp_ns
        assert restored.pred_err == original.pred_err
        assert restored.flags == original.flags
        np.testing.assert_array_equal(restored.z, original.z)

    store.close()


def test_records_survive_close_and_reopen(tmp_path):
    dim = 4
    n = 15
    records = [_make_experience(i, dim=dim) for i in range(n)]

    store = SegmentStore(tmp_path, dim=dim, segment_size=1000)
    store.append_many(records)
    store.close()

    reopened = SegmentStore.open(tmp_path)
    assert reopened.count == n
    result = list(reopened.scan())
    assert [r.id for r in result] == list(range(n))
    for original, restored in zip(records, result):
        np.testing.assert_array_equal(restored.z, original.z)

    reopened.close()


def test_segment_sealing_creates_multiple_files_and_reconstructs(tmp_path):
    dim = 4
    segment_size = 10
    n = 25
    records = [_make_experience(i, dim=dim) for i in range(n)]

    store = SegmentStore(tmp_path, dim=dim, segment_size=segment_size)
    for i in range(0, n, 5):
        store.append_many(records[i : i + 5])
    store.close()

    seg_files = sorted(tmp_path.glob("seg_*.dat"))
    assert len(seg_files) == 2

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest["segments"]) == 2
    assert manifest["segments"][0]["record_count"] == segment_size
    assert manifest["segments"][1]["record_count"] == segment_size
    assert manifest["segments"][0]["first_id"] == 0
    assert manifest["segments"][1]["first_id"] == segment_size

    reopened = SegmentStore.open(tmp_path)
    assert reopened.count == n
    result = list(reopened.scan())
    assert [r.id for r in result] == list(range(n))
    reopened.close()


def test_crash_recovery_without_close(tmp_path):
    dim = 4
    n = 7
    records = [_make_experience(i, dim=dim) for i in range(n)]

    store = SegmentStore(tmp_path, dim=dim, segment_size=1000)
    store.append_many(records)
    # Simulate a crash: no store.close(), just abandon the object.

    recovered = SegmentStore.open(tmp_path)
    assert recovered.count == n
    result = list(recovered.scan())
    assert [r.id for r in result] == list(range(n))
    recovered.close()


def test_torn_write_recovery_ignores_partial_trailing_bytes(tmp_path):
    dim = 4
    n = 5
    records = [_make_experience(i, dim=dim) for i in range(n)]

    store = SegmentStore(tmp_path, dim=dim, segment_size=1000)
    store.append_many(records)
    store.close()

    wal_path = tmp_path / "wal.dat"
    with open(wal_path, "ab") as f:
        f.write(b"\x00" * 7)  # smaller than one record (52 bytes for dim=4)

    recovered = SegmentStore.open(tmp_path)
    assert recovered.count == n
    result = list(recovered.scan())
    assert [r.id for r in result] == list(range(n))
    recovered.close()


def test_mid_seal_crash_dedups_on_recovery(tmp_path):
    dim = 4
    segment_size = 10
    records = [_make_experience(i, dim=dim) for i in range(segment_size)]

    store = SegmentStore(tmp_path, dim=dim, segment_size=segment_size)
    store.append_many(records)  # exactly fills segment_size -> triggers a real seal
    store.close()

    wal_path = tmp_path / "wal.dat"
    # Right after a successful seal, wal.dat holds only the header.
    header_bytes = wal_path.read_bytes()

    # Simulate a crash that happened before the WAL-truncation step of sealing
    # completed: the WAL still holds the same records already sealed into
    # seg_00000.dat.
    stale_payload = header_bytes + serialize_many(records, dim, None)
    wal_path.write_bytes(stale_payload)

    recovered = SegmentStore.open(tmp_path)
    assert recovered.count == segment_size
    ids = [r.id for r in recovered.scan()]
    assert ids == list(range(segment_size))
    assert len(set(ids)) == segment_size
    recovered.close()


def test_wrong_magic_raises_on_open(tmp_path):
    dim = 4
    store = SegmentStore(tmp_path, dim=dim, segment_size=1000)
    store.append_many([_make_experience(0, dim=dim)])
    store.close()

    wal_path = tmp_path / "wal.dat"
    data = bytearray(wal_path.read_bytes())
    data[0:4] = b"XXXX"
    wal_path.write_bytes(bytes(data))

    with pytest.raises(ValueError):
        SegmentStore.open(tmp_path)


def test_dim_mismatch_raises_on_construction(tmp_path):
    dim = 4
    store = SegmentStore(tmp_path, dim=dim, segment_size=1000)
    store.close()

    with pytest.raises(ValueError):
        SegmentStore(tmp_path, dim=8, segment_size=1000)
