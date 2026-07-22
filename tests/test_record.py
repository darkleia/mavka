import struct

import numpy as np
import pytest

from mavka.record import (
    FLAG_DELETED,
    FLAG_PINNED,
    Experience,
    RecordLayout,
    deserialize,
    deserialize_many,
    serialize,
    serialize_many,
)


def _make_experience(dim, action_dim, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(dim).astype(np.float32)
    action = rng.standard_normal(action_dim).astype(np.float32) if action_dim is not None else None
    return Experience(
        id=seed,
        episode_id=seed // 2,
        seq_no=seed,
        timestamp_ns=1_000_000 + seed,
        pred_err=float(np.float32(0.1 * seed)),
        flags=0,
        z=z,
        action=action,
    )


def test_round_trip_basic():
    dim = 8
    action_dim = 3
    exp = _make_experience(dim, action_dim, seed=7)

    buf = serialize(exp, dim=dim, action_dim=action_dim)
    result = deserialize(buf, dim=dim, action_dim=action_dim)

    assert result.id == exp.id
    assert result.episode_id == exp.episode_id
    assert result.seq_no == exp.seq_no
    assert result.timestamp_ns == exp.timestamp_ns
    assert result.pred_err == exp.pred_err
    assert result.flags == exp.flags
    np.testing.assert_array_equal(result.z, exp.z)
    np.testing.assert_array_equal(result.action, exp.action)


def test_record_size_matches_serialized_length():
    dim = 16
    action_dim = 4
    layout = RecordLayout(dim, action_dim)
    exp = _make_experience(dim, action_dim, seed=1)

    buf = serialize(exp, dim=dim, action_dim=action_dim)

    assert len(buf) == layout.record_size


def test_batch_round_trip():
    dim = 8
    action_dim = 2
    experiences = [_make_experience(dim, action_dim, seed=i) for i in range(5)]

    buf = serialize_many(experiences, dim=dim, action_dim=action_dim)
    result = deserialize_many(buf, count=5, dim=dim, action_dim=action_dim)

    assert len(result) == 5
    for original, restored in zip(experiences, result):
        assert restored.id == original.id
        assert restored.flags == original.flags
        np.testing.assert_array_equal(restored.z, original.z)
        np.testing.assert_array_equal(restored.action, original.action)


def test_no_action_round_trip_and_smaller_size():
    dim = 8
    exp = Experience(
        id=1,
        episode_id=1,
        seq_no=0,
        timestamp_ns=100,
        pred_err=0.25,
        flags=0,
        z=np.arange(dim, dtype=np.float32),
        action=None,
    )

    buf = serialize(exp, dim=dim, action_dim=None)
    result = deserialize(buf, dim=dim, action_dim=None)

    assert result.action is None
    np.testing.assert_array_equal(result.z, exp.z)

    with_action_size = RecordLayout(dim, action_dim=4).record_size
    without_action_size = RecordLayout(dim, action_dim=None).record_size
    assert without_action_size < with_action_size


@pytest.mark.parametrize("flags", [0, FLAG_DELETED, FLAG_PINNED, FLAG_DELETED | FLAG_PINNED])
def test_flags_round_trip(flags):
    dim = 4
    exp = Experience(
        id=1,
        episode_id=1,
        seq_no=0,
        timestamp_ns=1,
        pred_err=0.0,
        flags=flags,
        z=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        action=None,
    )

    buf = serialize(exp, dim=dim, action_dim=None)
    result = deserialize(buf, dim=dim, action_dim=None)

    assert result.flags == flags


def test_field_offsets_read_correctly():
    dim = 4
    layout = RecordLayout(dim, action_dim=None)
    exp = Experience(
        id=42,
        episode_id=99,
        seq_no=7,
        timestamp_ns=123_456_789,
        pred_err=1.5,
        flags=FLAG_PINNED,
        z=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        action=None,
    )

    buf = serialize(exp, dim=dim, action_dim=None)

    (episode_id,) = struct.unpack_from("<Q", buf, layout.episode_id_offset)
    assert episode_id == exp.episode_id

    (seq_no,) = struct.unpack_from("<I", buf, layout.seq_no_offset)
    assert seq_no == exp.seq_no

    (flags,) = struct.unpack_from("<B", buf, layout.flags_offset)
    assert flags == exp.flags

    z_value = np.frombuffer(buf, dtype="<f4", count=dim, offset=layout.z_offset)
    np.testing.assert_array_equal(z_value, exp.z)


def test_wrong_z_length_raises():
    exp = Experience(1, 1, 0, 1, 0.0, 0, z=np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError):
        serialize(exp, dim=4, action_dim=None)


def test_action_present_when_none_expected_raises():
    exp = Experience(
        1, 1, 0, 1, 0.0, 0, z=np.zeros(4, dtype=np.float32), action=np.zeros(2, dtype=np.float32)
    )
    with pytest.raises(ValueError):
        serialize(exp, dim=4, action_dim=None)


def test_action_absent_when_expected_raises():
    exp = Experience(1, 1, 0, 1, 0.0, 0, z=np.zeros(4, dtype=np.float32), action=None)
    with pytest.raises(ValueError):
        serialize(exp, dim=4, action_dim=2)


def test_action_wrong_length_raises():
    exp = Experience(
        1, 1, 0, 1, 0.0, 0, z=np.zeros(4, dtype=np.float32), action=np.zeros(3, dtype=np.float32)
    )
    with pytest.raises(ValueError):
        serialize(exp, dim=4, action_dim=2)


def test_deserialize_wrong_buffer_length_raises():
    with pytest.raises(ValueError):
        deserialize(b"\x00" * 10, dim=4, action_dim=None)


def test_deserialize_many_wrong_buffer_length_raises():
    with pytest.raises(ValueError):
        deserialize_many(b"\x00" * 10, count=2, dim=4, action_dim=None)


def test_record_size_regression_guard():
    layout = RecordLayout(dim=1024, action_dim=7)
    assert layout.record_size == 4160
