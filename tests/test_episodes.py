import numpy as np
import pytest

from mavka.log import AppendLog


def _z(seed, dim=4):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def test_interleaved_episodes_come_back_unmixed_in_order():
    log = AppendLog(dim=4)
    ep1_ids = []
    ep2_ids = []
    for i in range(4):
        ep1_ids.append(log.append(_z(i), episode_id=1))
        ep2_ids.append(log.append(_z(100 + i), episode_id=2))

    ep1_records = log.get_episode(1)
    ep2_records = log.get_episode(2)

    assert [r.id for r in ep1_records] == ep1_ids
    assert [r.seq_no for r in ep1_records] == [0, 1, 2, 3]
    assert [r.id for r in ep2_records] == ep2_ids
    assert [r.seq_no for r in ep2_records] == [0, 1, 2, 3]


def test_next_and_prev_in_episode_return_correct_neighbors():
    log = AppendLog(dim=4)
    ids = [log.append(_z(i), episode_id=1) for i in range(5)]

    assert log.next_in_episode(ids[1]).id == ids[2]
    assert log.prev_in_episode(ids[1]).id == ids[0]
    assert log.next_in_episode(ids[3]).id == ids[4]
    assert log.prev_in_episode(ids[3]).id == ids[2]


def test_episode_boundary_next_is_none_despite_higher_global_id():
    log = AppendLog(dim=4)
    ep3_ids = [log.append(_z(i), episode_id=3) for i in range(3)]
    # Append to a different episode afterward, so it has higher global ids
    # than the last record of episode 3.
    log.append(_z(10), episode_id=4)
    log.append(_z(11), episode_id=4)

    last_ep3_id = ep3_ids[-1]
    assert log.next_in_episode(last_ep3_id) is None


def test_episode_boundary_prev_is_none_for_first_record():
    log = AppendLog(dim=4)
    first_id = log.append(_z(0), episode_id=5)
    log.append(_z(1), episode_id=5)

    assert log.prev_in_episode(first_id) is None


def test_walk_forward_returns_n_records_when_available():
    log = AppendLog(dim=4)
    ids = [log.append(_z(i), episode_id=1) for i in range(6)]

    result = log.walk_forward(ids[1], 3)

    assert [r.id for r in result] == ids[2:5]


def test_walk_forward_stops_short_at_episode_end():
    log = AppendLog(dim=4)
    ids = [log.append(_z(i), episode_id=1) for i in range(4)]
    log.append(_z(10), episode_id=2)

    result = log.walk_forward(ids[2], 5)

    assert [r.id for r in result] == [ids[3]]


def test_walk_back_returns_n_records_when_available():
    log = AppendLog(dim=4)
    ids = [log.append(_z(i), episode_id=1) for i in range(6)]

    result = log.walk_back(ids[4], 3)

    assert [r.id for r in result] == [ids[3], ids[2], ids[1]]


def test_walk_back_stops_short_at_episode_start():
    log = AppendLog(dim=4)
    ids = [log.append(_z(i), episode_id=1) for i in range(4)]

    result = log.walk_back(ids[1], 5)

    assert [r.id for r in result] == [ids[0]]


def test_get_episode_ordered_and_length_matches():
    log = AppendLog(dim=4)
    ids = [log.append(_z(i), episode_id=7) for i in range(5)]

    records = log.get_episode(7)

    assert [r.id for r in records] == ids
    assert [r.seq_no for r in records] == list(range(5))
    assert log.episode_length(7) == 5


def test_episode_ids_returns_all_episodes():
    log = AppendLog(dim=4)
    log.append(_z(0), episode_id=1)
    log.append(_z(1), episode_id=2)
    log.append(_z(2), episode_id=3)

    assert set(log.episode_ids()) == {1, 2, 3}


def test_single_record_episode_has_no_next_or_prev():
    log = AppendLog(dim=4)
    id0 = log.append(_z(0), episode_id=9)

    assert log.next_in_episode(id0) is None
    assert log.prev_in_episode(id0) is None


def test_unknown_id_raises():
    log = AppendLog(dim=4)
    log.append(_z(0), episode_id=1)

    with pytest.raises(ValueError):
        log.next_in_episode(999)
    with pytest.raises(ValueError):
        log.prev_in_episode(999)
    with pytest.raises(ValueError):
        log.walk_forward(999, 1)
    with pytest.raises(ValueError):
        log.walk_back(999, 1)


def test_unknown_episode_id_raises():
    log = AppendLog(dim=4)
    log.append(_z(0), episode_id=1)

    with pytest.raises(ValueError):
        log.get_episode(999)
    with pytest.raises(ValueError):
        log.episode_length(999)
