import threading

import numpy as np
import pytest

from mavka.log import AppendLog
from mavka.record import FLAG_DELETED


def _z(dim, seed=0):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def test_empty_log_has_count_zero():
    log = AppendLog(dim=4)
    assert log.count == 0
    assert len(log) == 0


def test_get_nonexistent_id_raises():
    log = AppendLog(dim=4)
    with pytest.raises(ValueError):
        log.get(0)


def test_append_returns_sequential_ids_and_get_matches():
    log = AppendLog(dim=4)
    id0 = log.append(_z(4, 0))
    id1 = log.append(_z(4, 1))
    id2 = log.append(_z(4, 2))

    assert (id0, id1, id2) == (0, 1, 2)

    np.testing.assert_array_equal(log.get(id0).z, _z(4, 0))
    np.testing.assert_array_equal(log.get(id1).z, _z(4, 1))
    np.testing.assert_array_equal(log.get(id2).z, _z(4, 2))


def test_count_and_len_track_appends():
    log = AppendLog(dim=4)
    for i in range(5):
        log.append(_z(4, i))
        assert log.count == i + 1
        assert len(log) == i + 1


def test_append_many_returns_correct_sequential_ids():
    log = AppendLog(dim=4)
    log.append(_z(4, 0))
    zs = np.stack([_z(4, i) for i in range(1, 4)])
    ids = log.append_many(zs)
    assert ids == [1, 2, 3]
    assert log.count == 4


def test_seq_no_increments_per_episode_independently():
    log = AppendLog(dim=4)
    id_e1_0 = log.append(_z(4, 0), episode_id=1)
    id_e2_0 = log.append(_z(4, 1), episode_id=2)
    id_e1_1 = log.append(_z(4, 2), episode_id=1)
    id_e2_1 = log.append(_z(4, 3), episode_id=2)
    id_e1_2 = log.append(_z(4, 4), episode_id=1)

    assert log.get(id_e1_0).seq_no == 0
    assert log.get(id_e1_1).seq_no == 1
    assert log.get(id_e1_2).seq_no == 2
    assert log.get(id_e2_0).seq_no == 0
    assert log.get(id_e2_1).seq_no == 1


def test_scan_yields_in_insertion_order():
    log = AppendLog(dim=4)
    for i in range(5):
        log.append(_z(4, i))

    result = list(log.scan())
    assert [r.id for r in result] == [0, 1, 2, 3, 4]


def test_scan_with_start_id():
    log = AppendLog(dim=4)
    for i in range(5):
        log.append(_z(4, i))

    result = list(log.scan(start_id=2))
    assert [r.id for r in result] == [2, 3, 4]


def test_scan_with_start_and_end_id():
    log = AppendLog(dim=4)
    for i in range(5):
        log.append(_z(4, i))

    result = list(log.scan(start_id=1, end_id=3))
    assert [r.id for r in result] == [1, 2]


def test_tombstone_sets_deleted_flag_but_keeps_record():
    log = AppendLog(dim=4)
    id0 = log.append(_z(4, 0))
    log.tombstone(id0)

    record = log.get(id0)
    assert record.flags & FLAG_DELETED

    scanned_ids = [r.id for r in log.scan()]
    assert id0 in scanned_ids


def test_concurrent_appends_produce_unique_sequential_ids():
    log = AppendLog(dim=4)
    num_threads = 8
    appends_per_thread = 50
    total = num_threads * appends_per_thread
    all_ids = []
    lock = threading.Lock()

    def worker(thread_seed):
        for i in range(appends_per_thread):
            new_id = log.append(_z(4, thread_seed * 1000 + i))
            with lock:
                all_ids.append(new_id)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert log.count == total
    assert len(all_ids) == total
    assert len(set(all_ids)) == total
    assert set(all_ids) == set(range(total))


def test_records_with_no_action_round_trip():
    log = AppendLog(dim=4, action_dim=None)
    id0 = log.append(_z(4, 0))
    record = log.get(id0)
    assert record.action is None
    np.testing.assert_array_equal(record.z, _z(4, 0))


def test_records_with_action_round_trip():
    log = AppendLog(dim=4, action_dim=2)
    action = np.array([1.0, 2.0], dtype=np.float32)
    id0 = log.append(_z(4, 0), action=action)
    record = log.get(id0)
    np.testing.assert_array_equal(record.action, action)
