import numpy as np

from mavka.eval import format_sweep_table, sweep_nprobe
from mavka.ivf import IVFIndex
from mavka.store import VectorStore


def _random_vectors(n, dim, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def _build_index_and_ground_truth(n=2000, dim=16, n_lists=20, seed=0):
    vectors = _random_vectors(n, dim, seed=seed)

    ground_truth = VectorStore(dim=dim)
    ground_truth.add_batch(vectors)

    index = IVFIndex(dim=dim, n_lists=n_lists)
    index.train(vectors)
    index.add_batch(vectors)

    return index, ground_truth, n_lists


def test_sweep_returns_one_row_per_nprobe_with_expected_keys():
    index, ground_truth, n_lists = _build_index_and_ground_truth()
    queries = _random_vectors(20, 16, seed=1)
    nprobe_values = [1, 5, 10, n_lists]

    rows = sweep_nprobe(index, ground_truth, queries, k=10, nprobe_values=nprobe_values)

    assert len(rows) == len(nprobe_values)
    expected_keys = {"nprobe", "mean_recall", "p50_ms", "p95_ms", "p99_ms", "mean_ms"}
    for row in rows:
        assert set(row.keys()) == expected_keys


def test_recall_is_monotonically_non_decreasing():
    index, ground_truth, n_lists = _build_index_and_ground_truth()
    queries = _random_vectors(30, 16, seed=2)
    nprobe_values = [1, 2, 4, 8, 16, n_lists]

    rows = sweep_nprobe(index, ground_truth, queries, k=10, nprobe_values=nprobe_values)

    recalls = [row["mean_recall"] for row in rows]
    assert all(recalls[i] <= recalls[i + 1] for i in range(len(recalls) - 1))


def test_recall_is_one_at_nprobe_equals_n_lists():
    index, ground_truth, n_lists = _build_index_and_ground_truth()
    queries = _random_vectors(20, 16, seed=3)

    rows = sweep_nprobe(index, ground_truth, queries, k=10, nprobe_values=[n_lists])

    assert rows[0]["mean_recall"] == 1.0


def test_latency_generally_rises_with_nprobe():
    index, ground_truth, n_lists = _build_index_and_ground_truth(n=5000, dim=32, n_lists=50)
    queries = _random_vectors(30, 32, seed=4)
    nprobe_values = [1, 5, 10, 25, 50]

    rows = sweep_nprobe(index, ground_truth, queries, k=10, nprobe_values=nprobe_values)

    assert rows[-1]["mean_ms"] > rows[0]["mean_ms"]


def test_format_sweep_table_contains_each_nprobe_value():
    index, ground_truth, n_lists = _build_index_and_ground_truth()
    queries = _random_vectors(10, 16, seed=5)
    nprobe_values = [1, 5, n_lists]

    rows = sweep_nprobe(index, ground_truth, queries, k=10, nprobe_values=nprobe_values)
    table = format_sweep_table(rows)

    assert table != ""
    for nprobe in nprobe_values:
        assert str(nprobe) in table


def test_index_nprobe_is_restored_after_sweep():
    index, ground_truth, n_lists = _build_index_and_ground_truth()
    queries = _random_vectors(10, 16, seed=6)
    index.nprobe = 3

    sweep_nprobe(index, ground_truth, queries, k=10, nprobe_values=[1, 5, n_lists])

    assert index.nprobe == 3
