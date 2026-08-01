import time

import numpy as np


def recall_at_k(approx_ids, true_ids) -> float:
    if len(true_ids) == 0:
        return 1.0
    return len(set(approx_ids) & set(true_ids)) / len(true_ids)


def evaluate(index, ground_truth, queries, k: int) -> dict:
    recalls = []
    latencies_ms = []

    for query in queries:
        start = time.perf_counter()
        approx_results = index.search(query, k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

        true_results = ground_truth.search(query, k)

        approx_ids = [id_ for id_, _ in approx_results]
        true_ids = [id_ for id_, _ in true_results]
        recalls.append(recall_at_k(approx_ids, true_ids))

    latencies_ms = np.array(latencies_ms)

    return {
        "mean_recall": float(np.mean(recalls)),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "mean_ms": float(np.mean(latencies_ms)),
    }


def sweep_nprobe(index, ground_truth, queries, k: int, nprobe_values) -> list[dict]:
    """Measure recall/latency across a range of nprobe settings.

    index.nprobe is restored to its original value before returning, even if
    evaluation raises partway through the sweep.
    """
    original_nprobe = index.nprobe
    rows = []
    try:
        for nprobe in sorted(nprobe_values):
            index.nprobe = nprobe
            result = evaluate(index, ground_truth, queries, k)
            rows.append({"nprobe": nprobe, **result})
    finally:
        index.nprobe = original_nprobe

    return rows


def format_sweep_table(rows: list[dict]) -> str:
    header = f"{'nprobe':>8} {'recall':>8} {'p50_ms':>10} {'p95_ms':>10} {'p99_ms':>10}"
    lines = [header]
    for row in rows:
        lines.append(
            f"{row['nprobe']:>8} {row['mean_recall']:>8.4f} "
            f"{row['p50_ms']:>10.4f} {row['p95_ms']:>10.4f} {row['p99_ms']:>10.4f}"
        )
    return "\n".join(lines)
