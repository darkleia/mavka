import json
import os
from pathlib import Path

import numpy as np
import pytest

from mavka import VectorStore

GOLDEN_PATH = Path(__file__).parent / "golden" / "search_results.json"
DIM = 64
CORPUS_SIZE = 1000
NUM_QUERIES = 10
K = 10


def _build_corpus_and_queries():
    rng = np.random.default_rng(42)
    store = VectorStore(dim=DIM)
    vectors = rng.standard_normal((CORPUS_SIZE, DIM)).astype(np.float32)
    store.add_batch(vectors)
    queries = rng.standard_normal((NUM_QUERIES, DIM)).astype(np.float32)
    return store, queries


def _run_searches(store, queries):
    return [store.search(query, k=K) for query in queries]


def _write_golden(results):
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "queries": [
            {
                "ids": [int(id_) for id_, _ in result],
                "scores": [float(score) for _, score in result],
            }
            for result in results
        ]
    }
    GOLDEN_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def test_golden_search_results():
    store, queries = _build_corpus_and_queries()
    results = _run_searches(store, queries)

    if os.environ.get("MAVKA_UPDATE_GOLDEN") == "1":
        _write_golden(results)
        print(f"Updated golden file: {GOLDEN_PATH}")
        return

    if not GOLDEN_PATH.exists():
        _write_golden(results)
        print(f"Golden file did not exist, bootstrapped: {GOLDEN_PATH}")
        return

    expected = json.loads(GOLDEN_PATH.read_text())
    assert len(results) == len(expected["queries"])

    for result, expected_query in zip(results, expected["queries"]):
        actual_ids = [id_ for id_, _ in result]
        actual_scores = [score for _, score in result]
        assert actual_ids == expected_query["ids"]
        assert actual_scores == pytest.approx(expected_query["scores"], abs=1e-5)


def test_search_is_deterministic_across_two_builds():
    store_a, queries_a = _build_corpus_and_queries()
    store_b, queries_b = _build_corpus_and_queries()

    results_a = store_a.search(queries_a[0], k=K)
    results_b = store_b.search(queries_b[0], k=K)

    assert results_a == results_b
