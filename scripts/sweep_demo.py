import numpy as np

from mavka.eval import format_sweep_table, sweep_nprobe
from mavka.ivf import IVFIndex
from mavka.store import VectorStore


def main() -> None:
    dim = 32
    n = 20_000
    n_lists = 128
    n_queries = 200
    k = 10
    nprobe_values = [1, 2, 4, 8, 16, 32, 64, 128]

    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)

    ground_truth = VectorStore(dim=dim)
    ground_truth.add_batch(vectors)

    index = IVFIndex(dim=dim, n_lists=n_lists)
    index.train(vectors)
    index.add_batch(vectors)

    rows = sweep_nprobe(index, ground_truth, queries, k, nprobe_values)
    print(format_sweep_table(rows))


if __name__ == "__main__":
    main()
