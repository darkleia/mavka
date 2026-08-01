import numpy as np

from mavka.adapter import SyntheticWorldModel
from mavka.eval.sweep import evaluate
from mavka.pipeline import build_pipeline_from_adapter
from mavka.index.flat import FlatIndex


def main() -> None:
    dim = 64
    action_dim = 8
    n_episodes = 50
    episode_length = 40
    k = 10
    nprobe = 10

    adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=0)
    pipeline = build_pipeline_from_adapter(adapter, n_episodes=n_episodes, episode_length=episode_length)

    print(f"pipeline count: {pipeline.count}")

    # Pick a mid-episode step (not seq_no 0) so there's real trajectory
    # history behind it, rather than a fresh post-reset() state.
    query_id = (n_episodes // 2) * episode_length + episode_length // 2
    query_record = pipeline.get(query_id)
    results = pipeline.recall(query_record.z, k=k)

    print(
        f"\nquery id={query_id} (episode={query_record.episode_id}, "
        f"seq_no={query_record.seq_no})"
    )
    print("top results:")
    for id_, score in results:
        record = pipeline.get(id_)
        print(f"  id={id_} score={score:.4f} episode={record.episode_id} seq_no={record.seq_no}")

    assert results[0][0] == query_id
    print(f"\nself-retrieval OK: top hit is id={results[0][0]} with score={results[0][1]:.4f}")

    # Brute-force pipeline on the same trajectories, as ground truth.
    reference_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=0)
    brute_pipeline = build_pipeline_from_adapter(
        reference_adapter,
        n_episodes=n_episodes,
        episode_length=episode_length,
        index=FlatIndex(dim=dim),
    )

    pipeline._index.nprobe = nprobe
    queries = np.stack([pipeline.get(i).z for i in range(0, pipeline.count, 20)])
    result = evaluate(pipeline._index, brute_pipeline._index, queries, k=k)
    print(f"\nIVF recall@{k} (nprobe={nprobe}): {result['mean_recall']:.4f}")


if __name__ == "__main__":
    main()
