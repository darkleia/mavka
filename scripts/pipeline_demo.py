import numpy as np

from mavka.adapter import SyntheticWorldModel, generate_trajectory
from mavka.config import MavkaConfig
from mavka.eval.sweep import evaluate
from mavka.index.flat import FlatIndex
from mavka.memory import Memory


def _build_memory_from_adapter(adapter, n_episodes, episode_length, index=None):
    """Roll the adapter forward n_episodes trajectories and feed every
    experience through memory.observe() -- the Memory-based equivalent of
    the old build_pipeline_from_adapter helper, kept local to this script
    since it's just demo glue, not library surface.
    """
    all_steps = []
    for episode_id in range(n_episodes):
        all_steps.extend(generate_trajectory(adapter, episode_length, episode_id=episode_id))

    config = MavkaConfig(dim=adapter.dim, action_dim=adapter.action_dim)
    memory = Memory(config, index=index, action_scale=0.0)

    if hasattr(memory._index, "train") and not getattr(memory._index, "is_trained", True):
        training_sample = np.stack([step["z"] for step in all_steps])
        memory._index.train(training_sample)

    for step in all_steps:
        memory.observe(
            z=step["z"],
            action=step["action"],
            z_next=step["z_next"],
            pred_err=step["pred_err"],
            episode_id=step["episode_id"],
        )

    return memory


def main() -> None:
    dim = 64
    action_dim = 8
    n_episodes = 50
    episode_length = 40
    k = 10
    nprobe = 10

    adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=0)
    memory = _build_memory_from_adapter(adapter, n_episodes=n_episodes, episode_length=episode_length)

    print(f"memory count: {memory.count}")

    # Pick a mid-episode step (not seq_no 0) so there's real trajectory
    # history behind it, rather than a fresh post-reset() state.
    query_id = (n_episodes // 2) * episode_length + episode_length // 2
    query_record = memory.get(query_id)
    results = memory.recall(query_record.z, k=k)

    print(
        f"\nquery id={query_id} (episode={query_record.episode_id}, "
        f"seq_no={query_record.seq_no})"
    )
    print("top results:")
    for id_, score in results:
        record = memory.get(id_)
        print(f"  id={id_} score={score:.4f} episode={record.episode_id} seq_no={record.seq_no}")

    assert results[0][0] == query_id
    print(f"\nself-retrieval OK: top hit is id={results[0][0]} with score={results[0][1]:.4f}")

    # Brute-force memory on the same trajectories, as ground truth.
    reference_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=0)
    brute_memory = _build_memory_from_adapter(
        reference_adapter,
        n_episodes=n_episodes,
        episode_length=episode_length,
        index=FlatIndex(dim=dim),
    )

    memory._index.nprobe = nprobe
    queries = np.stack([memory.get(i).z for i in range(0, memory.count, 20)])
    result = evaluate(memory._index, brute_memory._index, queries, k=k)
    print(f"\nIVF recall@{k} (nprobe={nprobe}): {result['mean_recall']:.4f}")


if __name__ == "__main__":
    main()
