import numpy as np

from mavka.config import MavkaConfig
from mavka.core.distance import normalize
from mavka.graph.adjacency import EDGE_ANALOGOUS, EDGE_TEMPORAL, AdjacencyStore
from mavka.graph.builder import EdgeBuilder
from mavka.index.flat import FlatIndex
from mavka.memory import Memory


def _rand(dim, seed):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def _build_memory_with_edges(dim, n_analogous=3, similarity_threshold=0.5, temporal_weight=1.0, degree=6):
    """Memory has no built-in edge-building hook on observe() (unlike old
    Pipeline's graph=/edge_builder= constructor args) -- returns an
    observe() closure that drives EdgeBuilder.on_insert manually after
    each memory.observe() call, so call sites below stay a simple
    observe(...) call, same shape as the old pipeline.observe(...).
    """
    graph = AdjacencyStore(degree=degree)
    builder = EdgeBuilder(
        n_analogous=n_analogous, similarity_threshold=similarity_threshold, temporal_weight=temporal_weight
    )
    config = MavkaConfig(dim=dim)
    memory = Memory(config, index=FlatIndex(dim=dim), action_scale=0.0)

    def observe(z, action, z_next, episode_id):
        record_id = memory.observe(z=z, action=action, z_next=z_next, episode_id=episode_id)
        graph.add_node()  # kept in lockstep with record_id by construction
        record = memory.get(record_id)
        builder.on_insert(
            record_id,
            record.z,
            record.action,
            record.episode_id,
            record.seq_no,
            memory._log,
            memory._index,
            graph,
        )
        return record_id

    return memory, graph, observe


def test_temporal_edges_correct_and_agree_with_next_in_episode():
    dim = 8
    # High threshold suppresses analogous edges so this test is purely about temporal.
    memory, graph, observe = _build_memory_with_edges(dim, similarity_threshold=0.99)

    ids = [observe(z=_rand(dim, i), action=None, z_next=None, episode_id=0) for i in range(5)]

    for i in range(1, 5):
        temporal = dict(graph.neighbors_of_type(ids[i - 1], EDGE_TEMPORAL))
        assert ids[i] in temporal

        next_record = memory._log.next_in_episode(ids[i - 1])
        assert next_record.id == ids[i]
        assert graph.has_edge(ids[i - 1], ids[i])


def test_first_of_episode_has_no_temporal_predecessor():
    dim = 8
    memory, graph, observe = _build_memory_with_edges(dim, similarity_threshold=0.99)

    id0 = observe(z=_rand(dim, 0), action=None, z_next=None, episode_id=0)

    assert graph.degree_of(id0) == 0


def test_analogous_edges_exclude_same_episode():
    dim = 8
    memory, graph, observe = _build_memory_with_edges(dim, n_analogous=3, similarity_threshold=0.5)

    ep0_ids = [observe(z=_rand(dim, i), action=None, z_next=None, episode_id=0) for i in range(3)]

    anchor = normalize(memory.get(ep0_ids[1]).z)
    similar_z = anchor + _rand(dim, 100) * 0.001
    ep1_id = observe(z=similar_z, action=None, z_next=None, episode_id=1)

    analogous = graph.neighbors_of_type(ep1_id, EDGE_ANALOGOUS)
    analogous_ids = [id_ for id_, _ in analogous]

    assert len(analogous_ids) > 0
    for aid in analogous_ids:
        assert memory.get(aid).episode_id != 1
    assert ep0_ids[1] in analogous_ids


def test_analogous_edges_respect_threshold_and_count_limit():
    dim = 8
    memory, graph, observe = _build_memory_with_edges(
        dim, n_analogous=2, similarity_threshold=0.9, degree=10
    )

    z0 = _rand(dim, 0)
    observe(z=z0, action=None, z_next=None, episode_id=0)

    dissimilar_id = observe(z=-z0, action=None, z_next=None, episode_id=1)
    assert graph.neighbors_of_type(dissimilar_id, EDGE_ANALOGOUS) == []

    anchor = normalize(z0)
    for i in range(5):
        noisy = anchor + _rand(dim, 10 + i) * 0.001
        observe(z=noisy, action=None, z_next=None, episode_id=10 + i)

    final_z = anchor + _rand(dim, 999) * 0.001
    final_id = observe(z=final_z, action=None, z_next=None, episode_id=99)

    analogous = graph.neighbors_of_type(final_id, EDGE_ANALOGOUS)
    assert len(analogous) == 2
    for _, score in analogous:
        assert score >= 0.9


def test_edge_types_distinguishable():
    dim = 8
    memory, graph, observe = _build_memory_with_edges(dim, n_analogous=3, similarity_threshold=0.9)

    id1 = observe(z=_rand(dim, 1), action=None, z_next=None, episode_id=0)
    id1b = observe(z=_rand(dim, 2), action=None, z_next=None, episode_id=0)

    similar_to_id1 = normalize(memory.get(id1).z) + _rand(dim, 3) * 0.001
    id2 = observe(z=similar_to_id1, action=None, z_next=None, episode_id=1)

    temporal_from_id1 = [id_ for id_, _ in graph.neighbors_of_type(id1, EDGE_TEMPORAL)]
    analogous_from_id2 = [id_ for id_, _ in graph.neighbors_of_type(id2, EDGE_ANALOGOUS)]

    assert temporal_from_id1 == [id1b]
    assert id1 in analogous_from_id2


def test_no_future_leakage():
    dim = 8
    memory, graph, observe = _build_memory_with_edges(dim, n_analogous=3, similarity_threshold=0.5)

    ids = [observe(z=_rand(dim, i), action=None, z_next=None, episode_id=i) for i in range(20)]

    for id_ in ids:
        for match_id, _ in graph.neighbors_of_type(id_, EDGE_ANALOGOUS):
            assert match_id < id_


def test_fixed_degree_holds_when_flooded_with_analogies():
    dim = 8
    memory, graph, observe = _build_memory_with_edges(dim, n_analogous=5, similarity_threshold=0.5, degree=3)

    z0 = _rand(dim, 0)
    observe(z=z0, action=None, z_next=None, episode_id=0)

    anchor = normalize(z0)
    for i in range(6):
        noise_scale = 0.01 * (6 - i)
        noisy = anchor + _rand(dim, 10 + i) * noise_scale
        observe(z=noisy, action=None, z_next=None, episode_id=10 + i)

    final_z = anchor + _rand(dim, 999) * 0.001
    final_id = observe(z=final_z, action=None, z_next=None, episode_id=99)

    analogous = graph.neighbors_of_type(final_id, EDGE_ANALOGOUS)
    assert len(analogous) == 3


def test_determinism():
    def run():
        memory, graph, observe = _build_memory_with_edges(8, n_analogous=2, similarity_threshold=0.3)
        ids = [
            observe(z=_rand(8, i), action=None, z_next=None, episode_id=i % 3)
            for i in range(15)
        ]
        return [graph.neighbor_weights(id_) for id_ in ids]

    assert run() == run()
