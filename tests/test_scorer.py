import numpy as np

from mavka.action_conditioning import evaluate_with_retrieval
from mavka.adapter import SyntheticWorldModel, generate_trajectory
from mavka.eval.baseline import split_episodes
from mavka.retrieval.fusion import ConcatFusionPredictor
from mavka.storage.log import AppendLog
from mavka.pipeline import ActionConditionedPipeline
from mavka.retrieval.scorer import FixedWeightScorer
from mavka.index.flat import FlatIndex


def _rand(dim, seed):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def test_pure_similarity_weights_matches_index_order():
    log = AppendLog(dim=4, action_dim=2)
    ids = [log.append(z=_rand(4, i), action=_rand(2, i + 10)) for i in range(5)]

    candidates = [(ids[0], 0.9), (ids[1], 0.5), (ids[2], 0.95), (ids[3], 0.1), (ids[4], 0.7)]
    scorer = FixedWeightScorer(log, w_sim=1.0, w_action=0.0, w_recency=0.0)
    result = scorer.score(candidates, query_action=_rand(2, 99))

    expected_order = [id_ for id_, _ in sorted(candidates, key=lambda c: c[1], reverse=True)]
    assert [id_ for id_, _ in result] == expected_order


def test_recency_only_ranks_newer_higher():
    log = AppendLog(dim=4, action_dim=None)
    ids = [log.append(z=_rand(4, i)) for i in range(5)]

    candidates = [(id_, 0.5) for id_ in ids]  # equal similarity
    scorer = FixedWeightScorer(log, w_sim=0.0, w_action=0.0, w_recency=1.0)
    result = scorer.score(candidates, query_action=None)

    assert [id_ for id_, _ in result] == list(reversed(ids))


def test_action_only_ranks_closer_action_higher():
    log = AppendLog(dim=4, action_dim=2)
    z = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    id_close = log.append(z=z, action=np.array([1.0, 0.0], dtype=np.float32))
    id_far = log.append(z=z, action=np.array([-1.0, 0.0], dtype=np.float32))

    candidates = [(id_close, 0.5), (id_far, 0.5)]  # equal similarity
    scorer = FixedWeightScorer(log, w_sim=0.0, w_action=1.0, w_recency=0.0)
    result = scorer.score(candidates, query_action=np.array([1.0, 0.0], dtype=np.float32))

    assert result[0][0] == id_close


def test_normalization_prevents_raw_scale_domination():
    log = AppendLog(dim=4, action_dim=None)
    z = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    id_a = log.append(z=z)
    id_b = log.append(z=z)
    id_c = log.append(z=z)

    # sim strongly favors id_a; recency (id) strongly favors id_c; id_b is a
    # close second on sim and in the middle on recency. Without min-max
    # normalization, id_a's huge raw sim score would dominate the sum and
    # win outright; with normalization applied, id_b wins instead.
    candidates = [(id_a, 1000.0), (id_b, 990.0), (id_c, 1.0)]
    scorer = FixedWeightScorer(log, w_sim=1.0, w_action=0.0, w_recency=1.0)
    result = scorer.score(candidates, query_action=None)

    assert result[0][0] == id_b


def test_missing_actions_skips_action_signal():
    log = AppendLog(dim=4, action_dim=None)
    ids = [log.append(z=_rand(4, i)) for i in range(3)]

    candidates = [(ids[0], 0.9), (ids[1], 0.5), (ids[2], 0.1)]
    scorer = FixedWeightScorer(log, w_sim=1.0, w_action=1.0, w_recency=0.0)
    result = scorer.score(candidates, query_action=None)

    assert [id_ for id_, _ in result] == [ids[0], ids[1], ids[2]]


def test_empty_candidates_returns_empty():
    log = AppendLog(dim=4, action_dim=None)
    scorer = FixedWeightScorer(log)
    assert scorer.score([], None) == []


def test_single_candidate_returned_as_is():
    log = AppendLog(dim=4, action_dim=None)
    id_ = log.append(z=_rand(4, 0))
    scorer = FixedWeightScorer(log)
    result = scorer.score([(id_, 0.7)], None)
    assert result == [(id_, 0.7)]


def test_determinism():
    log = AppendLog(dim=4, action_dim=2)
    ids = [log.append(z=_rand(4, i), action=_rand(2, i + 50)) for i in range(6)]
    candidates = [(id_, float(np.random.default_rng(id_ + 100).random())) for id_ in ids]

    scorer = FixedWeightScorer(log, w_sim=1.0, w_action=0.5, w_recency=0.2)
    query_action = np.array([1.0, 0.0], dtype=np.float32)
    result_a = scorer.score(candidates, query_action)
    result_b = scorer.score(candidates, query_action)

    assert result_a == result_b


def test_recall_scored_returns_k_results():
    dim = 8
    action_dim = 2
    pipeline = ActionConditionedPipeline(
        dim=dim, action_dim=action_dim, index=FlatIndex(dim=dim + action_dim)
    )
    for i in range(20):
        pipeline.observe(z=_rand(dim, i), action=_rand(action_dim, i + 100), z_next=None)

    scorer = FixedWeightScorer(pipeline._log)
    results = pipeline.recall_scored(
        _rand(dim, 999), _rand(action_dim, 998), k=5, scorer=scorer, fetch_factor=3
    )
    assert len(results) == 5


def test_recall_scored_end_to_end_with_evaluate_with_retrieval():
    dim = 16
    action_dim = 4
    n_episodes = 8
    episode_length = 10
    k = 5

    gen_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=30)
    all_episodes = [
        generate_trajectory(gen_adapter, episode_length, episode_id=i) for i in range(n_episodes)
    ]
    memory_episodes, eval_episodes = split_episodes(all_episodes, holdout_frac=0.3, seed=30)

    eval_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=30)
    pipeline = ActionConditionedPipeline(
        dim=dim, action_dim=action_dim, index=FlatIndex(dim=dim + action_dim)
    )
    scorer = FixedWeightScorer(pipeline._log)
    predictor = ConcatFusionPredictor(eval_adapter, alpha=1.0)

    result = evaluate_with_retrieval(
        memory_episodes,
        eval_episodes,
        pipeline,
        predictor,
        k=k,
        scale=1.0,
        scorer=scorer,
        fetch_factor=3,
    )

    assert result["n_steps"] == sum(len(ep) for ep in eval_episodes)
    assert result["mean_error"] > 0
