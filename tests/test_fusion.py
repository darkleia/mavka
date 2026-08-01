import numpy as np

from mavka.action_conditioning import evaluate_with_retrieval
from mavka.adapter import SyntheticWorldModel, generate_trajectory
from mavka.eval.baseline import evaluate_no_memory, split_episodes
from mavka.retrieval.fusion import ConcatFusionPredictor, build_context
from mavka.storage.log import AppendLog
from mavka.pipeline import ActionConditionedPipeline
from mavka.index.flat import FlatIndex
from mavka.core.distance import normalize


def _rand(dim, seed):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def test_build_context_shapes_and_padding():
    dim = 4
    action_dim = 2
    log = AppendLog(dim=dim, action_dim=action_dim)
    ids = [log.append(z=_rand(dim, i), action=_rand(action_dim, i + 10), episode_id=0) for i in range(3)]

    k = 5
    # Only 2 candidates given, and k=5 requested -> must pad up to 5 rows.
    candidates = [(ids[0], 0.9), (ids[1], 0.5)]
    context = build_context(candidates, log, k)

    assert context["mem_z"].shape == (k, dim)
    assert context["mem_action"].shape == (k, action_dim)
    assert context["mem_outcome"].shape == (k, dim)
    assert context["mem_score"].shape == (k,)
    assert context["mask"].shape == (k,)
    assert context["mask"].dtype == bool

    assert context["mask"][:2].all()
    assert not context["mask"][2:].any()
    np.testing.assert_array_equal(context["mem_z"][2:], np.zeros((k - 2, dim), dtype=np.float32))
    np.testing.assert_array_equal(
        context["mem_outcome"][2:], np.zeros((k - 2, dim), dtype=np.float32)
    )


def test_outcomes_correct_and_last_in_episode_masked():
    dim = 4
    log = AppendLog(dim=dim, action_dim=None)
    id0 = log.append(z=_rand(dim, 0), episode_id=0)
    id1 = log.append(z=_rand(dim, 1), episode_id=0)  # last step of episode 0 -- no outcome

    candidates = [(id0, 0.8), (id1, 0.6)]
    context = build_context(candidates, log, k=2)

    np.testing.assert_array_equal(context["mem_outcome"][0], log.get(id1).z)
    assert context["mask"][0]
    assert not context["mask"][1]
    np.testing.assert_array_equal(context["mem_outcome"][1], np.zeros(dim, dtype=np.float32))


def test_alpha_zero_matches_model_prediction_exactly():
    dim = 8
    action_dim = 3
    z = normalize(_rand(dim, 1))
    action = _rand(action_dim, 2)

    log = AppendLog(dim=dim, action_dim=action_dim)
    mem_id = log.append(z=z, action=action, episode_id=0)
    log.append(z=normalize(_rand(dim, 3)), action=_rand(action_dim, 6), episode_id=0)
    context = build_context([(mem_id, 0.99)], log, k=1)

    predictor = ConcatFusionPredictor(SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=5), alpha=0.0)
    result = predictor.predict(z, action, context)

    expected = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=5).step(z, action)
    np.testing.assert_array_equal(result, expected)


def test_alpha_zero_matches_no_memory_baseline_exactly():
    dim = 16
    action_dim = 4
    n_episodes = 6
    episode_length = 8
    k = 5

    gen_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=40)
    all_episodes = [
        generate_trajectory(gen_adapter, episode_length, episode_id=i) for i in range(n_episodes)
    ]
    memory_episodes, eval_episodes = split_episodes(all_episodes, holdout_frac=0.3, seed=40)

    baseline_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=40)
    baseline_result = evaluate_no_memory(baseline_adapter, eval_episodes)

    fusion_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=40)
    pipeline = ActionConditionedPipeline(
        dim=dim, action_dim=action_dim, index=FlatIndex(dim=dim + action_dim)
    )
    predictor = ConcatFusionPredictor(fusion_adapter, alpha=0.0)
    fusion_result = evaluate_with_retrieval(
        memory_episodes, eval_episodes, pipeline, predictor, k=k, scale=1.0
    )

    assert fusion_result["mean_error"] == baseline_result["mean_error"]
    assert fusion_result["errors"] == baseline_result["errors"]


def test_alpha_one_reproduces_score_weighted_memory_average():
    dim = 4
    log = AppendLog(dim=dim, action_dim=None)
    z_query = normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

    id_a = log.append(z=z_query, episode_id=0)
    outcome_a = normalize(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    log.append(z=outcome_a, episode_id=0)

    id_b = log.append(z=z_query, episode_id=1)
    outcome_b = normalize(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))
    log.append(z=outcome_b, episode_id=1)

    candidates = [(id_a, 0.9), (id_b, 0.3)]
    context = build_context(candidates, log, k=2)

    weights = np.array([0.9, 0.3])
    weights = weights - weights.min() + 1e-8
    weights = weights / weights.sum()
    expected = weights[0] * outcome_a + weights[1] * outcome_b
    expected = expected / np.linalg.norm(expected)

    predictor = ConcatFusionPredictor(SyntheticWorldModel(dim=dim, action_dim=None, seed=1), alpha=1.0)
    result = predictor.predict(z_query, None, context)

    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_intermediate_alpha_blends_strictly_between():
    dim = 8
    action_dim = 2
    seed = 7
    z = normalize(_rand(dim, 10))
    action = _rand(action_dim, 11)

    log = AppendLog(dim=dim, action_dim=action_dim)
    mem_id = log.append(z=z, action=action, episode_id=0)
    outcome = normalize(_rand(dim, 12))
    log.append(z=outcome, action=_rand(action_dim, 13), episode_id=0)
    context = build_context([(mem_id, 0.9)], log, k=1)

    base = np.asarray(
        SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=seed).step(z, action),
        dtype=np.float32,
    )

    alpha = 0.5
    predictor = ConcatFusionPredictor(
        SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=seed), alpha=alpha
    )
    result = predictor.predict(z, action, context)

    expected = normalize(((1 - alpha) * base + alpha * outcome).astype(np.float32))
    np.testing.assert_allclose(result, expected, atol=1e-6)

    assert not np.allclose(result, base, atol=1e-3)
    assert not np.allclose(result, normalize(outcome), atol=1e-3)


def test_determinism():
    dim = 8
    action_dim = 2

    def run():
        adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=15)
        log = AppendLog(dim=dim, action_dim=action_dim)
        mem_id = log.append(z=normalize(_rand(dim, 1)), action=_rand(action_dim, 2), episode_id=0)
        log.append(z=normalize(_rand(dim, 3)), action=_rand(action_dim, 6), episode_id=0)
        context = build_context([(mem_id, 0.8)], log, k=1)
        predictor = ConcatFusionPredictor(adapter, alpha=0.5)
        return predictor.predict(normalize(_rand(dim, 4)), _rand(action_dim, 5), context)

    result_a = run()
    result_b = run()

    np.testing.assert_array_equal(result_a, result_b)


def test_end_to_end_fusion_evaluation():
    dim = 16
    action_dim = 4
    n_episodes = 8
    episode_length = 10
    k = 5

    gen_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=50)
    all_episodes = [
        generate_trajectory(gen_adapter, episode_length, episode_id=i) for i in range(n_episodes)
    ]
    memory_episodes, eval_episodes = split_episodes(all_episodes, holdout_frac=0.3, seed=50)

    eval_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=50)
    pipeline = ActionConditionedPipeline(
        dim=dim, action_dim=action_dim, index=FlatIndex(dim=dim + action_dim)
    )
    predictor = ConcatFusionPredictor(eval_adapter, alpha=0.5)

    result = evaluate_with_retrieval(
        memory_episodes, eval_episodes, pipeline, predictor, k=k, scale=1.0
    )

    assert set(result.keys()) >= {"mean_error", "median_error", "p90_error", "n_steps", "errors"}
    assert result["n_steps"] == sum(len(ep) for ep in eval_episodes)
    assert result["mean_error"] > 0
