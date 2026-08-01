import numpy as np
import pytest

from mavka.action_conditioning import evaluate_with_retrieval
from mavka.adapter import SyntheticWorldModel, generate_trajectory
from mavka.eval.baseline import split_episodes
from mavka.retrieval.fusion import ConcatFusionPredictor
from mavka.pipeline import ActionConditionedPipeline
from mavka.index.flat import VectorStore


def test_action_conditioned_retrieval_distinguishes_matching_action():
    dim = 4
    action_dim = 2

    z_common = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    z_common_near = np.array([0.999, 0.001, 0.0, 0.0], dtype=np.float32)
    action_a = np.array([1.0, 0.0], dtype=np.float32)
    action_b = np.array([-1.0, 0.0], dtype=np.float32)

    def build_pipeline(scale):
        pipeline = ActionConditionedPipeline(
            dim=dim, action_dim=action_dim, index=VectorStore(dim=dim + action_dim), scale=scale
        )
        matching_id = pipeline.observe(z=z_common, action=action_a, z_next=None, episode_id=0)
        mismatched_id = pipeline.observe(
            z=z_common_near, action=action_b, z_next=None, episode_id=1
        )
        return pipeline, matching_id, mismatched_id

    action_pipeline, matching_id, mismatched_id = build_pipeline(scale=1.0)
    results = action_pipeline.recall(z_common, action_a, k=2)
    ranked_ids = [id_ for id_, _ in results]
    assert ranked_ids.index(matching_id) < ranked_ids.index(mismatched_id)

    scores = dict(results)
    assert scores[matching_id] > scores[mismatched_id]

    appearance_pipeline, matching_id0, mismatched_id0 = build_pipeline(scale=0.0)
    results0 = appearance_pipeline.recall(z_common, action_a, k=2)
    scores0 = dict(results0)
    assert scores0[matching_id0] == pytest.approx(scores0[mismatched_id0], abs=1e-3)


def test_evaluate_with_retrieval_returns_comparable_stats():
    dim = 16
    action_dim = 4
    n_episodes = 10
    episode_length = 15
    k = 5

    gen_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=20)
    all_episodes = [
        generate_trajectory(gen_adapter, episode_length, episode_id=i) for i in range(n_episodes)
    ]
    memory_episodes, eval_episodes = split_episodes(all_episodes, holdout_frac=0.3, seed=20)

    eval_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=20)
    pipeline = ActionConditionedPipeline(
        dim=dim, action_dim=action_dim, index=VectorStore(dim=dim + action_dim)
    )

    predictor = ConcatFusionPredictor(eval_adapter, alpha=1.0)
    result = evaluate_with_retrieval(
        memory_episodes, eval_episodes, pipeline, predictor, k=k, scale=1.0
    )

    assert set(result.keys()) >= {"mean_error", "median_error", "p90_error", "n_steps", "errors"}
    expected_n_steps = sum(len(ep) for ep in eval_episodes)
    assert result["n_steps"] == expected_n_steps
    assert result["mean_error"] > 0
    assert len(result["errors"]) == result["n_steps"]


def test_determinism_same_seed_same_result():
    dim = 16
    action_dim = 4
    n_episodes = 8
    episode_length = 12
    k = 5

    def run():
        gen_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=21)
        all_episodes = [
            generate_trajectory(gen_adapter, episode_length, episode_id=i)
            for i in range(n_episodes)
        ]
        memory_episodes, eval_episodes = split_episodes(all_episodes, holdout_frac=0.25, seed=21)

        eval_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=21)
        pipeline = ActionConditionedPipeline(
            dim=dim, action_dim=action_dim, index=VectorStore(dim=dim + action_dim)
        )
        predictor = ConcatFusionPredictor(eval_adapter, alpha=1.0)
        return evaluate_with_retrieval(
            memory_episodes, eval_episodes, pipeline, predictor, k=k, scale=1.0
        )

    result_a = run()
    result_b = run()

    assert result_a["mean_error"] == result_b["mean_error"]
    assert result_a["errors"] == result_b["errors"]
