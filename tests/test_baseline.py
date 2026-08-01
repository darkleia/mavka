import numpy as np
import pytest

from mavka.adapter import SyntheticWorldModel, generate_trajectory
from mavka.eval.baseline import evaluate_no_memory, mean_prediction_error, prediction_error, split_episodes
from mavka.index.flat import normalize


def test_prediction_error_zero_when_equal_positive_when_different():
    z = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert prediction_error(z, z) == 0.0

    other = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    assert prediction_error(z, other) > 0.0


def test_mean_prediction_error_averages_correctly():
    predicted = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    true = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    result = mean_prediction_error(predicted, true)

    assert result == pytest.approx(0.25)


def test_evaluate_no_memory_returns_expected_keys_and_shapes():
    dim = 16
    action_dim = 4
    n_episodes = 3
    episode_length = 8

    gen_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=10)
    episodes = [
        generate_trajectory(gen_adapter, episode_length, episode_id=i) for i in range(n_episodes)
    ]

    eval_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=10)
    result = evaluate_no_memory(eval_adapter, episodes)

    assert set(result.keys()) >= {"mean_error", "median_error", "p90_error", "n_steps", "errors"}
    assert result["n_steps"] == n_episodes * episode_length
    assert isinstance(result["mean_error"], float)
    assert result["mean_error"] > 0
    assert len(result["errors"]) == result["n_steps"]


def test_determinism_same_seed_same_episodes_same_baseline():
    dim = 16
    action_dim = 4
    seed = 7

    gen_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=seed)
    episodes = [generate_trajectory(gen_adapter, 10, episode_id=i) for i in range(5)]

    eval_adapter_a = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=seed)
    eval_adapter_b = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=seed)

    result_a = evaluate_no_memory(eval_adapter_a, episodes)
    result_b = evaluate_no_memory(eval_adapter_b, episodes)

    assert result_a["mean_error"] == result_b["mean_error"]
    assert result_a["errors"] == result_b["errors"]


def test_split_episodes_disjoint_and_deterministic():
    episodes = [[{"episode_id": i}] for i in range(20)]

    memory_a, eval_a = split_episodes(episodes, holdout_frac=0.25, seed=1)
    memory_b, eval_b = split_episodes(episodes, holdout_frac=0.25, seed=1)

    assert len(eval_a) == 5
    assert len(memory_a) == 15
    assert len(memory_a) + len(eval_a) == len(episodes)

    memory_ids_a = {step["episode_id"] for ep in memory_a for step in ep}
    eval_ids_a = {step["episode_id"] for ep in eval_a for step in ep}
    assert memory_ids_a.isdisjoint(eval_ids_a)
    assert memory_ids_a | eval_ids_a == set(range(20))

    memory_ids_b = {step["episode_id"] for ep in memory_b for step in ep}
    eval_ids_b = {step["episode_id"] for ep in eval_b for step in ep}
    assert memory_ids_a == memory_ids_b
    assert eval_ids_a == eval_ids_b


class _PerfectPredictor:
    """Test-only stand-in whose .step() always returns the exact true
    z_next, in the same order evaluate_no_memory processes episodes/steps.
    """

    def __init__(self, episodes):
        self._true_next = [step["z_next"] for episode in episodes for step in episode]
        self._i = 0

    def step(self, z, action):
        z_next = self._true_next[self._i]
        self._i += 1
        return z_next


class _RandomPredictor:
    """Test-only stand-in whose .step() ignores its input and returns an
    unrelated random unit vector.
    """

    def __init__(self, dim, seed=0):
        self._rng = np.random.default_rng(seed)
        self._dim = dim

    def step(self, z, action):
        return normalize(self._rng.standard_normal(self._dim).astype(np.float32))


def test_metric_direction_perfect_vs_random_predictor():
    dim = 16
    action_dim = 4
    gen_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=8)
    episodes = [generate_trajectory(gen_adapter, 10, episode_id=i) for i in range(4)]

    perfect_result = evaluate_no_memory(_PerfectPredictor(episodes), episodes)
    assert perfect_result["mean_error"] == pytest.approx(0.0, abs=1e-6)

    random_result = evaluate_no_memory(_RandomPredictor(dim=dim, seed=9), episodes)
    assert random_result["mean_error"] > perfect_result["mean_error"]
    assert random_result["mean_error"] > 0.01
