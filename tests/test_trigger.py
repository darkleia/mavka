import numpy as np
import pytest

from mavka.adapter import SyntheticWorldModel, generate_trajectory
from mavka.config import MavkaConfig
from mavka.eval.baseline import evaluate_no_memory, split_episodes
from mavka.eval.retrieval_eval import evaluate_gated, evaluate_with_retrieval
from mavka.index.flat import FlatIndex
from mavka.memory import Memory
from mavka.retrieval.fusion import ConcatFusionPredictor
from mavka.retrieval.trigger import SurpriseTrigger


def test_flat_errors_never_trigger_after_warmup():
    trigger = SurpriseTrigger(smoothing=0.2, lam=1.5, warmup=10)
    triggers = []
    for _ in range(50):
        e = 0.01
        triggers.append(trigger.should_retrieve(e))
        trigger.update(e)
    assert not any(triggers[10:])


def test_clear_spike_triggers():
    trigger = SurpriseTrigger(smoothing=0.2, lam=1.5, warmup=10)
    rng = np.random.default_rng(0)
    for _ in range(20):
        trigger.update(0.01 + 0.0005 * rng.standard_normal())

    assert trigger.should_retrieve(10.0) is True


def test_lambda_controls_trigger_frequency():
    rng = np.random.default_rng(1)
    errors = list(np.abs(rng.standard_normal(200)) * 0.01 + 0.01)

    def count_triggers(lam):
        trigger = SurpriseTrigger(smoothing=0.1, lam=lam, warmup=10)
        count = 0
        for e in errors:
            if trigger.should_retrieve(e):
                count += 1
            trigger.update(e)
        return count

    assert count_triggers(0.5) > count_triggers(3.0)


def test_ema_rebaselines_after_sustained_shift():
    trigger = SurpriseTrigger(smoothing=0.2, lam=1.5, warmup=10)
    for _ in range(30):
        trigger.update(0.01)

    shifted_triggers = []
    for _ in range(100):
        e = 0.05
        shifted_triggers.append(trigger.should_retrieve(e))
        trigger.update(e)

    assert any(shifted_triggers[:5])
    assert not any(shifted_triggers[-10:])


def test_warmup_always_retrieves():
    trigger = SurpriseTrigger(smoothing=0.1, lam=1.5, warmup=15)
    for _ in range(15):
        assert trigger.should_retrieve(0.01) is True
        trigger.update(0.01)
    assert trigger.count == 15


def test_gated_evaluation_reports_rate_that_moves_with_lambda():
    dim = 16
    action_dim = 4
    n_episodes = 10
    episode_length = 15
    k = 5

    gen_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=60)
    all_episodes = [
        generate_trajectory(gen_adapter, episode_length, episode_id=i) for i in range(n_episodes)
    ]
    memory_episodes, eval_episodes = split_episodes(all_episodes, holdout_frac=0.3, seed=60)

    def run_gated(lam):
        adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=60)
        config = MavkaConfig(dim=dim, action_dim=action_dim)
        memory = Memory(config, index=FlatIndex(dim=dim + action_dim), action_scale=1.0)
        trigger = SurpriseTrigger(smoothing=0.1, lam=lam, warmup=5)
        return evaluate_gated(memory_episodes, eval_episodes, memory, adapter, trigger, k=k)

    low_lambda_result = run_gated(0.1)
    high_lambda_result = run_gated(5.0)

    assert 0.0 <= low_lambda_result["retrieval_rate"] <= 1.0
    assert 0.0 <= high_lambda_result["retrieval_rate"] <= 1.0
    assert low_lambda_result["retrieval_rate"] >= high_lambda_result["retrieval_rate"]
    assert "mean_error" in low_lambda_result


def test_low_lambda_matches_always_retrieve_exactly():
    dim = 16
    action_dim = 4
    n_episodes = 8
    episode_length = 10
    k = 5

    gen_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=61)
    all_episodes = [
        generate_trajectory(gen_adapter, episode_length, episode_id=i) for i in range(n_episodes)
    ]
    memory_episodes, eval_episodes = split_episodes(all_episodes, holdout_frac=0.3, seed=61)

    config = MavkaConfig(dim=dim, action_dim=action_dim)

    gated_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=61)
    gated_memory = Memory(config, index=FlatIndex(dim=dim + action_dim), action_scale=1.0)
    trigger = SurpriseTrigger(smoothing=0.1, lam=-1000.0, warmup=0)
    gated_result = evaluate_gated(memory_episodes, eval_episodes, gated_memory, gated_adapter, trigger, k=k)

    always_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=61)
    always_memory = Memory(config, index=FlatIndex(dim=dim + action_dim), action_scale=1.0)
    always_predictor = ConcatFusionPredictor(always_adapter, alpha=1.0)
    always_result = evaluate_with_retrieval(
        memory_episodes, eval_episodes, always_memory, always_predictor, k=k
    )

    assert gated_result["retrieval_rate"] == 1.0
    assert gated_result["mean_error"] == always_result["mean_error"]
    assert gated_result["errors"] == always_result["errors"]


def test_high_lambda_approximately_matches_never_retrieve():
    dim = 16
    action_dim = 4
    n_episodes = 40
    episode_length = 20
    k = 5

    gen_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=62)
    all_episodes = [
        generate_trajectory(gen_adapter, episode_length, episode_id=i) for i in range(n_episodes)
    ]
    memory_episodes, eval_episodes = split_episodes(all_episodes, holdout_frac=0.3, seed=62)

    config = MavkaConfig(dim=dim, action_dim=action_dim)
    gated_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=62)
    gated_memory = Memory(config, index=FlatIndex(dim=dim + action_dim), action_scale=1.0)
    # A tiny warmup (not 0) is needed even at extreme lambda: the EMA
    # variance is still genuinely 0 after only one sample, so the very next
    # value would trivially "exceed" a std of 0 regardless of lambda -- this
    # is a real cold-start effect warmup exists to guard against, not a
    # lambda problem. A larger eval set dilutes the few unavoidable
    # cold-start triggers' effect on the mean.
    trigger = SurpriseTrigger(smoothing=0.1, lam=1000.0, warmup=2)
    gated_result = evaluate_gated(memory_episodes, eval_episodes, gated_memory, gated_adapter, trigger, k=k)

    baseline_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=62)
    baseline_result = evaluate_no_memory(baseline_adapter, eval_episodes)

    assert gated_result["retrieval_rate"] < 0.02
    assert gated_result["mean_error"] == pytest.approx(baseline_result["mean_error"], rel=0.2)


def test_determinism():
    dim = 8
    action_dim = 2

    def run():
        gen_adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=63)
        all_episodes = [generate_trajectory(gen_adapter, 10, episode_id=i) for i in range(8)]
        memory_episodes, eval_episodes = split_episodes(all_episodes, holdout_frac=0.25, seed=63)

        adapter = SyntheticWorldModel(dim=dim, action_dim=action_dim, seed=63)
        config = MavkaConfig(dim=dim, action_dim=action_dim)
        memory = Memory(config, index=FlatIndex(dim=dim + action_dim), action_scale=1.0)
        trigger = SurpriseTrigger(smoothing=0.1, lam=1.5, warmup=5)
        return evaluate_gated(memory_episodes, eval_episodes, memory, adapter, trigger, k=5)

    result_a = run()
    result_b = run()

    assert result_a["mean_error"] == result_b["mean_error"]
    assert result_a["retrieval_rate"] == result_b["retrieval_rate"]
    assert result_a["errors"] == result_b["errors"]
