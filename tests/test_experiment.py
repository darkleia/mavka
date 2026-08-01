from mavka.adapter import SyntheticWorldModel, generate_trajectory
from mavka.baseline import split_episodes
from mavka.experiment import CONDITION_NAMES, memory_helps, run_memory_experiment


def test_experiment_runs_end_to_end_with_expected_structure():
    adapter = SyntheticWorldModel(dim=8, action_dim=2, seed=0)
    seeds = [0, 1]
    results = run_memory_experiment(adapter, n_episodes=10, episode_length=6, k=3, seeds=seeds)

    assert set(results["conditions"].keys()) == set(CONDITION_NAMES)
    for stats in results["conditions"].values():
        assert set(stats.keys()) >= {
            "mean_error",
            "std_error",
            "n_steps",
            "relative_improvement_pct",
            "per_seed_errors",
        }
        assert len(stats["per_seed_errors"]) == len(seeds)
        assert stats["mean_error"] >= 0
    assert results["baseline_condition"] == "no_memory"
    assert results["seeds"] == seeds


def test_same_eval_set_across_conditions():
    adapter = SyntheticWorldModel(dim=8, action_dim=2, seed=0)
    results = run_memory_experiment(adapter, n_episodes=10, episode_length=6, k=3, seeds=[0, 1])

    n_steps_values = {stats["n_steps"] for stats in results["conditions"].values()}
    assert len(n_steps_values) == 1


def test_leakage_guard_holds_per_seed():
    n_episodes = 20
    holdout_frac = 0.25
    episode_length = 5

    for seed in (1, 2, 3):
        adapter = SyntheticWorldModel(dim=8, action_dim=2, seed=seed)
        all_episodes = [
            generate_trajectory(adapter, episode_length, episode_id=i) for i in range(n_episodes)
        ]
        memory_episodes, eval_episodes = split_episodes(
            all_episodes, holdout_frac=holdout_frac, seed=seed
        )

        memory_ep_ids = {step["episode_id"] for ep in memory_episodes for step in ep}
        eval_ep_ids = {step["episode_id"] for ep in eval_episodes for step in ep}
        assert memory_ep_ids.isdisjoint(eval_ep_ids)
        assert memory_ep_ids | eval_ep_ids == set(range(n_episodes))


def test_determinism_same_seeds_identical_results():
    def run():
        adapter = SyntheticWorldModel(dim=8, action_dim=2, seed=0)
        return run_memory_experiment(adapter, n_episodes=10, episode_length=6, k=3, seeds=[0, 1])

    result_a = run()
    result_b = run()

    for name in CONDITION_NAMES:
        assert (
            result_a["conditions"][name]["mean_error"] == result_b["conditions"][name]["mean_error"]
        )
        assert (
            result_a["conditions"][name]["per_seed_errors"]
            == result_b["conditions"][name]["per_seed_errors"]
        )


def test_appearance_only_at_alpha_zero_matches_no_memory_exactly():
    adapter = SyntheticWorldModel(dim=8, action_dim=2, seed=0)
    results = run_memory_experiment(
        adapter,
        n_episodes=10,
        episode_length=6,
        k=3,
        seeds=[0, 1],
        config={"fusion_alpha": 0.0},
    )

    baseline_errors = results["conditions"]["no_memory"]["per_seed_errors"]
    appearance_errors = results["conditions"]["appearance_only"]["per_seed_errors"]
    assert appearance_errors == baseline_errors


def test_memory_helps_gates_on_significance():
    no_improvement = {
        "baseline_condition": "no_memory",
        "conditions": {
            "no_memory": {"mean_error": 0.001, "std_error": 0.0001},
            "full_system": {"mean_error": 0.001, "std_error": 0.0001},
        },
    }
    assert memory_helps(no_improvement) is False

    noisy = {
        "baseline_condition": "no_memory",
        "conditions": {
            "no_memory": {"mean_error": 0.001, "std_error": 0.0005},
            "full_system": {"mean_error": 0.0009, "std_error": 0.0005},
        },
    }
    assert memory_helps(noisy) is False

    clear = {
        "baseline_condition": "no_memory",
        "conditions": {
            "no_memory": {"mean_error": 0.010, "std_error": 0.0001},
            "full_system": {"mean_error": 0.002, "std_error": 0.0001},
        },
    }
    assert memory_helps(clear) is True
    assert memory_helps(clear, require_significance=False) is True


def test_full_system_not_worse_than_baseline_on_synthetic_world():
    adapter = SyntheticWorldModel(dim=16, action_dim=4, seed=0)
    results = run_memory_experiment(adapter, n_episodes=30, episode_length=20, k=5, seeds=[0, 1, 2])

    baseline_mean = results["conditions"]["no_memory"]["mean_error"]
    full_mean = results["conditions"]["full_system"]["mean_error"]
    assert full_mean <= baseline_mean
