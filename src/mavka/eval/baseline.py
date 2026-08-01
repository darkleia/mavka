import numpy as np


def prediction_error(predicted_z, true_z) -> float:
    predicted_z = np.asarray(predicted_z, dtype=np.float32)
    true_z = np.asarray(true_z, dtype=np.float32)
    return float(np.mean((predicted_z - true_z) ** 2))


def mean_prediction_error(predicted, true) -> float:
    predicted = np.asarray(predicted, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
    return float(np.mean((predicted - true) ** 2))


def evaluate_no_memory(adapter, episodes) -> dict:
    """Measure how well adapter.step() alone predicts z_next on held-out
    episodes, with no retrieval involved at all. For every step, asks the
    adapter to predict z_next fresh from (z, action) and compares that
    prediction against the z_next already recorded in the episode.
    """
    errors = []
    for episode in episodes:
        for step in episode:
            predicted_z_next = adapter.step(step["z"], step["action"])
            errors.append(prediction_error(predicted_z_next, step["z_next"]))

    errors_arr = np.array(errors, dtype=np.float64)

    return {
        "mean_error": float(np.mean(errors_arr)),
        "median_error": float(np.median(errors_arr)),
        "p90_error": float(np.percentile(errors_arr, 90)),
        "n_steps": len(errors),
        "errors": errors_arr.tolist(),
    }


def split_episodes(all_episodes, holdout_frac: float = 0.2, seed: int = 0):
    """Split a list of episodes into a memory set and a disjoint eval set,
    so that a future with-memory evaluation never retrieves from the exact
    episodes it is being tested on.
    """
    n = len(all_episodes)
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(n)

    n_eval = int(round(n * holdout_frac))
    eval_indices = set(shuffled_indices[:n_eval].tolist())

    memory_episodes = [ep for i, ep in enumerate(all_episodes) if i not in eval_indices]
    eval_episodes = [ep for i, ep in enumerate(all_episodes) if i in eval_indices]

    return memory_episodes, eval_episodes
