from typing import Protocol, runtime_checkable

import numpy as np

from mavka.core.distance import normalize


@runtime_checkable
class WorldModelAdapter(Protocol):
    """The boundary between Mavka and any world model.

    Mavka depends only on this interface, never on a specific ML framework:
    every method here takes and returns plain NumPy arrays. A real model
    (e.g. a PyTorch V-JEPA/Dreamer-style encoder) implements it by converting
    at its own boundary -- encode() runs the model's encoder and returns
    something like predicted_z.detach().cpu().numpy().astype(np.float32),
    step() runs the model's dynamics/predictor the same way, and
    reset()/sample_action() wrap whatever environment or replay source that
    system uses. Mavka never sees framework tensors, gradients, or devices.
    """

    dim: int
    action_dim: int | None

    def encode(self, observation) -> np.ndarray:
        """Turn a raw observation into a latent z (float32, length dim)."""
        ...

    def step(self, z: np.ndarray, action: np.ndarray | None) -> np.ndarray:
        """Predict the next latent z_next given the current latent and action."""
        ...

    def reset(self):
        """Start a new episode; return a fresh raw observation."""
        ...

    def sample_action(self) -> np.ndarray:
        """Sample a plausible action (float32, length action_dim)."""
        ...


class SyntheticWorldModel:
    """Dependency-free stand-in for a real world model, for development and tests.

    Latents evolve under fixed random linear dynamics, z_next = normalize(A
    @ z + B @ action + noise), with A, B drawn once at construction and never
    changed. Because A and B are fixed, step() is a smooth (Lipschitz, up to
    the noise term) function of (z, action): nearby inputs produce nearby
    outputs. That "similar in, similar out" property is the whole point --
    it is what makes retrieval over this synthetic world meaningful instead
    of hollow.
    """

    _NOISE_SCALE = 0.05

    def __init__(self, dim: int, action_dim: int | None = None, seed: int = 0):
        self.dim = dim
        self.action_dim = action_dim
        self._rng = np.random.default_rng(seed)

        self._A = (self._rng.standard_normal((dim, dim)) / np.sqrt(dim)).astype(np.float32)
        if action_dim is not None:
            self._B = (
                self._rng.standard_normal((dim, action_dim)) / np.sqrt(action_dim)
            ).astype(np.float32)
        else:
            self._B = None

    def reset(self) -> np.ndarray:
        return self._rng.standard_normal(self.dim).astype(np.float32)

    def encode(self, observation) -> np.ndarray:
        return normalize(np.asarray(observation, dtype=np.float32))

    def sample_action(self) -> np.ndarray:
        if self.action_dim is None:
            raise ValueError("this world model has no actions (action_dim is None)")
        return self._rng.standard_normal(self.action_dim).astype(np.float32)

    def step(self, z: np.ndarray, action: np.ndarray | None = None) -> np.ndarray:
        z = np.asarray(z, dtype=np.float32)
        raw = self._A @ z

        if self.action_dim is not None:
            if action is None:
                raise ValueError("action is required when action_dim is set")
            raw = raw + self._B @ np.asarray(action, dtype=np.float32)

        noise = self._rng.standard_normal(self.dim).astype(np.float32) * self._NOISE_SCALE
        return normalize(raw + noise)


def generate_trajectory(adapter, length: int, episode_id: int = 0) -> list[dict]:
    observation = adapter.reset()
    z = adapter.encode(observation)

    steps = []
    for seq_no in range(length):
        action = adapter.sample_action() if adapter.action_dim is not None else None
        z_next = adapter.step(z, action)
        # Placeholder: distance moved this step. A real predictor's actual
        # prediction-vs-outcome error is later work; this just guarantees a
        # real, non-constant number in the field for now.
        pred_err = float(np.linalg.norm(z_next - z))

        steps.append(
            {
                "z": z,
                "action": action,
                "z_next": z_next,
                "pred_err": pred_err,
                "episode_id": episode_id,
                "seq_no": seq_no,
            }
        )

        z = z_next

    return steps


def populate_store(adapter, log_or_index, n_episodes: int, episode_length: int) -> list[int]:
    """Feed n_episodes generated trajectories into an AppendLog or a plain
    vector index (FlatIndex/IVFIndex), assigning episode_id/seq_no along
    the way. An AppendLog (has .append) keeps the full record (z, action,
    pred_err, episode_id); a plain index (has .add) only has room for the
    latent itself, so only z is stored.
    """
    ids = []
    for episode_id in range(n_episodes):
        for step in generate_trajectory(adapter, episode_length, episode_id=episode_id):
            if hasattr(log_or_index, "append"):
                id_ = log_or_index.append(
                    z=step["z"],
                    action=step["action"],
                    pred_err=step["pred_err"],
                    episode_id=step["episode_id"],
                )
            else:
                id_ = log_or_index.add(step["z"])
            ids.append(id_)
    return ids
