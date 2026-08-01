import numpy as np

from mavka.store import normalize


def make_key(z, action, scale: float, action_dim: int | None) -> np.ndarray:
    """Build a retrieval key from a latent and (optionally) an action.

    key = normalize([z ; scale * action]), length dim + action_dim. If
    action or action_dim is None, falls back to normalize(z) alone
    (appearance-only). scale = 0 also reproduces appearance-only ranking:
    the action part becomes all zeros, which does not change the direction
    of the z part and contributes nothing to a dot product against another
    zero-padded key.
    """
    z = np.asarray(z, dtype=np.float32)
    if action is None or action_dim is None:
        return normalize(z)

    action = np.asarray(action, dtype=np.float32)
    key = np.concatenate([z, (scale * action).astype(np.float32)])
    return normalize(key)


def make_keys_batch(zs, actions, scale: float, action_dim: int | None) -> np.ndarray:
    zs = np.asarray(zs, dtype=np.float32)
    if actions is None or action_dim is None:
        return normalize(zs)

    actions = np.asarray(actions, dtype=np.float32)
    keys = np.concatenate([zs, (scale * actions).astype(np.float32)], axis=1)
    return normalize(keys)
