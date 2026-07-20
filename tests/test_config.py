import dataclasses

import pytest

from mavka import MavkaConfig


def test_valid_config_holds_values():
    config = MavkaConfig(dim=128, action_dim=4, k=16, latency_budget_ms=5.0)
    assert config.dim == 128
    assert config.action_dim == 4
    assert config.k == 16
    assert config.latency_budget_ms == 5.0


def test_defaults():
    config = MavkaConfig(dim=128)
    assert config.k == 8
    assert config.action_dim is None
    assert config.latency_budget_ms == 2.0


@pytest.mark.parametrize("dim", [0, -1, 1.5, "128", None])
def test_invalid_dim_raises(dim):
    with pytest.raises(ValueError):
        MavkaConfig(dim=dim)


@pytest.mark.parametrize("action_dim", [0, -1, 1.5, "4"])
def test_invalid_action_dim_raises(action_dim):
    with pytest.raises(ValueError):
        MavkaConfig(dim=128, action_dim=action_dim)


@pytest.mark.parametrize("k", [0, -1, 1.5, "8"])
def test_invalid_k_raises(k):
    with pytest.raises(ValueError):
        MavkaConfig(dim=128, k=k)


@pytest.mark.parametrize("latency_budget_ms", [0, -1.0, "2.0"])
def test_invalid_latency_budget_ms_raises(latency_budget_ms):
    with pytest.raises(ValueError):
        MavkaConfig(dim=128, latency_budget_ms=latency_budget_ms)


def test_immutable():
    config = MavkaConfig(dim=128)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.dim = 64
