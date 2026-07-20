from dataclasses import dataclass


@dataclass(frozen=True)
class MavkaConfig:
    dim: int
    action_dim: int | None = None
    k: int = 8
    latency_budget_ms: float = 2.0

    def __post_init__(self):
        if not isinstance(self.dim, int) or self.dim <= 0:
            raise ValueError(f"dim must be a positive int, got {self.dim!r}")
        if self.action_dim is not None and (
            not isinstance(self.action_dim, int) or self.action_dim <= 0
        ):
            raise ValueError(
                f"action_dim must be None or a positive int, got {self.action_dim!r}"
            )
        if not isinstance(self.k, int) or self.k <= 0:
            raise ValueError(f"k must be a positive int, got {self.k!r}")
        if not isinstance(self.latency_budget_ms, (int, float)) or self.latency_budget_ms <= 0:
            raise ValueError(
                f"latency_budget_ms must be a positive number, got {self.latency_budget_ms!r}"
            )
