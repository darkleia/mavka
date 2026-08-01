import numpy as np


class SurpriseTrigger:
    """Decides, per step, whether the model's recent prediction error is
    unusually high ("surprising") and retrieval is worth attempting.

    Maintains an exponential moving average (EMA) of recent prediction
    error's mean and variance, so it adapts to the model's typical error
    level over time without storing full history:

        delta = error - mean
        mean  = mean + smoothing * delta
        var   = (1 - smoothing) * (var + smoothing * delta**2)

    should_retrieve fires when error > ema_mean + lam * ema_std -- the
    current error is more than `lam` standard deviations above how the
    model has typically been doing lately.

    Warmup: before `warmup` calls to update(), there isn't enough history
    for the EMA to be a meaningful baseline, so should_retrieve (and
    should_retrieve_causal) always return True during warmup -- documented
    default: prefer retrieving when uncertain over silently trusting an
    unset baseline.

    Two modes, because of a real timing problem: you don't know a step's
    true outcome (and therefore its true error) until after you would
    already have had to decide whether to retrieve.

    - should_retrieve(error): OFFLINE/ORACLE mode. Takes the step's actual
      prediction error and compares it to the threshold. This "peeks" at
      information a real deployment wouldn't have yet -- valid only for
      *measuring* how well gating would work in hindsight, never for
      making a real retrieval decision live.
    - should_retrieve_causal(): CAUSAL/REAL-TIME mode, the honest one for
      actual deployment. Takes no current-step error at all -- only the
      running EMA state and the *previous* step's error (a crude "was I
      surprised last step? then stay cautious this step too" proxy).
      Finding a good real before-the-fact surprise signal (e.g. a
      predicted-uncertainty output from the model itself, or an
      embedding-novelty score) is an open design question and explicitly
      not built here -- this method is the clean hook a better signal
      would plug into, not a finished solution.
    """

    def __init__(self, smoothing: float = 0.1, lam: float = 1.5, warmup: int = 10):
        self.smoothing = smoothing
        self.lam = lam
        self.warmup = warmup

        self._mean = 0.0
        self._var = 0.0
        self._count = 0
        self._last_error: float | None = None

    @property
    def ema_mean(self) -> float:
        return self._mean

    @property
    def ema_std(self) -> float:
        return float(np.sqrt(self._var))

    @property
    def count(self) -> int:
        return self._count

    def update(self, error: float) -> None:
        error = float(error)
        if self._count == 0:
            self._mean = error
            self._var = 0.0
        else:
            delta = error - self._mean
            self._mean = self._mean + self.smoothing * delta
            self._var = (1 - self.smoothing) * (self._var + self.smoothing * delta**2)
        self._last_error = error
        self._count += 1

    def should_retrieve(self, error: float) -> bool:
        # count == 0 is a separate guard from warmup itself: with zero prior
        # observations, mean/std are still their uninitialized 0.0 defaults,
        # so comparing against them would be meaningless (any positive error
        # would trivially "exceed" a mean of 0) -- this holds even when
        # warmup is configured to 0.
        if self._count == 0 or self._count < self.warmup:
            return True
        return float(error) > self._mean + self.lam * self.ema_std

    def should_retrieve_causal(self) -> bool:
        if self._count == 0 or self._count < self.warmup or self._last_error is None:
            return True
        return self._last_error > self._mean + self.lam * self.ema_std
