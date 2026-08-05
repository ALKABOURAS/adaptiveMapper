"""Uniform interface for every causal filter in the study.

Every filter is a scalar, causal, streaming estimator with the same signature.
That uniformity is what makes them interchangeable *experimental conditions*
rather than bespoke code paths -- the evaluation harness can iterate over the
whole set without special cases, which is the only way to compare eleven
algorithms honestly.

Two rules that the v1 implementation violated and that matter for correctness:

1. ``dt`` is passed in explicitly. A filter must never call ``time.time()``
   itself. Otherwise the same filter cannot be run on recorded data, and every
   evaluation is contaminated by whatever the host was doing at the time.
2. ``reset()`` must fully restore the initial state, so that a filter can be
   re-run across trials without leaking state between experimental conditions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ScalarFilter(ABC):
    """A causal single-channel filter."""

    #: Short identifier used in result tables and plots.
    name: str = "unnamed"

    @abstractmethod
    def reset(self) -> None:
        """Restore the filter to its initial state."""

    @abstractmethod
    def update(self, x: float, dt: float) -> float:
        """Consume one sample and return the current estimate.

        Parameters
        ----------
        x:
            Measurement.
        dt:
            Seconds since the previous sample. Implementations must tolerate a
            non-uniform ``dt``; on real hardware it is never exactly constant.
        """

    def process(self, values, dt) -> np.ndarray:
        """Run the filter over a whole sequence. Resets first.

        ``dt`` may be a scalar or a per-sample array.
        """
        values = np.asarray(values, dtype=np.float64)
        if np.isscalar(dt):
            dts = np.full(values.shape, float(dt))
        else:
            dts = np.asarray(dt, dtype=np.float64)
            if dts.shape != values.shape:
                raise ValueError("dt array must match values shape")
        self.reset()
        out = np.empty_like(values)
        for i, (x, d) in enumerate(zip(values, dts)):
            out[i] = self.update(float(x), float(d))
        return out

    def describe(self) -> str:
        return self.name


class Passthrough(ScalarFilter):
    """Identity filter -- the floor condition F0."""

    name = "F0-passthrough"

    def reset(self) -> None:
        return None

    def update(self, x: float, dt: float) -> float:
        return x


class FilterChain(ScalarFilter):
    """Compose filters in series, applied left to right.

    Used to build the proposed pipeline (nonlinear outlier rejection followed by
    a linear stage) and to build the cascaded variants.
    """

    def __init__(self, *stages: ScalarFilter, name: str | None = None) -> None:
        if not stages:
            raise ValueError("FilterChain requires at least one stage")
        self.stages = list(stages)
        self.name = name or " -> ".join(s.name for s in stages)

    def reset(self) -> None:
        for stage in self.stages:
            stage.reset()

    def update(self, x: float, dt: float) -> float:
        for stage in self.stages:
            x = stage.update(x, dt)
        return x

    def describe(self) -> str:
        return " -> ".join(s.describe() for s in self.stages)


class VectorFilter:
    """Apply an independent copy of a scalar filter to each channel.

    Deliberately *not* a shared instance: the three gyro axes have independent
    noise and must not share adaptation state.
    """

    def __init__(self, factory, n_channels: int = 3, name: str | None = None) -> None:
        self.channels = [factory() for _ in range(n_channels)]
        self.name = name or self.channels[0].name

    def reset(self) -> None:
        for c in self.channels:
            c.reset()

    def update(self, x, dt: float) -> np.ndarray:
        return np.array(
            [c.update(float(v), dt) for c, v in zip(self.channels, x)],
            dtype=np.float64,
        )

    def process(self, values, dt) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("VectorFilter.process expects shape (N, channels)")
        out = np.empty_like(values)
        for ch, filt in enumerate(self.channels):
            out[:, ch] = filt.process(values[:, ch], dt)
        return out
