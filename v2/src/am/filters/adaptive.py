"""The One Euro family: first-order low-pass filters with a scheduled cutoff.

What the 1 euro filter actually is
----------------------------------
An exponential moving average whose cutoff frequency is a function of the
estimated signal speed:

    f_c(k) = f_min + beta * |dx_hat(k)|

Slow movement gives a low cutoff and heavy smoothing; fast movement raises the
cutoff and lets the signal through. That is the whole idea, and it is a
*first-order* filter, which means a stopband slope of only -6 dB/octave.

The consequence, which the v1 evaluation ran into without diagnosing it: the
filter has very limited noise rejection by construction. It was never designed
to be a good noise rejector -- it was designed to be cheap, stable, and tunable
with two interpretable parameters. Casiez et al. make exactly that argument.

Two derived variants address the two separate weaknesses:

``CascadedOneEuro``
    Two stages in series give -12 dB/octave. Tests whether the v1 complaint
    ("barely removes noise") was about filter *order* rather than about the
    scheduling idea.

``JerkOneEuro``
    Schedules on acceleration as well as speed. Plain 1 euro cannot distinguish
    "starting a flick" from "already moving at constant speed", because both
    have large |dx|. Adding the second derivative lets the cutoff open at the
    *onset* of a movement rather than after it.
"""

from __future__ import annotations

import math

from am.filters.base import ScalarFilter


def _smoothing_factor(dt: float, cutoff_hz: float) -> float:
    """Exponential smoothing coefficient for a given cutoff and time step.

    ``alpha = r / (r + 1)`` with ``r = 2*pi*f_c*dt``. This is the standard
    discretisation of a first-order low-pass.
    """
    r = 2.0 * math.pi * cutoff_hz * dt
    return r / (r + 1.0)


class OneEuroFilter(ScalarFilter):
    """Casiez, Roussel and Vogel (2012), 1 euro filter.

    Parameters
    ----------
    min_cutoff:
        Cutoff at zero speed, Hz. Lower means steadier when still, more lag.
    beta:
        Speed coefficient, Hz per (unit/second). Higher means less lag when
        moving fast, more jitter passed through.
    d_cutoff:
        Cutoff of the low-pass applied to the derivative estimate, Hz.

    Notes
    -----
    ``min_cutoff`` must be interpreted relative to the sample rate. At 33 Hz a
    1 Hz cutoff has almost no effect over a single step; at 200 Hz the same
    setting is meaningful. This is why the acquisition fix had to come first.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
        *,
        name: str | None = None,
    ) -> None:
        if min_cutoff <= 0:
            raise ValueError("min_cutoff must be positive")
        if d_cutoff <= 0:
            raise ValueError("d_cutoff must be positive")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.name = name or f"F3-1euro(fc={min_cutoff:g},b={beta:g})"
        self.reset()

    def reset(self) -> None:
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        #: Last cutoff used, exposed so the evaluation can report how much of
        #: the adaptive range was actually exercised during a trial.
        self.last_cutoff: float = self.min_cutoff

    def update(self, x: float, dt: float) -> float:
        if dt <= 0:
            return x if self._x_prev is None else self._x_prev
        if self._x_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            return x

        a_d = _smoothing_factor(dt, self.d_cutoff)
        dx = (x - self._x_prev) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        self.last_cutoff = cutoff

        a = _smoothing_factor(dt, cutoff)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


class CascadedOneEuro(ScalarFilter):
    """Two 1 euro stages in series: -12 dB/octave instead of -6.

    The second stage uses a proportionally higher cutoff so that the combined
    -3 dB point stays close to the single-stage equivalent, isolating the effect
    of filter order from the effect of overall bandwidth.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
        *,
        stage2_scale: float = 1.554,  # keeps the combined -3 dB point
        name: str | None = None,
    ) -> None:
        self.stage1 = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.stage2 = OneEuroFilter(min_cutoff * stage2_scale, beta, d_cutoff)
        self.name = name or f"F4-1euro2x(fc={min_cutoff:g},b={beta:g})"

    def reset(self) -> None:
        self.stage1.reset()
        self.stage2.reset()

    def update(self, x: float, dt: float) -> float:
        return self.stage2.update(self.stage1.update(x, dt), dt)


class JerkOneEuro(ScalarFilter):
    """1 euro with an additional acceleration (jerk) term in the schedule.

        f_c = f_min + beta * |dx| + gamma * |ddx|

    Rationale: the plain speed schedule reacts *after* the hand is already
    moving. Acceleration is large at movement onset, when the signal is still
    slow, so including it opens the cutoff at the start of a ballistic phase
    rather than partway through it. This targets exactly the transient lag that
    the 75 ms perceptual threshold is about.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        gamma: float = 0.0,
        d_cutoff: float = 1.0,
        dd_cutoff: float = 1.0,
        *,
        name: str | None = None,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.gamma = gamma
        self.d_cutoff = d_cutoff
        self.dd_cutoff = dd_cutoff
        self.name = name or f"F5-1euroJerk(fc={min_cutoff:g},b={beta:g},g={gamma:g})"
        self.reset()

    def reset(self) -> None:
        self._x_prev: float | None = None
        self._dx_prev = 0.0
        self._ddx_prev = 0.0
        self.last_cutoff = self.min_cutoff

    def update(self, x: float, dt: float) -> float:
        if dt <= 0:
            return x if self._x_prev is None else self._x_prev
        if self._x_prev is None:
            self._x_prev = x
            return x

        a_d = _smoothing_factor(dt, self.d_cutoff)
        dx = (x - self._x_prev) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        a_dd = _smoothing_factor(dt, self.dd_cutoff)
        ddx = (dx_hat - self._dx_prev) / dt
        ddx_hat = a_dd * ddx + (1.0 - a_dd) * self._ddx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat) + self.gamma * abs(ddx_hat)
        self.last_cutoff = cutoff

        a = _smoothing_factor(dt, cutoff)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._ddx_prev = ddx_hat
        return x_hat


class DoubleExponentialSmoothing(ScalarFilter):
    """Holt double exponential smoothing with prediction (LaViola, 2003).

    Maintains a level and a trend, then extrapolates ``predict_steps`` ahead.
    Because it extrapolates, it can exhibit **negative** effective latency: the
    output can lead the input during steady motion.

    This is the standard competitor to the 1 euro filter in the HCI filtering
    literature and its absence from the v1 comparison was a gap. The trade-off
    is overshoot at direction reversals, which the evaluation must measure
    rather than assume away.

    Parameters
    ----------
    alpha:
        Level smoothing, 0..1. Higher tracks faster.
    gamma:
        Trend smoothing, 0..1.
    predict_steps:
        How many samples ahead to extrapolate. 0 disables prediction and gives
        plain Holt smoothing.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        gamma: float = 0.1,
        predict_steps: float = 1.0,
        *,
        name: str | None = None,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        self.alpha = alpha
        self.gamma = gamma
        self.predict_steps = predict_steps
        self.name = name or f"F6-DES(a={alpha:g},g={gamma:g},p={predict_steps:g})"
        self.reset()

    def reset(self) -> None:
        self._level: float | None = None
        self._trend: float = 0.0

    def update(self, x: float, dt: float) -> float:
        if self._level is None:
            self._level = x
            self._trend = 0.0
            return x

        prev_level = self._level
        self._level = self.alpha * x + (1.0 - self.alpha) * (prev_level + self._trend)
        self._trend = self.gamma * (self._level - prev_level) + (
            1.0 - self.gamma
        ) * self._trend
        return self._level + self.predict_steps * self._trend
