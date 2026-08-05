"""Nonlinear outlier rejection and classical linear filtering.

Why the Hampel filter comes first in every chain
------------------------------------------------
The v1 recordings contain isolated excursions to about -150 dps against a signal
envelope of roughly +-50 dps. Those are impulsive outliers from transport
corruption, not Gaussian sensor noise.

This distinction is not cosmetic. A linear filter cannot remove an impulse: by
linearity it can only *redistribute* the impulse energy over its own impulse
response, turning a one-sample spike into a smeared bump lasting several tens of
milliseconds. Visually the spike shrinks, which is why it is easy to believe the
filter dealt with it. In cursor terms the spike becomes a visible jump.

Removing impulses requires an order-statistic (nonlinear) stage, and it must run
*before* the linear stage, otherwise the linear stage has already spread the
contamination across neighbouring samples.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from am.filters.base import ScalarFilter

#: Scale factor making the MAD a consistent estimator of the standard deviation
#: for normally distributed data: 1 / Phi^-1(0.75).
MAD_TO_SIGMA = 1.4826


class HampelFilter(ScalarFilter):
    """Sliding-window Hampel identifier: a robust outlier replacer.

    For each sample, compares it against the median of a causal window. If it
    deviates by more than ``n_sigmas`` robust standard deviations (estimated
    from the median absolute deviation), it is replaced by the median.

    Parameters
    ----------
    window:
        Number of past samples considered. At 200 Hz a window of 7 spans 35 ms,
        short enough not to distort voluntary motion below 5 Hz.
    n_sigmas:
        Detection threshold. 3.0 is the conventional choice.

    Notes
    -----
    Causal by construction: only past samples are used, so the filter is usable
    in real time and adds no lookahead latency. It does introduce a delay of
    roughly half a window in its *median* behaviour, which the evaluation
    measures rather than assumes.
    """

    name = "F1-hampel"

    def __init__(self, window: int = 7, n_sigmas: float = 3.0) -> None:
        if window < 3:
            raise ValueError("window must be at least 3")
        self.window = window
        self.n_sigmas = n_sigmas
        self.name = f"F1-hampel(w={window},n={n_sigmas:g})"
        self.reset()

    def reset(self) -> None:
        self._buf: deque[float] = deque(maxlen=self.window)
        #: Count of samples replaced, reported per trial as a data-quality metric.
        self.n_replaced = 0

    def update(self, x: float, dt: float) -> float:
        self._buf.append(x)
        if len(self._buf) < self.window:
            return x

        arr = np.fromiter(self._buf, dtype=np.float64, count=len(self._buf))
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        sigma = MAD_TO_SIGMA * mad

        # A degenerate window (all values identical) gives sigma = 0. Do not
        # treat every subsequent sample as an outlier in that case.
        if sigma <= 0.0:
            return x

        if abs(x - median) > self.n_sigmas * sigma:
            self.n_replaced += 1
            self._buf[-1] = median
            return median
        return x


class MedianFilter(ScalarFilter):
    """Plain causal median filter. Cheaper than Hampel, blunter."""

    def __init__(self, window: int = 5) -> None:
        if window < 3 or window % 2 == 0:
            raise ValueError("window must be odd and at least 3")
        self.window = window
        self.name = f"median(w={window})"
        self.reset()

    def reset(self) -> None:
        self._buf: deque[float] = deque(maxlen=self.window)

    def update(self, x: float, dt: float) -> float:
        self._buf.append(x)
        return float(np.median(np.fromiter(self._buf, dtype=np.float64)))


class BiquadLowpass(ScalarFilter):
    """Second-order Butterworth low-pass as a direct-form-II transposed biquad.

    Included as the reference linear filter: fixed cutoff, -12 dB/octave,
    completely characterised magnitude and phase response. It is the honest
    comparison point for "what does a textbook low-pass achieve here", which the
    v1 study never established before moving to adaptive filtering.

    Coefficients are recomputed whenever ``dt`` changes materially, so the filter
    stays correct on a non-uniform stream rather than silently detuning.
    """

    def __init__(self, cutoff_hz: float, rate_hz: float = 200.0, q: float | None = None):
        if cutoff_hz <= 0:
            raise ValueError("cutoff_hz must be positive")
        self.cutoff_hz = cutoff_hz
        self.rate_hz = rate_hz
        #: Butterworth (maximally flat) has Q = 1/sqrt(2).
        self.q = q if q is not None else 1.0 / math.sqrt(2.0)
        self.name = f"F2-butter2(fc={cutoff_hz:g})"
        self._design(rate_hz)
        self.reset()

    def _design(self, rate_hz: float) -> None:
        nyquist = rate_hz / 2.0
        fc = min(self.cutoff_hz, 0.99 * nyquist)
        w0 = 2.0 * math.pi * fc / rate_hz
        cos_w0 = math.cos(w0)
        alpha = math.sin(w0) / (2.0 * self.q)

        b0 = (1.0 - cos_w0) / 2.0
        b1 = 1.0 - cos_w0
        b2 = (1.0 - cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

        self._b = (b0 / a0, b1 / a0, b2 / a0)
        self._a = (a1 / a0, a2 / a0)
        self._designed_rate = rate_hz

    def reset(self) -> None:
        self._z1 = 0.0
        self._z2 = 0.0
        self._primed = False

    def update(self, x: float, dt: float) -> float:
        if dt > 0:
            rate = 1.0 / dt
            # Redesign only on a material change, to avoid recomputing
            # trigonometry every sample on a naturally jittery stream.
            if abs(rate - self._designed_rate) / self._designed_rate > 0.05:
                self._design(rate)

        if not self._primed:
            # Initialise the state to the DC value of the first sample, so the
            # filter does not spend its first samples ramping up from zero.
            b0, b1, b2 = self._b
            a1, a2 = self._a
            dc_gain = (b0 + b1 + b2) / (1.0 + a1 + a2)
            self._z1 = x * (b1 + b2 - dc_gain * (a1 + a2)) if dc_gain else 0.0
            self._z2 = x * (b2 - dc_gain * a2) if dc_gain else 0.0
            self._primed = True

        b0, b1, b2 = self._b
        a1, a2 = self._a
        y = b0 * x + self._z1
        self._z1 = b1 * x - a1 * y + self._z2
        self._z2 = b2 * x - a2 * y
        return y
