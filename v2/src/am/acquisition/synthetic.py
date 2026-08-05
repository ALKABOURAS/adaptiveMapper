"""Synthetic IMU generator with ground truth.

Purpose
-------
Every claim of the form "filter F reduces noise by X without adding lag" needs a
known clean signal to compare against. With real hardware there is no such
reference: the hand movement is unrepeatable and the true angular rate unknown.

This generator produces a *known* voluntary motion plus separately controllable
error terms, so a filter can be scored against ground truth exactly. It is the
fixture for the filter unit tests, and it also lets the whole pipeline be built
and validated before the device is even connected.

Error model
-----------
Each term corresponds to a documented physical mechanism:

=================== ============================================================
term                mechanism
=================== ============================================================
voluntary motion    band-limited below ~5 Hz -- all deliberate hand movement
physiological tremor narrowband 8-12 Hz, present in every healthy hand
white noise         MEMS thermo-mechanical noise (angle random walk)
bias random walk    slow zero-point drift, the source of integration drift
spikes              transport corruption -- **not** Gaussian, defeats linear
                    filters, requires a nonlinear pre-stage
packet loss         Bluetooth retransmission, appears as gaps not as noise
=================== ============================================================

Note that packet loss is modelled as *missing samples*, never as noise. That
distinction is the point of research design §3.1: the transport problem is a
timing problem, not a spectral one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from am.acquisition.base import ImuSource
from am.core.types import ImuSample


@dataclass
class SyntheticConfig:
    """Parameters of the synthetic IMU model.

    Defaults approximate a consumer IMU held in the hand during a pointing task.
    """

    rate_hz: float = 200.0
    duration_s: float = 20.0

    # --- voluntary motion ---------------------------------------------------
    #: Peak angular rate of the deliberate movement, dps.
    motion_amplitude_dps: float = 60.0
    #: Frequencies composing the voluntary motion. All below 5 Hz by design.
    motion_freqs_hz: tuple[float, ...] = (0.3, 0.7, 1.3, 2.1)

    # --- physiological tremor ----------------------------------------------
    #: Tremor amplitude, dps. Typical postural tremor is a few dps.
    tremor_amplitude_dps: float = 3.0
    #: Tremor centre frequency, Hz. Physiological tremor sits at 8-12 Hz.
    tremor_freq_hz: float = 9.5
    #: Slow wander of the tremor frequency, Hz. Nonzero is what makes a fixed
    #: notch inadequate and motivates an adaptive one (WFLC).
    tremor_freq_drift_hz: float = 1.0

    # --- sensor noise -------------------------------------------------------
    #: White noise standard deviation, dps.
    white_noise_dps: float = 2.0
    #: Bias random-walk intensity, dps per sqrt(second).
    bias_walk_dps_per_sqrt_s: float = 0.05
    #: Initial bias offset, dps.
    initial_bias_dps: float = 0.5

    # --- transport ----------------------------------------------------------
    #: Probability per sample of an impulsive outlier.
    spike_probability: float = 0.002
    #: Outlier magnitude, dps.
    spike_magnitude_dps: float = 150.0
    #: Probability per report of the whole report being lost.
    packet_loss_probability: float = 0.01
    #: Samples carried per report, for loss modelling.
    frames_per_report: int = 3

    seed: int | None = 42

    # --- populated by the generator ----------------------------------------
    _rng: np.random.Generator | None = field(default=None, repr=False)


class SyntheticSource(ImuSource):
    """An :class:`ImuSource` that also exposes the ground truth it generated.

    After iterating, :attr:`truth` holds the clean voluntary signal and
    :attr:`truth_t` the corresponding device times, aligned sample for sample
    with what was yielded (dropped samples excluded from both).
    """

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        super().__init__()
        self.config = config or SyntheticConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self.truth_t: np.ndarray = np.empty(0)
        self.truth: np.ndarray = np.empty((0, 3))
        self._generated = False

    @property
    def nominal_rate_hz(self) -> float:
        return self.config.rate_hz

    def open(self) -> bool:
        return True

    def close(self) -> None:
        return None

    # ----------------------------------------------------------------- signals

    def _voluntary(self, t: np.ndarray) -> np.ndarray:
        """Band-limited deliberate motion, shape (N, 3)."""
        c = self.config
        out = np.zeros((t.size, 3))
        for axis in range(3):
            phase = self._rng.uniform(0, 2 * np.pi, size=len(c.motion_freqs_hz))
            weight = self._rng.uniform(0.5, 1.0, size=len(c.motion_freqs_hz))
            weight /= weight.sum()
            signal = np.zeros_like(t)
            for f, p, w in zip(c.motion_freqs_hz, phase, weight):
                signal += w * np.sin(2 * np.pi * f * t + p)
            # Axis 2 (yaw) carries the dominant motion in a pointing task.
            scale = c.motion_amplitude_dps * (1.0 if axis == 2 else 0.4)
            out[:, axis] = scale * signal
        return out

    def _tremor(self, t: np.ndarray) -> np.ndarray:
        """Narrowband tremor with a slowly wandering centre frequency."""
        c = self.config
        if c.tremor_amplitude_dps <= 0:
            return np.zeros((t.size, 3))
        # Frequency wanders slowly, so instantaneous phase is the integral of
        # instantaneous frequency rather than f*t.
        wander = c.tremor_freq_drift_hz * np.sin(2 * np.pi * 0.05 * t)
        inst_freq = c.tremor_freq_hz + wander
        dt = np.gradient(t)
        phase = np.cumsum(2 * np.pi * inst_freq * dt)
        out = np.zeros((t.size, 3))
        for axis in range(3):
            out[:, axis] = c.tremor_amplitude_dps * np.sin(
                phase + self._rng.uniform(0, 2 * np.pi)
            )
        return out

    def _bias_walk(self, t: np.ndarray) -> np.ndarray:
        c = self.config
        dt = float(np.mean(np.diff(t))) if t.size > 1 else 1.0 / c.rate_hz
        steps = self._rng.normal(
            0.0, c.bias_walk_dps_per_sqrt_s * np.sqrt(dt), size=(t.size, 3)
        )
        return c.initial_bias_dps + np.cumsum(steps, axis=0)

    def _spikes(self, n: int) -> np.ndarray:
        c = self.config
        out = np.zeros((n, 3))
        if c.spike_probability <= 0:
            return out
        hit = self._rng.random((n, 3)) < c.spike_probability
        sign = self._rng.choice([-1.0, 1.0], size=(n, 3))
        magnitude = c.spike_magnitude_dps * self._rng.uniform(0.5, 1.5, size=(n, 3))
        out[hit] = (sign * magnitude)[hit]
        return out

    # ------------------------------------------------------------------ stream

    def generate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build the whole session at once.

        Returns
        -------
        (t, clean, measured, keep_mask)
            ``clean`` is the voluntary motion only -- the ground truth a filter
            should recover. ``measured`` is what the sensor reports.
            ``keep_mask`` is False where a packet was lost.
        """
        c = self.config
        n = int(c.duration_s * c.rate_hz)
        t = np.arange(n) / c.rate_hz

        clean = self._voluntary(t)
        measured = (
            clean
            + self._tremor(t)
            + self._bias_walk(t)
            + self._rng.normal(0.0, c.white_noise_dps, size=(n, 3))
            + self._spikes(n)
        )

        keep = np.ones(n, dtype=bool)
        if c.packet_loss_probability > 0:
            n_reports = int(np.ceil(n / c.frames_per_report))
            lost = self._rng.random(n_reports) < c.packet_loss_probability
            for report_index in np.flatnonzero(lost):
                start = report_index * c.frames_per_report
                keep[start : start + c.frames_per_report] = False

        return t, clean, measured, keep

    def samples(self) -> Iterator[ImuSample]:
        t, clean, measured, keep = self.generate()
        self.truth_t = t[keep]
        self.truth = clean[keep]
        self._generated = True

        accel = np.zeros(3)
        seq = 0
        pending_drops = 0
        for i in range(t.size):
            if not keep[i]:
                pending_drops += 1
                self.stats.samples_dropped += 1
                continue

            sample = ImuSample(
                t_device=float(t[i]),
                t_host=float(t[i]),
                gyro=measured[i].copy(),
                accel=accel,
                dropped_before=pending_drops,
                seq=seq,
            )
            pending_drops = 0
            seq += 1

            if self.stats.t_first is None:
                self.stats.t_first = sample.t_device
            self.stats.t_last = sample.t_device
            self.stats.samples_emitted += 1
            yield sample


def make_step_response_signal(
    rate_hz: float = 200.0,
    duration_s: float = 2.0,
    step_time_s: float = 0.5,
    step_dps: float = 100.0,
    noise_dps: float = 2.0,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clean step input for measuring rise time, overshoot and settling.

    A step is the sharpest possible test of the jitter/lag trade-off: any filter
    that suppresses noise well will round the corner, and how much it rounds it
    is precisely the cost being paid.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * rate_hz)
    t = np.arange(n) / rate_hz
    clean = np.where(t >= step_time_s, step_dps, 0.0)
    measured = clean + rng.normal(0.0, noise_dps, size=n)
    return t, clean, measured


def make_ramp_signal(
    rate_hz: float = 200.0,
    duration_s: float = 2.0,
    slope_dps_per_s: float = 100.0,
    noise_dps: float = 2.0,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constant-velocity ramp.

    This is the input that separates an honest estimator from a degenerate one.
    A scalar random-walk Kalman filter -- the v1 implementation -- has a
    *structural* steady-state error on a ramp, because its process model says
    the signal is constant. A constant-velocity estimator tracks it with zero
    steady-state error. The v1 conclusion "Kalman lags" was a statement about
    that modelling choice, not about Kalman filtering.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * rate_hz)
    t = np.arange(n) / rate_hz
    clean = slope_dps_per_s * t
    measured = clean + rng.normal(0.0, noise_dps, size=n)
    return t, clean, measured
