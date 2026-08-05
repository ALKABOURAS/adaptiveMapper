"""Kalman estimators, including an honest constant-velocity baseline.

The methodological problem this module fixes
--------------------------------------------
The v1 study concluded that "the Kalman filter lags" and moved to the 1 euro
filter on that basis. The filter it tested was:

    P = P + Q
    K = P / (P + R)
    x = x + K * (z - x)
    P = (1 - K) * P

This is a scalar random-walk estimator. Its process model asserts that the true
signal is *constant* and only perturbed by small random steps. Feed it a ramp --
a hand moving at steady angular velocity, which is most of any pointing
movement -- and it exhibits a steady-state tracking error proportional to the
slope. That is a structural property of the chosen model, provable from the
steady-state equations; it is not an empirical finding about Kalman filtering.

Comparing an adaptive filter against a model that cannot represent motion is
comparing against a straw man, and a reader with a signals background will say
so. :class:`ConstantVelocityKalman` includes velocity in the state and tracks a
ramp with **zero** steady-state error.

The point is not that the v1 conclusion was reversed but that it must be
re-earned. With honest baselines the interesting question becomes *where* the
1 euro filter still wins: cost, absence of a dynamics model, two interpretable
parameters, no covariance tuning. That is a stronger claim than the original.
"""

from __future__ import annotations

import numpy as np

from am.filters.base import ScalarFilter


class ScalarRandomWalkKalman(ScalarFilter):
    """The v1 estimator, preserved verbatim as a reproducible condition.

    Kept so that the thesis can *show* the structural ramp error rather than
    assert it, and so the v1 results remain reproducible. Not recommended for
    tracking a moving signal.
    """

    name = "K0-randomwalk(v1)"

    def __init__(
        self,
        process_noise: float = 1e-5,
        measurement_noise: float = 0.1,
        initial_error: float = 1.0,
    ) -> None:
        self.q = process_noise
        self.r = measurement_noise
        self.initial_error = initial_error
        self.reset()

    def reset(self) -> None:
        self._p = self.initial_error
        self._x = 0.0
        self._initialised = False

    def update(self, x: float, dt: float) -> float:
        if not self._initialised:
            self._x = x
            self._initialised = True
            return x
        self._p += self.q
        k = self._p / (self._p + self.r)
        self._x += k * (x - self._x)
        self._p *= 1.0 - k
        return self._x


class ConstantVelocityKalman(ScalarFilter):
    """Two-state (position, velocity) Kalman filter with a CV process model.

    State ``[x, v]``, transition ``F = [[1, dt], [0, 1]]``, and the standard
    continuous white-noise-acceleration process covariance

        Q = q * [[dt^3/3, dt^2/2],
                 [dt^2/2, dt     ]]

    Because velocity is part of the state, a constant-velocity input is tracked
    with zero steady-state error -- the property the v1 estimator lacked.

    Parameters
    ----------
    process_var:
        Spectral density of the acceleration noise, (unit/s^2)^2 per Hz. This is
        the single tuning knob: larger means the filter believes the velocity
        can change quickly, so it tracks harder and smooths less.
    measurement_var:
        Variance of the measurement noise, (unit)^2. Set it from a measured
        stationary recording rather than by trial and error -- that is what the
        Allan variance and the stationary noise metric are for.
    """

    def __init__(
        self,
        process_var: float = 1e3,
        measurement_var: float = 4.0,
        *,
        name: str | None = None,
    ) -> None:
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.name = name or f"F7-KalmanCV(q={process_var:g},r={measurement_var:g})"
        self.reset()

    def reset(self) -> None:
        self._x = np.zeros(2, dtype=np.float64)
        self._p = np.eye(2, dtype=np.float64) * 1e3
        self._initialised = False

    @property
    def velocity(self) -> float:
        """Current velocity estimate. Free by-product, useful for gain control."""
        return float(self._x[1])

    def update(self, z: float, dt: float) -> float:
        if not self._initialised:
            self._x = np.array([z, 0.0], dtype=np.float64)
            self._initialised = True
            return z
        if dt <= 0:
            return float(self._x[0])

        f = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        q = self.process_var * np.array(
            [[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]], dtype=np.float64
        )

        # Predict
        self._x = f @ self._x
        self._p = f @ self._p @ f.T + q

        # Update (scalar measurement of position -> no matrix inverse needed)
        innovation = z - self._x[0]
        s = self._p[0, 0] + self.measurement_var
        k = self._p[:, 0] / s
        self._x = self._x + k * innovation
        self._p = self._p - np.outer(k, self._p[0, :])

        return float(self._x[0])


class AdaptiveKalman(ConstantVelocityKalman):
    """Constant-velocity Kalman whose process noise is scheduled on speed.

    This is the optimal-estimation analogue of what the 1 euro filter does
    heuristically: when the hand moves fast, trust the model less and the
    measurement more.

        q_eff = q_base * (1 + kappa * |v_hat|)

    Including it makes the comparison fair in the other direction too. If the
    1 euro filter still wins on the task metrics against *this*, the result
    means something. If it does not, that is a finding worth reporting, and the
    thesis argument shifts to cost and tunability -- which it can defend.

    Parameters
    ----------
    speed_coupling:
        ``kappa``, in inverse units of velocity. Zero reduces this to the plain
        constant-velocity filter.
    innovation_window:
        If > 0, the measurement variance is additionally re-estimated online
        from the recent innovation sequence, making the filter robust to a
        misspecified ``measurement_var``.
    """

    def __init__(
        self,
        process_var: float = 1e2,
        measurement_var: float = 4.0,
        speed_coupling: float = 0.05,
        innovation_window: int = 0,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(process_var, measurement_var)
        self.speed_coupling = speed_coupling
        self.innovation_window = innovation_window
        self.name = name or f"F8-KalmanAdaptive(q={process_var:g},k={speed_coupling:g})"
        self.reset()

    def reset(self) -> None:
        super().reset()
        self._innovations: list[float] = []

    def update(self, z: float, dt: float) -> float:
        if not self._initialised:
            return super().update(z, dt)
        if dt <= 0:
            return float(self._x[0])

        speed = abs(self._x[1])
        q_eff = self.process_var * (1.0 + self.speed_coupling * speed)

        f = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        q = q_eff * np.array(
            [[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]], dtype=np.float64
        )

        self._x = f @ self._x
        self._p = f @ self._p @ f.T + q

        innovation = z - self._x[0]

        r = self.measurement_var
        if self.innovation_window > 0:
            self._innovations.append(innovation)
            if len(self._innovations) > self.innovation_window:
                self._innovations.pop(0)
            if len(self._innovations) >= max(5, self.innovation_window // 2):
                # Innovation-based adaptive estimation: var(innovation) should
                # equal H P H' + R, so R is recovered by subtraction.
                emp = float(np.var(self._innovations))
                r = max(1e-6, emp - self._p[0, 0])

        s = self._p[0, 0] + r
        k = self._p[:, 0] / s
        self._x = self._x + k * innovation
        self._p = self._p - np.outer(k, self._p[0, :])

        return float(self._x[0])
