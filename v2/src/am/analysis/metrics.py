"""Signal-level metrics for Level 1 filter evaluation.

Every metric here answers one specific question from the research design, and
each is defined so that it can be computed identically on synthetic data (with
exact ground truth) and on hardware recordings against a mechanical reference.

The pair that matters most is (:func:`stationary_noise`, :func:`phase_lag_ms`).
Any filter can win either one alone -- pass everything through for zero lag, or
smooth to a constant for zero noise. The contribution is the trade-off between
them, which is why they are always reported together.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

#: Perceptual latency threshold for direct manipulation, milliseconds.
#: MacKenzie and Ware (1993) report a marked degradation in pointing performance
#: beyond roughly this value; it is the acceptance criterion for RQ1.
PERCEPTUAL_LAG_THRESHOLD_MS = 75.0


@dataclass
class FilterScore:
    """Result of evaluating one filter against a known reference."""

    name: str
    rmse: float
    stationary_noise: float
    lag_ms: float
    noise_reduction_db: float
    overshoot_percent: float
    spikes_remaining: int
    passes_lag_threshold: bool

    def as_dict(self) -> dict:
        return asdict(self)


def stationary_noise(signal, sample_slice: slice | None = None) -> float:
    """Standard deviation over an interval where the device was held still.

    Directly operationalises "jitter" in RQ1. Report in dps for gyro data.
    """
    a = np.asarray(signal, dtype=np.float64)
    if sample_slice is not None:
        a = a[sample_slice]
    return float(np.std(a)) if a.size > 1 else 0.0


def phase_lag_ms(reference, filtered, rate_hz: float, max_lag_ms: float = 300.0) -> float:
    """Effective latency, from the peak of the cross-correlation.

    Estimates the delay that best aligns ``filtered`` with ``reference``.
    A positive result means the filtered signal lags. A **negative** result
    means it leads, which predictive filters such as double exponential
    smoothing genuinely can achieve.

    Uses the correlation peak rather than a fixed group-delay formula because
    adaptive filters have no single group delay -- theirs depends on the motion,
    which is the entire point of using them.
    """
    ref = np.asarray(reference, dtype=np.float64)
    filt = np.asarray(filtered, dtype=np.float64)
    n = min(ref.size, filt.size)
    ref, filt = ref[:n], filt[:n]

    ref = ref - ref.mean()
    filt = filt - filt.mean()
    if np.allclose(ref, 0) or np.allclose(filt, 0):
        return 0.0

    max_lag = int(max_lag_ms * 1e-3 * rate_hz)
    max_lag = max(1, min(max_lag, n - 1))

    corr = np.correlate(filt, ref, mode="full")
    centre = n - 1
    window = corr[centre - max_lag : centre + max_lag + 1]
    best = int(np.argmax(window)) - max_lag
    return best / rate_hz * 1e3


def noise_reduction_db(raw, filtered, reference=None) -> float:
    """Noise attenuation in dB.

    With a ``reference``, measures the reduction in error energy against ground
    truth -- the meaningful quantity, because it penalises a filter that removes
    noise by also removing signal. Without one, falls back to the ratio of
    signal variances, which should be read as indicative only.
    """
    raw = np.asarray(raw, dtype=np.float64)
    filtered = np.asarray(filtered, dtype=np.float64)

    if reference is not None:
        ref = np.asarray(reference, dtype=np.float64)
        n = min(raw.size, filtered.size, ref.size)
        e_raw = np.var(raw[:n] - ref[:n])
        e_filt = np.var(filtered[:n] - ref[:n])
    else:
        e_raw = np.var(raw)
        e_filt = np.var(filtered)

    if e_filt <= 0 or e_raw <= 0:
        return 0.0
    return float(10.0 * np.log10(e_raw / e_filt))


def rmse(reference, estimate) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    n = min(ref.size, est.size)
    return float(np.sqrt(np.mean((ref[:n] - est[:n]) ** 2)))


def lag_compensated_rmse(reference, estimate, rate_hz: float) -> float:
    """RMSE after removing the estimated lag.

    Separates the two ways a filter can be wrong: *delayed* (recoverable by
    shifting) and *distorted* (not recoverable). Comparing this with plain RMSE
    shows how much of the error is pure latency.
    """
    lag_samples = int(round(phase_lag_ms(reference, estimate, rate_hz) * 1e-3 * rate_hz))
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    if lag_samples > 0:
        est = est[lag_samples:]
        ref = ref[: est.size]
    elif lag_samples < 0:
        ref = ref[-lag_samples:]
        est = est[: ref.size]
    return rmse(ref, est)


def step_response_metrics(
    t, response, step_time: float, step_amplitude: float
) -> dict[str, float]:
    """Rise time, overshoot and settling time for a step input."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(response, dtype=np.float64)
    after = t >= step_time
    if not np.any(after) or step_amplitude == 0:
        return {"rise_time_ms": 0.0, "overshoot_percent": 0.0, "settling_time_ms": 0.0}

    t_after = t[after] - step_time
    y_after = y[after]

    def first_crossing(fraction: float) -> float:
        target = fraction * step_amplitude
        idx = np.flatnonzero(
            y_after >= target if step_amplitude > 0 else y_after <= target
        )
        return float(t_after[idx[0]]) if idx.size else float("nan")

    t10, t90 = first_crossing(0.1), first_crossing(0.9)
    rise = (t90 - t10) * 1e3 if np.isfinite(t90) and np.isfinite(t10) else float("nan")

    peak = float(np.max(y_after) if step_amplitude > 0 else np.min(y_after))
    overshoot = (peak - step_amplitude) / abs(step_amplitude) * 100.0
    overshoot = max(0.0, overshoot if step_amplitude > 0 else -overshoot)

    tol = 0.02 * abs(step_amplitude)
    outside = np.flatnonzero(np.abs(y_after - step_amplitude) > tol)
    settling = float(t_after[outside[-1]]) * 1e3 if outside.size else 0.0

    return {
        "rise_time_ms": rise,
        "overshoot_percent": overshoot,
        "settling_time_ms": settling,
    }


def count_spikes(signal, threshold_sigmas: float = 5.0) -> int:
    """Count impulsive outlier *events* in a signal.

    Detection operates on the discrete second difference

        d[j] = x[j] - 2*x[j+1] + x[j+2]

    rather than on the raw values. This matters: comparing samples against a
    global median works only for a signal with no structure. Here the voluntary
    motion is large -- tens of dps -- so it dominates the global median absolute
    deviation and inflates the detection threshold above the very spikes it is
    meant to catch. A smooth signal has a near-zero second difference regardless
    of its amplitude, whereas an isolated impulse of height h produces a second
    difference of 2h, so the transform isolates impulses from signal content.

    Consecutive detections are grouped, so one impulse counts once rather than
    once per affected sample.
    """
    a = np.asarray(signal, dtype=np.float64).ravel()
    if a.size < 5:
        return 0

    d2 = a[:-2] - 2.0 * a[1:-1] + a[2:]
    sigma = 1.4826 * float(np.median(np.abs(d2 - np.median(d2))))
    if sigma <= 0:
        return 0

    flagged = np.abs(d2) > threshold_sigmas * sigma
    if not np.any(flagged):
        return 0
    # Count runs of consecutive flags: one impulse spans up to 3 samples of d2.
    return int(np.sum(flagged[1:] & ~flagged[:-1]) + int(flagged[0]))


def evaluate_filter(
    filt,
    raw,
    reference,
    rate_hz: float,
    stationary_slice: slice | None = None,
) -> FilterScore:
    """Score one filter against ground truth. The Level 1 workhorse."""
    raw = np.asarray(raw, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    out = filt.process(raw, 1.0 / rate_hz)

    lag = phase_lag_ms(ref, out, rate_hz)
    peak_ref = float(np.max(np.abs(ref))) or 1.0
    overshoot = max(0.0, (float(np.max(np.abs(out))) - peak_ref) / peak_ref * 100.0)

    return FilterScore(
        name=getattr(filt, "name", type(filt).__name__),
        rmse=rmse(ref, out),
        stationary_noise=stationary_noise(out - ref, stationary_slice),
        lag_ms=lag,
        noise_reduction_db=noise_reduction_db(raw, out, ref),
        overshoot_percent=overshoot,
        spikes_remaining=count_spikes(out),
        passes_lag_threshold=abs(lag) <= PERCEPTUAL_LAG_THRESHOLD_MS,
    )
