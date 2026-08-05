"""Spectral characterisation of the gyroscope signal.

This module produces the evidence for the central design claim: that voluntary
motion and physiological tremor occupy separable frequency bands, and that the
processing chain should therefore be designed in the frequency domain rather
than tuned by eye.

Validity precondition
---------------------
Every function here assumes a **uniform** sampling grid. The v1 stream had 21 %
timing jitter, which smears spectral estimates and makes them uninterpretable.
:func:`assert_uniform` enforces the precondition explicitly rather than letting
an invalid analysis proceed silently.
"""

from __future__ import annotations

import warnings

import numpy as np

#: Band definitions used throughout the study, in Hz.
BAND_VOLUNTARY = (0.0, 5.0)
BAND_TRANSITION = (5.0, 8.0)
BAND_TREMOR = (8.0, 12.0)
BAND_NOISE = (15.0, None)  # None means "up to Nyquist"


def assert_uniform(t, tolerance_percent: float = 5.0) -> float:
    """Verify a uniform time base and return the sample rate.

    Raises
    ------
    ValueError
        If the coefficient of variation of dt exceeds ``tolerance_percent``.
        Spectral estimates on such a series are not trustworthy, and silently
        computing one is worse than refusing.
    """
    t = np.asarray(t, dtype=np.float64)
    if t.size < 3:
        raise ValueError("need at least 3 samples")
    dt = np.diff(t)
    dt = dt[dt > 0]
    mean, std = float(dt.mean()), float(dt.std())
    cv = std / mean * 100.0
    if cv > tolerance_percent:
        raise ValueError(
            f"Time base is not uniform: dt CV = {cv:.1f}% "
            f"(limit {tolerance_percent}%). Spectral analysis on this series is "
            f"invalid. Resample onto a uniform grid or fix acquisition first."
        )
    return 1.0 / mean


def _segment(x: np.ndarray, nperseg: int, noverlap: int) -> np.ndarray:
    """Split into overlapping segments, shape (n_segments, nperseg)."""
    step = nperseg - noverlap
    n_seg = 1 + (x.size - nperseg) // step if x.size >= nperseg else 0
    if n_seg <= 0:
        return np.empty((0, nperseg), dtype=np.float64)
    idx = np.arange(nperseg)[None, :] + step * np.arange(n_seg)[:, None]
    return x[idx]


def welch_psd(
    signal,
    rate_hz: float,
    nperseg: int | None = None,
    detrend: bool = True,
    overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided power spectral density by Welch's method.

    Implemented directly in numpy so that the analysis has no scipy dependency
    -- the study should be reproducible from a minimal environment.

    Returns ``(frequencies_hz, psd)`` with the PSD in units^2/Hz.
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    if x.size < 8:
        raise ValueError("need at least 8 samples for a spectral estimate")

    if nperseg is None:
        # ~2 s windows give ~0.5 Hz resolution: enough to resolve an 8-12 Hz
        # peak while retaining several averaging segments in a 20 s recording.
        target = min(x.size, int(2.0 * rate_hz))
        nperseg = max(64, 1 << int(np.floor(np.log2(max(target, 64)))))
    nperseg = int(min(nperseg, x.size))

    noverlap = int(nperseg * overlap)
    segments = _segment(x, nperseg, noverlap)
    if segments.shape[0] == 0:
        segments = x[:nperseg][None, :]

    if detrend:
        segments = segments - segments.mean(axis=1, keepdims=True)

    window = np.hanning(nperseg + 1)[:nperseg]  # periodic Hann
    windowed = segments * window

    spectrum = np.fft.rfft(windowed, axis=1)
    psd = (np.abs(spectrum) ** 2) / (rate_hz * np.sum(window**2))

    # One-sided: fold the negative frequencies onto the positive ones, leaving
    # DC and (for even nperseg) Nyquist unscaled.
    if psd.shape[1] > 2:
        psd[:, 1:-1] *= 2.0 if nperseg % 2 == 0 else 1.0
        if nperseg % 2 != 0:
            psd[:, 1:] *= 2.0

    freqs = np.fft.rfftfreq(nperseg, d=1.0 / rate_hz)
    return freqs, psd.mean(axis=0)


def band_power(freqs, psd, band: tuple[float, float | None]) -> float:
    """Integrate the PSD over a frequency band."""
    freqs = np.asarray(freqs)
    psd = np.asarray(psd)
    lo, hi = band
    hi = freqs[-1] if hi is None else hi
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def band_report(signal, rate_hz: float) -> dict[str, float]:
    """Fraction of total power in each band of interest.

    This is the number that decides whether tremor suppression is worth
    including at all. If the 8-12 Hz band holds a negligible share of the power
    for this device and task, WFLC should be dropped and the thesis should say
    so -- a negative result honestly reported.
    """
    freqs, psd = welch_psd(signal, rate_hz)
    total = band_power(freqs, psd, (0.0, None))
    if total <= 0:
        return {}

    bands = {
        "voluntary_0_5Hz": BAND_VOLUNTARY,
        "transition_5_8Hz": BAND_TRANSITION,
        "tremor_8_12Hz": BAND_TREMOR,
        "noise_15Hz_up": BAND_NOISE,
    }
    out = {name: band_power(freqs, psd, b) / total * 100.0 for name, b in bands.items()}

    tremor_mask = (freqs >= BAND_TREMOR[0]) & (freqs <= BAND_TREMOR[1])
    if np.any(tremor_mask):
        out["tremor_peak_hz"] = float(freqs[tremor_mask][np.argmax(psd[tremor_mask])])
        out["tremor_prominence"] = tremor_prominence(freqs, psd)
        out["tremor_band_power"] = band_power(freqs, psd, BAND_TREMOR)
    out["total_power"] = total
    return out


def find_tremor_peak(
    freqs, psd, search: tuple[float, float] = (6.0, 14.0)
) -> tuple[float, float]:
    """Locate a genuine tremor peak, returning ``(frequency_hz, prominence)``.

    Taking the plain argmax over a search window is unsafe here, and the failure
    is not hypothetical: during fast movement, 88 % of the power sits below 5 Hz,
    so the monotonically falling tail of the voluntary motion is the largest
    value anywhere in 6-14 Hz. The argmax then lands on the lower edge of the
    window and reports it as a "tremor peak at 6.01 Hz", which would have
    inflated an apparent tremor-frequency shift from about 0.9 Hz to 4 Hz.

    This function instead removes the broadband trend first, by fitting a
    straight line in log-log space across the search window, and then requires
    the maximum of the residual to be an **interior local maximum**. A peak
    sitting on the window boundary is a tail, not a resonance, and is rejected.

    Returns ``(nan, nan)`` when no interior peak exists.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)

    window = (freqs >= search[0]) & (freqs <= search[1]) & (psd > 0) & (freqs > 0)
    if np.count_nonzero(window) < 7:
        return (float("nan"), float("nan"))

    f_w = freqs[window]
    p_w = psd[window]

    # Remove the broadband 1/f-like trend so that only local structure remains.
    coeffs = np.polyfit(np.log10(f_w), np.log10(p_w), 1)
    baseline = 10.0 ** np.polyval(coeffs, np.log10(f_w))
    residual = p_w / baseline

    interior = np.arange(1, residual.size - 1)
    is_local_max = (residual[interior] > residual[interior - 1]) & (
        residual[interior] > residual[interior + 1]
    )
    candidates = interior[is_local_max]
    if candidates.size == 0:
        return (float("nan"), float("nan"))

    best = candidates[np.argmax(residual[candidates])]
    return (float(f_w[best]), float(residual[best]))


def find_tremor_peak_robust(
    signal,
    rate_hz: float,
    search: tuple[float, float] = (6.0, 14.0),
    windows: tuple[int, ...] = (512, 1024, 2048, 4096),
    max_spread_hz: float = 0.5,
) -> dict[str, float]:
    """Locate a tremor peak and verify it does not depend on the window length.

    A single Welch window can manufacture a peak. Longer windows give finer
    frequency resolution but fewer segments to average, so the estimate becomes
    noisy; shorter windows average well but may not resolve the peak. A genuine
    resonance appears at the same frequency under all of them.

    This check matters, and not in theory. On the measured recordings the
    pointing condition returned 8.98, 8.98, 9.08, 9.03 Hz across the four
    windows -- stable to within 0.1 Hz. The fast-movement condition returned
    8.98, 12.89, 8.98, 8.11 Hz, because its spectrum is dominated by harmonics
    of the repetitive voluntary movement, and which harmonic wins depends on the
    resolution. Reading a single window there would have supported a claim of a
    tremor-frequency shift that the data does not support.

    Returns
    -------
    dict with ``peak_hz`` (median across windows), ``spread_hz``,
    ``prominence`` (median), ``n_valid``, and ``stable``.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    peaks, prominences = [], []
    for nperseg in windows:
        if nperseg > signal.size:
            continue
        freqs, psd = welch_psd(signal, rate_hz, nperseg=nperseg)
        peak, prominence = find_tremor_peak(freqs, psd, search)
        if np.isfinite(peak):
            peaks.append(peak)
            prominences.append(prominence)

    if not peaks:
        return {
            "peak_hz": float("nan"),
            "spread_hz": float("nan"),
            "prominence": float("nan"),
            "n_valid": 0.0,
            "stable": 0.0,
        }

    peaks = np.asarray(peaks)
    spread = float(peaks.max() - peaks.min())
    return {
        "peak_hz": float(np.median(peaks)),
        "spread_hz": spread,
        "prominence": float(np.median(prominences)),
        "n_valid": float(peaks.size),
        "stable": float(spread <= max_spread_hz and peaks.size >= 3),
    }


def tremor_prominence(freqs, psd) -> float:
    """How much the 8-12 Hz peak stands above the surrounding spectrum.

    Reporting only ``tremor_peak_hz`` is misleading, because the maximum of a
    *flat* spectrum inside 8-12 Hz is still some frequency in 8-12 Hz. A
    stationary recording will therefore appear to have a "tremor peak" when it
    has nothing of the sort. Prominence distinguishes a genuine resonance from
    the arbitrary argmax of a noise floor:

    - below ~1.5 : no peak, the band is just part of a smooth spectrum
    - 1.5 to 3   : weak
    - above 3    : a real, localised peak

    The baseline is the geometric mean of the 6-7 Hz and 13-15 Hz shoulders,
    which brackets the tremor band without including it.
    """
    freqs = np.asarray(freqs)
    psd = np.asarray(psd)
    band = (freqs >= BAND_TREMOR[0]) & (freqs <= BAND_TREMOR[1])
    low = (freqs >= 6.0) & (freqs < 7.0)
    high = (freqs > 13.0) & (freqs <= 15.0)
    if not (np.any(band) and np.any(low) and np.any(high)):
        return float("nan")

    positive = psd > 0
    low &= positive
    high &= positive
    if not (np.any(low) and np.any(high)):
        return float("nan")

    baseline = np.exp(
        0.5 * (np.log(psd[low]).mean() + np.log(psd[high]).mean())
    )
    if baseline <= 0:
        return float("nan")
    return float(psd[band].max() / baseline)


def spectrogram(
    signal, rate_hz: float, window_s: float = 1.0, overlap: float = 0.75
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-frequency view, ``(freqs, times, Sxx)``.

    Shows the tremor frequency wandering within a session -- the empirical
    justification for tracking it adaptively instead of using a fixed notch.
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    nperseg = int(min(max(32, int(window_s * rate_hz)), x.size))
    noverlap = int(nperseg * overlap)

    segments = _segment(x, nperseg, noverlap)
    if segments.shape[0] == 0:
        raise ValueError("signal shorter than one spectrogram window")

    segments = segments - segments.mean(axis=1, keepdims=True)
    window = np.hanning(nperseg + 1)[:nperseg]
    spectrum = np.fft.rfft(segments * window, axis=1)
    sxx = (np.abs(spectrum) ** 2) / (rate_hz * np.sum(window**2))
    if sxx.shape[1] > 2:
        sxx[:, 1:-1] *= 2.0

    freqs = np.fft.rfftfreq(nperseg, d=1.0 / rate_hz)
    step = nperseg - noverlap
    times = (np.arange(segments.shape[0]) * step + nperseg / 2) / rate_hz
    return freqs, times, sxx.T


def allan_deviation(
    signal, rate_hz: float, n_taus: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping Allan deviation of a gyroscope rate signal.

    The standard IEEE characterisation of inertial sensor noise. The log-log
    slope identifies the dominant noise process:

    ======  ==================================  =========================
    slope   process                              read at
    ======  ==================================  =========================
    -1/2    angle random walk (white noise)      tau = 1 s
    0       bias instability (the floor)         minimum of the curve
    +1/2    rate random walk (slow drift)        long tau
    ======  ==================================  =========================

    This gives principled values for the Kalman filter's ``measurement_var`` and
    ``process_var`` instead of hand tuning, and it quantifies the integration
    drift discussed in thesis §3.1 as a measured device property.

    Returns ``(taus_seconds, allan_deviation)`` in the units of ``signal``.
    """
    x = np.asarray(signal, dtype=np.float64)
    n = x.size
    if n < 16:
        raise ValueError("need at least 16 samples")

    dt = 1.0 / rate_hz
    # Cumulative sum converts rate into angle; Allan variance is defined on the
    # integrated quantity.
    theta = np.cumsum(x) * dt

    max_m = (n - 1) // 2
    ms = np.unique(np.logspace(0, np.log10(max_m), n_taus).astype(int))
    ms = ms[ms >= 1]

    taus = ms * dt
    avar = np.empty(ms.size, dtype=np.float64)
    for i, m in enumerate(ms):
        k = n - 2 * m
        if k <= 0:
            avar[i] = np.nan
            continue
        d = theta[2 * m :] - 2.0 * theta[m : m + k] + theta[:k]
        avar[i] = np.sum(d**2) / (2.0 * k * (m * dt) ** 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adev = np.sqrt(avar)

    valid = np.isfinite(adev)
    return taus[valid], adev[valid]


def is_stationary(signal, rate_hz: float, max_voluntary_share: float = 20.0) -> bool:
    """Heuristic check that a recording contains no deliberate motion.

    Allan variance characterises *sensor* noise and is only meaningful on a
    stationary recording. Applied to a signal containing hand motion it reports
    the motion, producing an angle random walk inflated by orders of magnitude.
    """
    try:
        freqs, psd = welch_psd(signal, rate_hz)
    except ValueError:
        return False
    total = band_power(freqs, psd, (0.0, None))
    if total <= 0:
        return True
    share = band_power(freqs, psd, (0.05, 5.0)) / total * 100.0
    return share <= max_voluntary_share


def noise_parameters(signal, rate_hz: float, check_stationary: bool = True) -> dict[str, float]:
    """Extract angle random walk and bias instability from the Allan curve.

    Parameters
    ----------
    check_stationary:
        Warn if the recording appears to contain voluntary motion, in which
        case the returned values describe the movement rather than the sensor
        and must not be quoted as device characteristics.
    """
    if check_stationary and not is_stationary(signal, rate_hz):
        warnings.warn(
            "Allan variance computed on a signal with substantial 0.05-5 Hz "
            "power. This looks like a recording containing hand motion, not a "
            "stationary one. The resulting ARW and bias instability describe "
            "the motion, not the sensor, and must not be reported as device "
            "specifications. Record with the device at rest instead.",
            RuntimeWarning,
            stacklevel=2,
        )

    taus, adev = allan_deviation(signal, rate_hz)
    if taus.size == 0:
        return {}

    # ARW: the -1/2 slope asymptote evaluated at tau = 1 s.
    idx = int(np.argmin(np.abs(taus - 1.0)))
    arw = float(adev[idx])

    # Bias instability: the curve minimum, scaled by the standard 0.664 factor.
    imin = int(np.argmin(adev))
    bias_instability = float(adev[imin] / 0.664)

    return {
        "arw_dps_per_sqrt_hz": arw,
        "arw_deg_per_sqrt_hour": arw * 60.0,
        "bias_instability_dps": bias_instability,
        "bias_instability_tau_s": float(taus[imin]),
        # Direct input for ConstantVelocityKalman.measurement_var.
        "suggested_measurement_var": float((arw * np.sqrt(rate_hz)) ** 2),
    }


def attenuation_db(raw, filtered, rate_hz: float, band: tuple[float, float]) -> float:
    """Attenuation achieved in a specific band, in dB.

    The direct test of a tremor suppressor: it should show large attenuation in
    8-12 Hz and near-zero attenuation in 0-5 Hz. A filter that attenuates both
    is just a low-pass wearing a costume.
    """
    f_raw, p_raw = welch_psd(raw, rate_hz)
    f_filt, p_filt = welch_psd(filtered, rate_hz)
    e_raw = band_power(f_raw, p_raw, band)
    e_filt = band_power(f_filt, p_filt, band)
    if e_raw <= 0 or e_filt <= 0:
        return 0.0
    return float(10.0 * np.log10(e_raw / e_filt))
