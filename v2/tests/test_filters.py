"""Filter correctness tests against synthetic ground truth.

These are not smoke tests. Each one asserts a claim that the thesis will make in
prose, so that no claim rests on a plot that looked convincing.

The most important is :func:`test_ramp_error_is_structural_not_empirical`, which
demonstrates the v1 methodological error directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from am.acquisition.synthetic import (  # noqa: E402
    SyntheticConfig,
    SyntheticSource,
    make_ramp_signal,
    make_step_response_signal,
)
from am.analysis.metrics import (  # noqa: E402
    PERCEPTUAL_LAG_THRESHOLD_MS,
    count_spikes,
    phase_lag_ms,
    rmse,
)
from am.analysis.spectral import attenuation_db, band_report  # noqa: E402
from am.core.timebase import CounterTimebase  # noqa: E402
from am.filters import (  # noqa: E402
    AdaptiveKalman,
    BiquadLowpass,
    CascadedOneEuro,
    ConstantVelocityKalman,
    DoubleExponentialSmoothing,
    HampelFilter,
    OneEuroFilter,
    Passthrough,
    ScalarRandomWalkKalman,
    WFLCTremorFilter,
    build_registry,
)

RATE = 200.0


# --------------------------------------------------------------------------
# The v1 methodological error, demonstrated rather than asserted
# --------------------------------------------------------------------------


def test_ramp_error_is_structural_not_empirical():
    """A scalar random-walk Kalman filter has steady-state ramp error; a
    constant-velocity one does not.

    This is the claim that invalidates the v1 conclusion "the Kalman filter
    lags". The lag came from omitting velocity from the state, not from Kalman
    filtering. With a fair estimator the comparison must be redone.
    """
    t, clean, measured = make_ramp_signal(
        rate_hz=RATE, duration_s=3.0, slope_dps_per_s=100.0, noise_dps=2.0
    )
    settled = slice(int(1.5 * RATE), None)  # ignore initial transient

    v1 = ScalarRandomWalkKalman(process_noise=1e-5, measurement_noise=0.1)
    cv = ConstantVelocityKalman(process_var=1e3, measurement_var=4.0)

    err_v1 = np.mean(np.abs(v1.process(measured, 1 / RATE)[settled] - clean[settled]))
    err_cv = np.mean(np.abs(cv.process(measured, 1 / RATE)[settled] - clean[settled]))

    assert err_v1 > 10.0, f"expected large v1 ramp error, got {err_v1:.3f} dps"
    assert err_cv < 1.0, f"expected near-zero CV ramp error, got {err_cv:.3f} dps"
    assert err_cv < err_v1 / 10.0


def test_constant_velocity_kalman_beats_v1_on_realistic_motion():
    src = SyntheticSource(SyntheticConfig(rate_hz=RATE, duration_s=15.0, seed=1))
    t, clean, measured, _ = src.generate()
    yaw_clean, yaw_measured = clean[:, 2], measured[:, 2]

    e_v1 = rmse(yaw_clean, ScalarRandomWalkKalman().process(yaw_measured, 1 / RATE))
    e_cv = rmse(yaw_clean, ConstantVelocityKalman().process(yaw_measured, 1 / RATE))
    assert e_cv < e_v1


# --------------------------------------------------------------------------
# Nonlinear stage: only an order statistic removes impulses
# --------------------------------------------------------------------------


def test_hampel_removes_spikes_that_linear_filters_only_smear():
    rng = np.random.default_rng(7)
    n = 4000
    t = np.arange(n) / RATE
    clean = 30.0 * np.sin(2 * np.pi * 0.8 * t)
    measured = clean + rng.normal(0, 1.5, n)
    measured[[500, 1200, 2300, 3100]] += np.array([150.0, -160.0, 145.0, -155.0])

    before = count_spikes(measured)
    after_hampel = count_spikes(HampelFilter(window=7).process(measured, 1 / RATE))
    after_linear = count_spikes(
        BiquadLowpass(cutoff_hz=8.0, rate_hz=RATE).process(measured, 1 / RATE)
    )

    assert before >= 4
    assert after_hampel < before / 2, "Hampel must remove the majority of impulses"
    assert after_hampel < after_linear, (
        "the nonlinear stage must outperform the linear one on impulses; "
        f"hampel={after_hampel} linear={after_linear}"
    )


def test_hampel_preserves_clean_signal():
    """The outlier stage must not distort data that contains no outliers."""
    t = np.arange(2000) / RATE
    clean = 40.0 * np.sin(2 * np.pi * 1.2 * t)
    out = HampelFilter(window=7, n_sigmas=3.0).process(clean, 1 / RATE)
    assert rmse(clean, out) < 1.0


# --------------------------------------------------------------------------
# Order matters: the v1 complaint about the 1 euro filter, tested
# --------------------------------------------------------------------------


def test_cascading_buys_noise_rejection_and_pays_in_lag():
    """Second order gives -12 dB/octave instead of -6, but it is not free.

    Measured separately, because a single combined error figure conflates the
    two effects and hides the trade-off that is the subject of RQ1:

    * noise rejection is measured with the device **stationary**, where there
      is no motion to track and the output should ideally be flat;
    * lag is measured while **moving**, against the known clean trajectory.

    The result is the honest version of "just cascade it": cascading really
    does reject more noise, and it really does cost latency. Which side of that
    trade wins is a task-level question, not a signal-level one -- which is why
    the study needs Level 2 at all.
    """
    still = SyntheticSource(
        SyntheticConfig(
            rate_hz=RATE, duration_s=20.0, motion_amplitude_dps=0.0,
            tremor_amplitude_dps=0.0, bias_walk_dps_per_sqrt_s=0.0,
            spike_probability=0.0, packet_loss_probability=0.0,
            white_noise_dps=2.0, seed=3,
        )
    )
    _, still_clean, still_measured, _ = still.generate()

    single_still = OneEuroFilter(1.0, 0.007).process(still_measured[:, 2], 1 / RATE)
    double_still = CascadedOneEuro(1.0, 0.007).process(still_measured[:, 2], 1 / RATE)

    noise_single = np.std(single_still[int(RATE):] - still_clean[int(RATE):, 2])
    noise_double = np.std(double_still[int(RATE):] - still_clean[int(RATE):, 2])
    assert noise_double < noise_single, (
        f"cascade must reject more noise: {noise_double:.3f} vs {noise_single:.3f}"
    )

    moving = SyntheticSource(SyntheticConfig(rate_hz=RATE, duration_s=20.0, seed=3))
    _, clean, measured, _ = moving.generate()
    lag_single = phase_lag_ms(
        clean[:, 2], OneEuroFilter(1.0, 0.007).process(measured[:, 2], 1 / RATE), RATE
    )
    lag_double = phase_lag_ms(
        clean[:, 2], CascadedOneEuro(1.0, 0.007).process(measured[:, 2], 1 / RATE), RATE
    )
    assert lag_double >= lag_single, "cascading cannot reduce lag"


def test_one_euro_cutoff_actually_adapts():
    """Verify the scheduling mechanism is exercised, not pinned at min_cutoff."""
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    t = np.arange(2000) / RATE
    signal = np.concatenate([np.zeros(1000), 200.0 * t[:1000]])

    cutoffs = []
    f.reset()
    for x in signal:
        f.update(float(x), 1 / RATE)
        cutoffs.append(f.last_cutoff)

    assert min(cutoffs) == pytest.approx(1.0, abs=0.2)
    assert max(cutoffs) > 2.0, "cutoff never opened; beta is ineffective"


# --------------------------------------------------------------------------
# Latency: the constraint that makes the problem hard
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Passthrough(),
        lambda: OneEuroFilter(min_cutoff=1.0, beta=0.007),
        lambda: CascadedOneEuro(min_cutoff=1.0, beta=0.007),
        lambda: ConstantVelocityKalman(),
        lambda: AdaptiveKalman(),
        lambda: DoubleExponentialSmoothing(alpha=0.25, gamma=0.1),
    ],
)
def test_candidate_filters_stay_under_perceptual_threshold(factory):
    """RQ1's acceptance criterion: no filter may exceed 75 ms of lag."""
    src = SyntheticSource(SyntheticConfig(rate_hz=RATE, duration_s=20.0, seed=5))
    _, clean, measured, _ = src.generate()
    out = factory().process(measured[:, 2], 1 / RATE)
    lag = phase_lag_ms(clean[:, 2], out, RATE)
    assert abs(lag) <= PERCEPTUAL_LAG_THRESHOLD_MS, f"lag {lag:.1f} ms exceeds threshold"


def test_prediction_can_produce_non_positive_lag():
    """DES extrapolates, so it should not lag more than plain smoothing."""
    t, clean, measured = make_ramp_signal(rate_hz=RATE, duration_s=4.0, noise_dps=1.0)
    des = DoubleExponentialSmoothing(alpha=0.3, gamma=0.15, predict_steps=2.0)
    euro = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    lag_des = phase_lag_ms(clean, des.process(measured, 1 / RATE), RATE)
    lag_euro = phase_lag_ms(clean, euro.process(measured, 1 / RATE), RATE)
    assert lag_des <= lag_euro + 1e-9


def test_stronger_smoothing_costs_lag_monotonically():
    """The jitter/lag trade-off is real and ordered, not a tuning accident."""
    t, clean, measured = make_step_response_signal(rate_hz=RATE, noise_dps=3.0)
    lags = [
        phase_lag_ms(clean, OneEuroFilter(min_cutoff=fc, beta=0.0).process(measured, 1 / RATE), RATE)
        for fc in (8.0, 2.0, 0.5)
    ]
    assert lags[0] <= lags[1] <= lags[2]


# --------------------------------------------------------------------------
# Tremor: the frequency-domain claim
# --------------------------------------------------------------------------


def test_synthetic_generator_places_tremor_in_the_expected_band():
    """The fixture must actually contain what the tests claim to detect."""
    src = SyntheticSource(
        SyntheticConfig(
            rate_hz=RATE, duration_s=30.0, tremor_amplitude_dps=6.0,
            tremor_freq_hz=9.5, tremor_freq_drift_hz=0.0,
            white_noise_dps=0.5, spike_probability=0.0,
            packet_loss_probability=0.0, motion_amplitude_dps=20.0, seed=11,
        )
    )
    _, _, measured, _ = src.generate()
    report = band_report(measured[:, 2], RATE)
    assert 8.0 <= report["tremor_peak_hz"] <= 12.0
    assert report["tremor_8_12Hz"] > 1.0


def test_wflc_attenuates_tremor_band_far_more_than_voluntary_band():
    """The distinguishing property of a notch versus a low-pass.

    A low-pass that reduces tremor also reduces fast voluntary motion. A
    correctly tuned adaptive notch must not.
    """
    src = SyntheticSource(
        SyntheticConfig(
            rate_hz=RATE, duration_s=40.0, tremor_amplitude_dps=8.0,
            tremor_freq_hz=9.5, tremor_freq_drift_hz=0.0,
            white_noise_dps=0.5, spike_probability=0.0,
            packet_loss_probability=0.0, motion_amplitude_dps=40.0, seed=13,
        )
    )
    _, _, measured, _ = src.generate()
    raw = measured[:, 2]

    out = WFLCTremorFilter(
        rate_hz=RATE, f0_hz=9.5, mu_freq=1e-6, mu_amp=1e-2
    ).process(raw, 1 / RATE)

    tremor_att = attenuation_db(raw, out, RATE, (8.0, 12.0))
    voluntary_att = attenuation_db(raw, out, RATE, (0.0, 5.0))

    assert tremor_att > 3.0, f"tremor attenuation only {tremor_att:.1f} dB"
    assert tremor_att > voluntary_att + 3.0, (
        f"not selective: tremor {tremor_att:.1f} dB vs voluntary {voluntary_att:.1f} dB"
    )


def test_wflc_refuses_to_run_when_band_is_above_nyquist():
    """Guards against silently repeating the v1 mistake at 33 Hz."""
    with pytest.raises(ValueError, match="Nyquist"):
        WFLCTremorFilter(rate_hz=20.0, f0_hz=9.0, band_hz=(6.0, 14.0))


# --------------------------------------------------------------------------
# Infrastructure
# --------------------------------------------------------------------------


def test_counter_timebase_handles_wrap_and_detects_loss():
    tb = CounterTimebase(modulus=256, ticks_per_report=3, seconds_per_tick=0.005)
    tb.update(250)
    t, dropped = tb.update(253)
    assert dropped == 0 and t == pytest.approx(0.015)
    t, dropped = tb.update(0)  # wrapped 253 -> 256 == 0
    assert dropped == 0 and t == pytest.approx(0.030)
    t, dropped = tb.update(9)  # delta 9 == three reports' worth
    assert dropped == 2, "two intervening reports must be reported as lost"


def test_timebase_is_uniform_by_construction():
    """The whole point: device time has no jitter even when the host does."""
    tb = CounterTimebase()
    times = [tb.update(c % 256)[0] for c in range(0, 300, 3)]
    dt = np.diff(times)
    assert np.allclose(dt, 0.015)


def test_synthetic_source_reports_dropped_samples_explicitly():
    src = SyntheticSource(
        SyntheticConfig(rate_hz=RATE, duration_s=5.0, packet_loss_probability=0.05, seed=17)
    )
    samples = list(src.samples())
    assert src.stats.samples_dropped > 0
    assert sum(s.dropped_before for s in samples) == src.stats.samples_dropped


def test_every_registry_filter_runs_and_is_finite():
    """No condition may crash or emit NaN -- an incomplete comparison table is
    worse than none."""
    src = SyntheticSource(SyntheticConfig(rate_hz=RATE, duration_s=10.0, seed=19))
    _, _, measured, _ = src.generate()
    raw = measured[:, 2]
    for key, filt in build_registry(rate_hz=RATE).items():
        out = filt.process(raw, 1 / RATE)
        assert out.shape == raw.shape, key
        assert np.all(np.isfinite(out)), f"{key} produced non-finite output"


def test_filters_reset_completely_between_trials():
    """State must not leak across experimental conditions."""
    src = SyntheticSource(SyntheticConfig(rate_hz=RATE, duration_s=5.0, seed=23))
    _, _, measured, _ = src.generate()
    raw = measured[:, 2]
    for key, filt in build_registry(rate_hz=RATE).items():
        first = filt.process(raw, 1 / RATE)
        second = filt.process(raw, 1 / RATE)
        assert np.allclose(first, second), f"{key} leaks state between runs"
