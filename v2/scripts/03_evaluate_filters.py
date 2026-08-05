#!/usr/bin/env python3
"""Level 1 evaluation: score every filter against ground truth.

Runs the full condition set F0..F10 plus the v1 estimator K0 over a common
input, so no algorithm is compared on a different signal from any other. On
synthetic data the ground truth is exact; on a hardware recording it comes from
the mechanical reference trajectory (research design B3).

The output table is the basis for selecting the 3-4 conditions that go forward
to the user study, and is itself a table in Chapter 4.

Usage
-----
    python scripts/03_evaluate_filters.py                    # synthetic
    python scripts/03_evaluate_filters.py --csv data/rec.csv # recording
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

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
    lag_compensated_rmse,
    phase_lag_ms,
    rmse,
    step_response_metrics,
)
from am.analysis.spectral import (  # noqa: E402
    attenuation_db,
    band_report,
    is_stationary,
    noise_parameters,
)
from am.filters import build_registry  # noqa: E402


def _hdr(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def characterise_input(raw: np.ndarray, rate_hz: float) -> None:
    _hdr("A. INPUT CHARACTERISATION -- what is actually in this signal")
    report = band_report(raw, rate_hz)
    print("  Power distribution by band:")
    for key in ("voluntary_0_5Hz", "transition_5_8Hz", "tremor_8_12Hz", "noise_15Hz_up"):
        if key in report:
            print(f"    {key:<20s} {report[key]:6.2f} %")
    if "tremor_peak_hz" in report:
        print(f"    tremor peak          {report['tremor_peak_hz']:6.2f} Hz")
    print()
    print("  Decision rule: if the 8-12 Hz band holds a negligible share of the")
    print("  power, drop WFLC from the study and report that as a finding.")
    print()
    if not is_stationary(raw, rate_hz):
        print("  Allan variance: SKIPPED.")
        print("  This signal contains voluntary motion, and Allan variance")
        print("  characterises sensor noise only. Run it on a dedicated")
        print("  stationary recording (device at rest, several minutes), which")
        print("  is also what gives principled Kalman tuning values.")
        return

    params = noise_parameters(raw, rate_hz, check_stationary=False)
    if params:
        print("  Allan variance characterisation (drives Kalman tuning):")
        print(f"    angle random walk    {params['arw_deg_per_sqrt_hour']:8.3f} deg/sqrt(h)")
        print(f"    bias instability     {params['bias_instability_dps']:8.4f} dps "
              f"at tau = {params['bias_instability_tau_s']:.1f} s")
        print(f"    -> suggested Kalman measurement_var = "
              f"{params['suggested_measurement_var']:.3f}")


def main_table(raw, clean, rate_hz: float, still_slice: slice) -> None:
    _hdr("B. MAIN COMPARISON -- the jitter/lag trade-off")
    print(f"{'id':<5}{'filter':<34}{'RMSE':>8}{'lag-comp':>10}{'lag ms':>9}"
          f"{'noise':>9}{'dB':>8}{'spikes':>8}{'<75ms':>7}")
    print("-" * 100)

    rows = []
    for key, filt in build_registry(rate_hz=rate_hz).items():
        out = filt.process(raw, 1.0 / rate_hz)
        lag = phase_lag_ms(clean, out, rate_hz)
        residual = out - clean
        noise = float(np.std(residual[still_slice]))
        e_raw = float(np.var(raw - clean))
        e_out = float(np.var(residual))
        db = 10.0 * np.log10(e_raw / e_out) if e_out > 0 else 0.0
        row = dict(
            key=key, name=filt.name, rmse=rmse(clean, out),
            lagcomp=lag_compensated_rmse(clean, out, rate_hz), lag=lag,
            noise=noise, db=db, spikes=count_spikes(out),
            ok=abs(lag) <= PERCEPTUAL_LAG_THRESHOLD_MS,
        )
        rows.append(row)
        print(f"{key:<5}{filt.name[:33]:<34}{row['rmse']:8.3f}{row['lagcomp']:10.3f}"
              f"{lag:9.1f}{noise:9.3f}{db:8.2f}{row['spikes']:8d}"
              f"{'yes' if row['ok'] else 'NO':>7}")

    print("-" * 100)
    print("  RMSE      total error, conflates lag with distortion")
    print("  lag-comp  error after removing the lag: pure distortion")
    print("  noise     residual std over the stationary interval (RQ1 jitter)")
    print("  dB        error-energy reduction against ground truth; negative")
    print("            means the filter made the estimate worse than the raw signal")

    ok = [r for r in rows if r["ok"] and r["key"] != "F0"]
    if ok:
        best_noise = min(ok, key=lambda r: r["noise"])
        best_error = min(ok, key=lambda r: r["lagcomp"])
        print()
        print(f"  lowest jitter within the 75 ms budget : {best_noise['key']} "
              f"({best_noise['name']}) -- {best_noise['noise']:.3f} dps")
        print(f"  lowest distortion within the budget   : {best_error['key']} "
              f"({best_error['name']}) -- {best_error['lagcomp']:.3f}")


def ramp_demonstration(rate_hz: float) -> None:
    _hdr("C. THE v1 METHODOLOGICAL ERROR, DEMONSTRATED")
    t, clean, measured = make_ramp_signal(rate_hz=rate_hz, duration_s=3.0,
                                          slope_dps_per_s=100.0, noise_dps=2.0)
    settled = slice(int(1.5 * rate_hz), None)
    print("  Constant-velocity input (a hand moving at steady angular rate).")
    print("  Mean absolute steady-state error:")
    print()
    from am.filters import ConstantVelocityKalman, ScalarRandomWalkKalman

    for label, filt in [
        ("v1 scalar random-walk Kalman", ScalarRandomWalkKalman(1e-5, 0.1)),
        ("constant-velocity Kalman", ConstantVelocityKalman(1e3, 4.0)),
    ]:
        out = filt.process(measured, 1.0 / rate_hz)
        err = float(np.mean(np.abs(out[settled] - clean[settled])))
        print(f"    {label:<32s} {err:9.3f} dps")
    print()
    print("  The v1 filter's error is structural: its process model asserts the")
    print("  signal is constant, so it cannot track a ramp without steady-state")
    print("  error. 'The Kalman filter lags' described that modelling choice,")
    print("  not Kalman filtering. The comparison has to be redone.")


def step_demonstration(rate_hz: float) -> None:
    _hdr("D. STEP RESPONSE -- the cost side of the trade-off")
    t, clean, measured = make_step_response_signal(rate_hz=rate_hz, duration_s=2.0,
                                                  step_time_s=0.5, step_dps=100.0,
                                                  noise_dps=3.0)
    print(f"{'id':<5}{'filter':<34}{'rise ms':>10}{'overshoot %':>13}{'settling ms':>13}")
    print("-" * 100)
    for key, filt in build_registry(rate_hz=rate_hz).items():
        out = filt.process(measured, 1.0 / rate_hz)
        m = step_response_metrics(t, out, 0.5, 100.0)
        print(f"{key:<5}{filt.name[:33]:<34}{m['rise_time_ms']:10.1f}"
              f"{m['overshoot_percent']:13.1f}{m['settling_time_ms']:13.1f}")
    print("-" * 100)
    print("  Overshoot is where predictive filters pay: extrapolation that")
    print("  removes lag during steady motion overshoots at reversals.")


def selectivity_check(raw, rate_hz: float) -> None:
    _hdr("E. SELECTIVITY -- is the tremor stage a notch or a disguised low-pass?")
    print(f"{'id':<5}{'filter':<34}{'8-12Hz dB':>12}{'0-5Hz dB':>11}{'selectivity':>13}")
    print("-" * 100)
    for key, filt in build_registry(rate_hz=rate_hz).items():
        if key in ("F0", "K0"):
            continue
        out = filt.process(raw, 1.0 / rate_hz)
        tremor = attenuation_db(raw, out, rate_hz, (8.0, 12.0))
        voluntary = attenuation_db(raw, out, rate_hz, (0.0, 5.0))
        print(f"{key:<5}{filt.name[:33]:<34}{tremor:12.2f}{voluntary:11.2f}"
              f"{tremor - voluntary:13.2f}")
    print("-" * 100)
    print("  Selectivity = tremor attenuation minus voluntary attenuation.")
    print("  High selectivity means the filter removes tremor specifically.")
    print("  Near zero means it is simply attenuating everything -- which a")
    print("  plain low-pass already does, so the extra stage earns nothing.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", help="recording to evaluate instead of synthetic data")
    ap.add_argument("--rate", type=float, default=200.0)
    ap.add_argument("--duration", type=float, default=30.0)
    args = ap.parse_args()

    if args.csv:
        data = np.genfromtxt(args.csv, delimiter=",", names=True)
        raw = np.asarray(data["gyro_z"], dtype=np.float64)
        clean = None
        print("NOTE: a recording has no ground truth. Metrics requiring a")
        print("      reference are unavailable; supply a mechanical reference")
        print("      trajectory (research design B3) for those.")
        characterise_input(raw, args.rate)
        return 0

    print(f"Synthetic evaluation: {args.duration:.0f} s at {args.rate:.0f} Hz")
    src = SyntheticSource(
        SyntheticConfig(rate_hz=args.rate, duration_s=args.duration, seed=101)
    )
    _, clean_all, measured_all, _ = src.generate()
    raw, clean = measured_all[:, 2], clean_all[:, 2]

    # First second is treated as the stationary interval for the jitter metric.
    still = slice(0, int(args.rate))

    characterise_input(raw, args.rate)
    main_table(raw, clean, args.rate, still)
    ramp_demonstration(args.rate)
    step_demonstration(args.rate)
    selectivity_check(raw, args.rate)

    _hdr("NEXT STEPS")
    print("  1. Run scripts/01_verify_acquisition.py with the controller attached.")
    print("  2. Record the three conditions (stationary / pointing / flick).")
    print("  3. Re-run this script on real data with --csv.")
    print("  4. Pick 3-4 conditions for the user study on the evidence above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
