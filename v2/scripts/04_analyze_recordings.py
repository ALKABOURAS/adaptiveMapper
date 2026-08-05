#!/usr/bin/env python3
"""Spectral analysis of the three reference recordings.

Answers the questions that decide the condition set:

1. Does physiological tremor exist in this signal, and is it a real peak rather
   than the arbitrary maximum of a flat band?
2. Is it biological, or an instrument artefact? Settled by comparing against the
   stationary recording, where no hand is present.
3. Does its frequency move? If it does, a fixed notch is the wrong tool and the
   adaptive one is justified. If it does not, WFLC has to earn its place against
   a fixed notch.
4. What are the sensor's actual noise parameters, for principled Kalman tuning
   instead of guessed covariances?

Usage
-----
    python scripts/04_analyze_recordings.py
    python scripts/04_analyze_recordings.py --dir data/recordings --stamp 20260805_215734
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from am.analysis.spectral import (  # noqa: E402
    band_power,
    find_tremor_peak,
    find_tremor_peak_robust,
    is_stationary,
    noise_parameters,
    tremor_prominence,
    welch_psd,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ["stationary", "pointing", "flick"]
AXES = [("gyro_x", "roll"), ("gyro_y", "pitch"), ("gyro_z", "yaw")]


def load(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    t = np.asarray(data["t_device"], dtype=np.float64)
    gyro = np.stack(
        [np.asarray(data[name], dtype=np.float64) for name, _ in AXES], axis=1
    )
    rate = 1.0 / float(np.median(np.diff(t)))
    return t, gyro, rate


def find_recordings(directory: Path, stamp: str | None) -> dict[str, Path]:
    found = {}
    for condition in CONDITIONS:
        pattern = f"*{stamp}*_{condition}.csv" if stamp else f"*_{condition}.csv"
        matches = sorted(directory.glob(pattern))
        if matches:
            found[condition] = matches[-1]
    return found


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--stamp", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    directory = Path(args.dir) if args.dir else PROJECT_ROOT / "data" / "recordings"
    recordings = find_recordings(directory, args.stamp)
    if not recordings:
        print(f"No recordings found in {directory}. Run scripts/02_record_raw.py first.")
        return 1

    loaded = {c: load(p) for c, p in recordings.items()}
    print(f"Loaded from {directory}:")
    for condition, path in recordings.items():
        t, gyro, rate = loaded[condition]
        print(f"  {condition:<12} {len(t):>7} samples  {t[-1]:6.1f} s  {rate:6.1f} Hz")

    # ---------------------------------------------------------------- Q1 + Q2
    section("A. IS THERE A TREMOR PEAK, AND IS IT BIOLOGICAL?")
    print("  Prominence is the peak height in 8-12 Hz over the 6-7 / 13-15 Hz")
    print("  shoulders. Band power alone cannot answer this: the maximum of a")
    print("  flat spectrum inside 8-12 Hz is still a frequency in 8-12 Hz.")
    print()
    print(f"  {'condition':<12}{'axis':<8}{'peak Hz':>9}{'prominence':>12}"
          f"{'band power':>14}{'verdict':>12}")
    print("  " + "-" * 68)

    band_powers: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        if condition not in loaded:
            continue
        _, gyro, rate = loaded[condition]
        band_powers[condition] = {}
        for index, (_, axis) in enumerate(AXES):
            freqs, psd = welch_psd(gyro[:, index], rate, nperseg=2048)
            mask = (freqs >= 8.0) & (freqs <= 12.0)
            peak_hz = float(freqs[mask][np.argmax(psd[mask])])
            prominence = tremor_prominence(freqs, psd)
            power = band_power(freqs, psd, (8.0, 12.0))
            band_powers[condition][axis] = power
            verdict = (
                "REAL PEAK" if prominence > 3 else "weak" if prominence > 1.5 else "flat"
            )
            print(f"  {condition:<12}{axis:<8}{peak_hz:>9.2f}{prominence:>11.2f}x"
                  f"{power:>14.6f}{verdict:>12}")
        print()

    if "stationary" in band_powers and "pointing" in band_powers:
        section("B. THE DECISIVE COMPARISON")
        print("  The stationary recording has no hand in it. Any 8-12 Hz content")
        print("  there is instrumental: sensor resonance, bench vibration, or the")
        print("  noise floor. Comparing absolute power settles whether the peak")
        print("  seen while pointing is biological.")
        print()
        print(f"  {'axis':<8}{'stationary':>14}{'pointing':>14}{'ratio':>12}"
              f"{'instrumental':>15}")
        print("  " + "-" * 63)
        for _, axis in AXES:
            still = band_powers["stationary"][axis]
            point = band_powers["pointing"][axis]
            ratio = point / still if still > 0 else float("inf")
            share = still / point * 100 if point > 0 else 0.0
            print(f"  {axis:<8}{still:>14.6f}{point:>14.6f}{ratio:>11.0f}x"
                  f"{share:>14.3f}%")
        print()
        worst = max(
            band_powers["stationary"][a] / band_powers["pointing"][a] * 100
            for _, a in AXES
            if band_powers["pointing"][a] > 0
        )
        if worst < 5.0:
            print(f"  -> Instrumental contribution is at most {worst:.2f} % of the")
            print("     tremor-band power seen while pointing. The peak is")
            print("     biological. Physiological tremor is present and measurable.")
        else:
            print(f"  -> Instrumental contribution reaches {worst:.1f} %. Not safe to")
            print("     attribute the peak to tremor. Investigate the bench setup.")

    # ------------------------------------------------------------- Q3: shift
    section("C. DOES THE TREMOR FREQUENCY MOVE?")
    print("  A fixed sensor resonance cannot move. A hand-arm resonance moves with")
    print("  grip force and limb loading. This is what decides adaptive versus")
    print("  fixed narrowband suppression.")
    print()
    print("  Peaks are found after removing the broadband trend, must be interior")
    print("  local maxima, and must appear at the same frequency under four")
    print("  different Welch windows. A single window can manufacture a peak.")
    print()
    print(f"  {'condition':<12}{'peak Hz':>10}{'spread':>9}{'prominence':>13}"
          f"{'windows':>9}{'verdict':>13}")
    print("  " + "-" * 66)
    peaks = {}
    stable_peaks = {}
    for condition in CONDITIONS:
        if condition not in loaded:
            continue
        _, gyro, rate = loaded[condition]
        r = find_tremor_peak_robust(gyro[:, 2], rate)
        if not r["n_valid"]:
            print(f"  {condition:<12}{'none':>10}{'--':>9}{'--':>13}{0:>9}{'no peak':>13}")
            continue
        verdict = "STABLE" if r["stable"] else "UNSTABLE"
        peaks[condition] = r["peak_hz"]
        if r["stable"]:
            stable_peaks[condition] = r["peak_hz"]
        print(f"  {condition:<12}{r['peak_hz']:>10.2f}{r['spread_hz']:>8.2f}Hz"
              f"{r['prominence']:>12.2f}x{int(r['n_valid']):>9}{verdict:>13}")

    print()
    unstable = [c for c in peaks if c not in stable_peaks]
    if unstable:
        print(f"  {', '.join(unstable)}: the peak frequency depends on the window,")
        print("  so no tremor frequency can be quoted for it. In fast movement the")
        print("  spectrum is dominated by harmonics of the repetitive motion, and")
        print("  which harmonic wins depends on the resolution. This is a limit of")
        print("  the measurement, not a property of the hand.")
        print()

    # The stationary peak is instrumental, not biological (section B).
    biological = {k: v for k, v in stable_peaks.items() if k != "stationary"}
    if len(biological) >= 2:
        spread = max(biological.values()) - min(biological.values())
        print(f"  Shift across held-in-hand conditions: {spread:.2f} Hz")
        if spread > 1.0:
            print("  -> The tremor frequency moves. A fixed notch centred on one")
            print("     condition is mistuned for the others: WFLC is justified.")
        else:
            print("  -> The shift is under 1 Hz. WFLC is NOT yet justified; it must")
            print("     beat a fixed notch on the task metrics to earn its place.")
    elif len(biological) == 1:
        condition, value = next(iter(biological.items()))
        print(f"  Only one condition yields a stable tremor frequency: "
              f"{condition} at {value:.2f} Hz.")
        print()
        print("  -> There is therefore NO evidence here that the tremor frequency")
        print("     moves, and so no empirical basis yet for preferring an adaptive")
        print("     notch over a fixed one. F9 (WFLC) must be compared against a")
        print(f"     fixed notch at {value:.1f} Hz and win on the task metrics.")
        print("     Tremor itself is confirmed real and biological (section B);")
        print("     it is only the *adaptive* part that remains unjustified.")

    # -------------------------------------------------------- Q4: Allan noise
    section("D. SENSOR NOISE PARAMETERS -> KALMAN TUNING")
    if "stationary" not in loaded:
        print("  No stationary recording; cannot characterise the sensor.")
    else:
        _, gyro, rate = loaded["stationary"]
        if not is_stationary(gyro[:, 2], rate):
            print("  ! The stationary recording does not look stationary. Redo it.")
        print("  Allan variance on the stationary recording. These replace guessed")
        print("  covariances with measured ones.")
        print()
        print(f"  {'axis':<8}{'ARW deg/sqrt(h)':>18}{'bias instab dps':>18}"
              f"{'tau s':>9}{'measurement_var':>18}")
        print("  " + "-" * 71)
        variances = []
        for index, (_, axis) in enumerate(AXES):
            params = noise_parameters(gyro[:, index], rate, check_stationary=False)
            variances.append(params["suggested_measurement_var"])
            print(f"  {axis:<8}{params['arw_deg_per_sqrt_hour']:>18.3f}"
                  f"{params['bias_instability_dps']:>18.5f}"
                  f"{params['bias_instability_tau_s']:>9.1f}"
                  f"{params['suggested_measurement_var']:>18.5f}")
        print()
        print(f"  Use measurement_var = {np.mean(variances):.5f} in build_registry().")
        print(f"  Stationary noise std: "
              f"{', '.join(f'{gyro[:, i].std():.4f}' for i in range(3))} dps")

    if args.out or True:
        out_path = (
            Path(args.out) if args.out else PROJECT_ROOT / "data" / "spectral_analysis.png"
        )
        try:
            make_figure(loaded, peaks, out_path)
        except ImportError:
            print("\n  (matplotlib not available; skipping the figure)")

    return 0


def make_figure(loaded: dict, peaks: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    colours = {"stationary": "tab:green", "pointing": "tab:blue", "flick": "tab:red"}

    # (a) PSD, all three conditions, log-log.
    ax = axes[0, 0]
    for condition, colour in colours.items():
        if condition not in loaded:
            continue
        _, gyro, rate = loaded[condition]
        freqs, psd = welch_psd(gyro[:, 2], rate, nperseg=2048)
        ax.loglog(freqs[1:], psd[1:], lw=1.2, color=colour, label=condition)
    ax.axvspan(8, 12, alpha=0.2, color="tab:orange", label="tremor band")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dps$^2$/Hz)")
    ax.set_title("(a) Power spectral density, yaw")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # (b) Normalised, to expose the peak rather than the overall level.
    ax = axes[0, 1]
    for condition, colour in colours.items():
        if condition not in loaded:
            continue
        _, gyro, rate = loaded[condition]
        freqs, psd = welch_psd(gyro[:, 2], rate, nperseg=4096)
        window = (freqs >= 3) & (freqs <= 20)
        f_w, p_w = freqs[window], psd[window]
        ax.plot(f_w, p_w / p_w.max(), lw=1.4, color=colour, label=condition)
        if condition in peaks:
            ax.axvline(peaks[condition], color=colour, ls=":", lw=1.2)
    ax.axvspan(8, 12, alpha=0.2, color="tab:orange")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalised PSD")
    ax.set_title("(b) The peak moves with limb loading")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) Absolute tremor-band power: the biological argument.
    ax = axes[1, 0]
    names, values = [], []
    for condition in CONDITIONS:
        if condition not in loaded:
            continue
        _, gyro, rate = loaded[condition]
        freqs, psd = welch_psd(gyro[:, 2], rate, nperseg=2048)
        names.append(condition)
        values.append(band_power(freqs, psd, (8.0, 12.0)))
    ax.bar(names, values, color=[colours[n] for n in names], alpha=0.85)
    ax.set_yscale("log")
    ax.set_ylabel("8-12 Hz band power (dps$^2$)")
    ax.set_title("(c) Tremor-band power: hand versus table")
    ax.grid(alpha=0.3, axis="y", which="both")
    for i, value in enumerate(values):
        ax.text(i, value * 1.5, f"{value:.2e}", ha="center", fontsize=8)

    # (d) Allan deviation.
    ax = axes[1, 1]
    if "stationary" in loaded:
        from am.analysis.spectral import allan_deviation

        _, gyro, rate = loaded["stationary"]
        for index, (_, axis) in enumerate(AXES):
            taus, adev = allan_deviation(gyro[:, index], rate)
            ax.loglog(taus, adev, lw=1.3, label=axis)
        ax.set_xlabel(r"Averaging time $\tau$ (s)")
        ax.set_ylabel(r"Allan deviation (dps)")
        ax.set_title("(d) Allan deviation: sensor noise characterisation")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        "Spectral characterisation of the Joy-Con gyroscope at 200 Hz", fontsize=13
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\n  Figure written to {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
