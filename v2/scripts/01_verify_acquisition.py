#!/usr/bin/env python3
"""Empirically verify the acquisition path. Run this first, with the device.

This script does not assume the report layout is what the documentation says.
It *measures* it, and produces the first figure of Chapter 4.

Questions answered
------------------
1. Does input report 0x30 really carry three distinct IMU frames, or is the
   claim of ~200 Hz unfounded?
2. Which temporal order are the three frames in? Determined by choosing the
   ordering that minimises signal discontinuity across report boundaries -- no
   guessing, no reliance on third-party documentation.
3. What is the true sample rate, from the device counter rather than the host
   clock?
4. What fraction of reports is lost in transit?
5. How much of the v1 timing jitter was transport and how much was host
   scheduling?

Usage
-----
    python scripts/01_verify_acquisition.py --device left --duration 20

Move the controller in smooth sweeps during the recording. Frame-order
detection needs actual motion; a stationary device gives no discontinuity to
minimise, and the script will say so rather than report a coin flip.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from am.acquisition.joycon import (  # noqa: E402
    GYRO_DPS_PER_LSB,
    IMU_BLOCK_OFFSET,
    IMU_FRAME_SIZE,
    JoyConSource,
)
from am.core.timebase import CounterTimebase, uniformity  # noqa: E402

FRAME_STRUCT = struct.Struct("<6h")


def collect_reports(source: JoyConSource, duration_s: float) -> list[tuple[float, bytes]]:
    """Capture raw reports with host arrival times, doing no interpretation."""
    print(f"Recording for {duration_s:.0f} s -- move the controller in smooth sweeps.")
    out: list[tuple[float, bytes]] = []
    start = time.perf_counter()
    while time.perf_counter() - start < duration_s:
        report = source.read_report()
        if report is not None:
            out.append((time.perf_counter() - start, report))
    print(f"Captured {len(out)} reports.\n")
    return out


def decode_frames(report: bytes) -> np.ndarray:
    """Return the three gyro frames of a report, shape (3, 3), in dps."""
    frames = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        off = IMU_BLOCK_OFFSET + i * IMU_FRAME_SIZE
        _, _, _, gx, gy, gz = FRAME_STRUCT.unpack_from(report, off)
        frames[i] = np.array([gx, gy, gz], dtype=np.float64) * GYRO_DPS_PER_LSB
    return frames


def q1_frames_are_distinct(reports) -> np.ndarray:
    """Are the three frames genuinely different samples, or duplicates?"""
    print("=" * 72)
    print("Q1  Does report 0x30 carry three distinct IMU frames?")
    print("=" * 72)

    all_frames = np.stack([decode_frames(r) for _, r in reports])  # (N, 3, 3)

    d01 = np.abs(all_frames[:, 0] - all_frames[:, 1]).mean()
    d12 = np.abs(all_frames[:, 1] - all_frames[:, 2]).mean()
    d02 = np.abs(all_frames[:, 0] - all_frames[:, 2]).mean()
    identical = int(np.sum(np.all(all_frames[:, 0] == all_frames[:, 1], axis=1)))

    print(f"  mean |frame0 - frame1| = {d01:8.4f} dps")
    print(f"  mean |frame1 - frame2| = {d12:8.4f} dps")
    print(f"  mean |frame0 - frame2| = {d02:8.4f} dps")
    print(f"  reports where frame0 == frame1 exactly: {identical}/{len(reports)}")

    if d01 < 1e-9 and d12 < 1e-9:
        print("\n  RESULT: frames are duplicates. The 200 Hz claim does NOT hold.")
        print("          Report this as a negative finding and keep 60 Hz.")
    else:
        # Adjacent frames (5 ms apart) should differ less than the outer pair
        # (10 ms apart) for a smoothly moving signal.
        ordered = d02 > max(d01, d12)
        print("\n  RESULT: three distinct samples confirmed.")
        print(f"          Effective IMU rate is 3x the report rate.")
        print(
            f"          Temporal spacing consistent with sequential sampling: "
            f"{'yes' if ordered else 'no -- inspect manually'}"
        )
    print()
    return all_frames


ORDERS: dict[str, tuple[int, int, int]] = {
    "OLDEST_FIRST": (0, 1, 2),
    "NEWEST_FIRST": (2, 1, 0),
}


def reconstruct(all_frames: np.ndarray, order: str, axis: int = 2) -> np.ndarray:
    """Flatten reports into one sample stream under a given frame ordering."""
    return all_frames[:, list(ORDERS[order]), axis].reshape(-1)


def report_rate_artifact_db(
    all_frames: np.ndarray, order: str, imu_rate_hz: float, report_rate_hz: float
) -> float:
    """Excess power at the report rate, in dB above the local spectral floor.

    The decisive test of frame ordering, and much sharper than total variation.

    If the three frames are stitched together in the wrong temporal order, every
    report boundary inserts an identical small discontinuity. Identical, evenly
    spaced discontinuities are a *periodic* disturbance whose fundamental sits
    exactly at the report rate. So the wrong ordering does not merely look
    rougher -- it plants a spectral line at a known, predictable frequency that
    has no physical counterpart in hand motion.

    Hand movement has no mechanism for producing energy at 67 Hz. Any peak there
    is an artefact of reconstruction, and the ordering that minimises it is the
    correct one.
    """
    from am.analysis.spectral import welch_psd

    signal = reconstruct(all_frames, order)
    if signal.size < 512:
        return float("nan")

    freqs, psd = welch_psd(signal, imu_rate_hz)

    # Narrow band around the report rate versus a nearby baseline that excludes
    # it, so the comparison is against the local noise floor rather than the
    # whole spectrum.
    half_width = 3.0
    peak = (freqs >= report_rate_hz - half_width) & (freqs <= report_rate_hz + half_width)
    floor = (
        (freqs >= report_rate_hz - 20.0)
        & (freqs <= report_rate_hz + 20.0)
        & ~peak
        & (freqs > 15.0)
    )
    if not np.any(peak) or not np.any(floor):
        return float("nan")

    return float(10.0 * np.log10(psd[peak].max() / np.median(psd[floor])))


def q2_frame_order(
    all_frames: np.ndarray, imu_rate_hz: float, report_rate_hz: float
) -> str:
    """Determine frame ordering from the report-rate reconstruction artefact."""
    print("=" * 72)
    print("Q2  Which temporal order are the three frames in?")
    print("=" * 72)

    def total_variation(order: str) -> float:
        seq = all_frames[:, list(ORDERS[order]), :].reshape(-1, 3)
        return float(np.abs(np.diff(seq, axis=0)).sum())

    print(f"  Report rate measured at {report_rate_hz:.1f} Hz. A wrong ordering")
    print(f"  plants a spectral line there; hand motion cannot.")
    print()
    print(f"  {'ordering':<16}{'total variation':>18}{'artefact @ report rate':>26}")
    print("  " + "-" * 60)

    scores = {}
    for name in ORDERS:
        tv = total_variation(name)
        artifact = report_rate_artifact_db(
            all_frames, name, imu_rate_hz, report_rate_hz
        )
        scores[name] = (tv, artifact)
        print(f"  {name:<16}{tv:18.1f}{artifact:23.1f} dB")

    tv_ratio = max(s[0] for s in scores.values()) / max(
        min(s[0] for s in scores.values()), 1e-12
    )
    artifacts = {k: v[1] for k, v in scores.items()}
    winner = min(artifacts, key=lambda k: artifacts[k])
    margin = max(artifacts.values()) - min(artifacts.values())

    print()
    if tv_ratio < 1.02 or margin < 3.0:
        print("  RESULT: INCONCLUSIVE -- the two orderings are too close.")
        print("          The device was probably too still. Re-run while moving it.")
        return "inconclusive"

    print(f"  RESULT: {winner}")
    print(f"          Artefact is {margin:.1f} dB lower than the alternative,")
    print(f"          and total variation {(tv_ratio - 1) * 100:.0f}% smoother.")
    print(f"          Set FrameOrder.{winner} in the driver.")
    if artifacts[winner] > 6.0:
        print()
        print(f"  ! Even the best ordering leaves a {artifacts[winner]:.1f} dB residual")
        print("    at the report rate. The 5 ms inter-frame spacing may not be")
        print("    exact. Worth investigating before spectral analysis above 50 Hz;")
        print("    it does not affect the 0-15 Hz bands this study cares about.")
    print()
    return winner


def q3_true_sample_rate(reports) -> dict:
    """Compare the hardware counter time base against the host clock."""
    print("=" * 72)
    print("Q3  What is the true sample rate, and how jittery is the host clock?")
    print("=" * 72)

    tb = CounterTimebase()
    device_times, dropped_total, counter_deltas = [], 0, []
    prev_counter = None

    for _, report in reports:
        counter = report[1]
        if prev_counter is not None:
            counter_deltas.append((counter - prev_counter) % 256)
        prev_counter = counter
        t_report, dropped = tb.update(counter)
        dropped_total += dropped
        device_times.append(t_report)

    host_times = np.array([t for t, _ in reports])
    device_times = np.array(device_times)

    host_u = uniformity(np.diff(host_times))
    dev_u = uniformity(np.diff(device_times))

    deltas = np.array(counter_deltas)
    modal_delta = int(np.bincount(deltas).argmax()) if deltas.size else 0

    print(f"  modal counter increment per report : {modal_delta}")
    print(f"  (3 confirms one tick per IMU frame; 1 would mean one per report)")
    print(f"  counter resyncs                    : {tb.resync_count}")
    print()
    print("  HOST clock (what v1 used as its time base):")
    print(f"    report dt   = {host_u['mean_ms']:.2f} +- {host_u['std_ms']:.2f} ms")
    print(f"    jitter CV   = {host_u['cv_percent']:.1f} %")
    print(f"    report rate = {host_u['rate_hz']:.1f} Hz")
    print()
    print("  DEVICE counter (what v2 uses):")
    print(f"    report dt   = {dev_u['mean_ms']:.2f} +- {dev_u['std_ms']:.2f} ms")
    print(f"    jitter CV   = {dev_u['cv_percent']:.1f} %")
    print()

    imu_rate = dev_u["rate_hz"] * 3
    print(f"  v1 effective IMU rate : {host_u['rate_hz']:8.1f} Hz  "
          f"(Nyquist {host_u['rate_hz'] / 2:.1f} Hz)")
    print(f"  v2 effective IMU rate : {imu_rate:8.1f} Hz  (Nyquist {imu_rate / 2:.1f} Hz)")
    print(f"  improvement           : {imu_rate / max(host_u['rate_hz'], 1e-9):8.1f}x")
    print()
    print(f"  8-12 Hz tremor band observable at v1 rate: "
          f"{'yes' if host_u['rate_hz'] / 2 > 12 else 'NO -- above Nyquist'}")
    print(f"  8-12 Hz tremor band observable at v2 rate: "
          f"{'yes' if imu_rate / 2 > 12 else 'NO'}")
    print()

    total_expected = len(reports) + dropped_total
    loss = dropped_total / total_expected * 100 if total_expected else 0.0
    print(f"  reports received : {len(reports)}")
    print(f"  reports lost     : {dropped_total}  ({loss:.2f} %)")
    print("  (Bluetooth interference appears HERE, as loss -- not as a spectral")
    print("   line in the gyro signal. There is nothing to notch.)")
    print()

    return {
        "host": host_u,
        "device": dev_u,
        "imu_rate_hz": imu_rate,
        "loss_percent": loss,
        "modal_counter_delta": modal_delta,
    }


def make_figure(reports, all_frames, result, order: str, out_path: Path) -> None:
    """Produce the Chapter 4 acquisition figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    good = order if order in ORDERS else "OLDEST_FIRST"
    bad = "NEWEST_FIRST" if good == "OLDEST_FIRST" else "OLDEST_FIRST"
    imu_rate = result["imu_rate_hz"]
    report_rate = result["device"]["rate_hz"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # (a) Correct vs incorrect frame ordering on the same reports.
    #
    # Plotting the wrong ordering alongside the right one is the point of this
    # panel: the sawtooth it produces has a period of exactly one report, which
    # is the time-domain face of the spectral line in panel (d).
    ax = axes[0, 0]
    n_show = min(60, len(all_frames))
    window = all_frames[:n_show]
    seq_good = window[:, list(ORDERS[good]), 2].reshape(-1)
    seq_bad = window[:, list(ORDERS[bad]), 2].reshape(-1)
    t = np.arange(seq_good.size) / imu_rate
    ax.plot(t, seq_bad, "-", lw=1.0, alpha=0.55, color="tab:red",
            label=f"{bad} (wrong): sawtooth at {report_rate:.0f} Hz")
    ax.plot(t, seq_good, "-", lw=1.4, color="tab:blue",
            label=f"{good} (correct), {imu_rate:.0f} Hz")
    ax.plot(t[::3], window[:, 0, 2], "o", ms=4, color="tab:orange",
            label="v1: one frame per report")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Yaw rate (dps)")
    ax.set_title("(a) Frame ordering determines reconstruction")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (b) inter-arrival jitter, host vs device.
    ax = axes[0, 1]
    host_dt = np.diff([t for t, _ in reports]) * 1e3
    ax.hist(host_dt, bins=50, alpha=0.75, color="tab:red")
    ax.axvline(result["device"]["mean_ms"], color="tab:blue", ls="--", lw=2,
               label=f"device counter ({result['device']['mean_ms']:.1f} ms, fixed)")
    ax.set_xlabel("Report inter-arrival time (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"(b) Host clock jitter: CV = {result['host']['cv_percent']:.1f} %")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) observable bandwidth.
    ax = axes[1, 0]
    v1_nyq = result["host"]["rate_hz"] / 2
    v2_nyq = result["imu_rate_hz"] / 2
    ax.barh(["v1", "v2"], [v1_nyq, v2_nyq], color=["tab:red", "tab:blue"], alpha=0.8)
    ax.axvspan(8, 12, alpha=0.25, color="tab:orange", label="tremor band 8-12 Hz")
    ax.axvspan(0, 5, alpha=0.2, color="tab:green", label="voluntary motion 0-5 Hz")
    ax.set_xlabel("Observable bandwidth, Nyquist limit (Hz)")
    ax.set_title("(c) The tremor band was not observable in v1")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")

    # (d) spectra, both orderings plus the v1 rate.
    ax = axes[1, 1]
    from am.analysis.spectral import welch_psd

    yaw_good = reconstruct(all_frames, good)
    yaw_bad = reconstruct(all_frames, bad)
    yaw_v1 = all_frames[:, 0, 2]
    if yaw_bad.size > 256:
        f, p = welch_psd(yaw_bad, imu_rate)
        ax.semilogy(f, p, lw=1.0, alpha=0.6, color="tab:red", label=f"{bad} (wrong)")
    if yaw_good.size > 256:
        f, p = welch_psd(yaw_good, imu_rate)
        ax.semilogy(f, p, lw=1.3, color="tab:blue", label=f"{good} (correct)")
    if yaw_v1.size > 128:
        f, p = welch_psd(yaw_v1, report_rate)
        ax.semilogy(f, p, lw=1.0, alpha=0.7, color="tab:orange",
                    label=f"v1 @ {report_rate:.0f} Hz")
    ax.axvline(report_rate, color="k", ls=":", lw=1.2,
               label=f"report rate {report_rate:.0f} Hz")
    ax.axvspan(8, 12, alpha=0.25, color="tab:orange")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dps$^2$/Hz)")
    ax.set_title("(d) The wrong ordering plants a line at the report rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Acquisition verification: Joy-Con input report 0x30", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Figure written to {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="left", choices=["left", "right", "pro"])
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument(
        "--out",
        default=None,
        help="output figure path; defaults to v2/data/ regardless of cwd",
    )
    args = ap.parse_args()

    # Resolve relative to the project, not the working directory. IDEs commonly
    # run a script with its own directory as cwd, which previously scattered
    # output into v2/scripts/data/.
    project_root = Path(__file__).resolve().parents[1]
    out_path = (
        Path(args.out)
        if args.out
        else project_root / "data" / "acquisition_verification.png"
    )

    source = JoyConSource(device_type=args.device)
    if not source.open():
        print(f"No {args.device} controller found. Pair it over Bluetooth first.")
        return 1

    try:
        reports = collect_reports(source, args.duration)
    finally:
        source.close()

    if len(reports) < 50:
        print(f"Only {len(reports)} reports captured -- not enough. Check the pairing.")
        return 1

    all_frames = q1_frames_are_distinct(reports)
    # Timing is measured first: the frame-order test needs the report rate, so
    # that it knows which frequency the reconstruction artefact would land on.
    result = q3_true_sample_rate(reports)
    order = q2_frame_order(
        all_frames, result["imu_rate_hz"], result["device"]["rate_hz"]
    )

    make_figure(reports, all_frames, result, order, out_path)

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  frame order            : {order}")
    print(f"  report rate (device)   : {result['device']['rate_hz']:.1f} Hz")
    print(f"  IMU rate, v2           : {result['imu_rate_hz']:.1f} Hz")
    print(f"  IMU rate, v1 ceiling   : {result['host']['rate_hz']:.1f} Hz "
          f"(one frame per report)")
    print(f"  host jitter CV         : {result['host']['cv_percent']:.1f} %")
    print(f"  device jitter CV       : {result['device']['cv_percent']:.1f} %")
    print(f"  packet loss            : {result['loss_percent']:.2f} %")
    print()
    print("  Note: the v1 figure above is the ceiling v1 could have reached by")
    print("  taking one frame per report. The v1 recording actually achieved")
    print("  ~33 Hz, because its non-blocking read loop also dropped reports.")
    print()
    print("  Next: scripts/02_record_raw.py to capture the three conditions")
    print("        (stationary / pointing / flick) for spectral analysis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
