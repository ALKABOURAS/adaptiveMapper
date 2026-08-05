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
    python scripts/01_verify_acquisition.py --device right --duration 20

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


def q2_frame_order(all_frames: np.ndarray) -> str:
    """Determine frame ordering by minimising cross-report discontinuity."""
    print("=" * 72)
    print("Q2  Which temporal order are the three frames in?")
    print("=" * 72)

    def total_variation(order: tuple[int, int, int]) -> float:
        # Concatenate all reports under this ordering and measure roughness.
        # The correct ordering yields a continuous trajectory; the wrong one
        # inserts a backward jump at every report boundary.
        seq = all_frames[:, list(order), :].reshape(-1, 3)
        return float(np.abs(np.diff(seq, axis=0)).sum())

    tv_oldest = total_variation((0, 1, 2))
    tv_newest = total_variation((2, 1, 0))

    print(f"  total variation, oldest-first (0,1,2) = {tv_oldest:12.1f}")
    print(f"  total variation, newest-first (2,1,0) = {tv_newest:12.1f}")

    ratio = max(tv_oldest, tv_newest) / max(min(tv_oldest, tv_newest), 1e-12)
    if ratio < 1.02:
        print(
            "\n  RESULT: INCONCLUSIVE -- the two orderings differ by less than 2%.\n"
            "          The device was probably too still. Re-run while moving it."
        )
        return "inconclusive"

    winner = "OLDEST_FIRST" if tv_oldest < tv_newest else "NEWEST_FIRST"
    print(f"\n  RESULT: {winner}  (smoother by {(ratio - 1) * 100:.1f}%)")
    print(f"          Set FrameOrder.{winner} in the driver.")
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


def make_figure(reports, all_frames, result, out_path: Path) -> None:
    """Produce the Chapter 4 acquisition figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # (a) v1 vs v2 sampling of the same motion, yaw axis.
    ax = axes[0, 0]
    n_show = min(60, len(reports))
    v2 = all_frames[:n_show, :, 2].reshape(-1)
    t_v2 = np.arange(v2.size) * 0.005
    v1 = all_frames[:n_show, 0, 2]
    t_v1 = np.arange(v1.size) * 0.015
    ax.plot(t_v2, v2, "-", lw=1.2, label=f"v2: all 3 frames ({result['imu_rate_hz']:.0f} Hz)")
    ax.plot(t_v1, v1, "o--", ms=4, lw=1.0, alpha=0.8,
            label=f"v1: frame 0 only ({result['host']['rate_hz']:.0f} Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Yaw rate (dps)")
    ax.set_title("(a) Discarded data: v1 kept 1 sample in 3")
    ax.legend(fontsize=8)
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

    # (d) spectra at both rates.
    ax = axes[1, 1]
    from am.analysis.spectral import welch_psd

    yaw_v2 = all_frames[:, :, 2].reshape(-1)
    yaw_v1 = all_frames[:, 0, 2]
    if yaw_v2.size > 256:
        f2, p2 = welch_psd(yaw_v2, result["imu_rate_hz"])
        ax.semilogy(f2, p2, lw=1.2, label=f"v2 @ {result['imu_rate_hz']:.0f} Hz")
    if yaw_v1.size > 128:
        f1, p1 = welch_psd(yaw_v1, result["host"]["rate_hz"])
        ax.semilogy(f1, p1, lw=1.2, alpha=0.8,
                    label=f"v1 @ {result['host']['rate_hz']:.0f} Hz")
    ax.axvspan(8, 12, alpha=0.25, color="tab:orange")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dps$^2$/Hz)")
    ax.set_title("(d) Power spectral density, yaw")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Acquisition verification: Joy-Con input report 0x30", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Figure written to {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="right", choices=["left", "right", "pro"])
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--out", default="data/acquisition_verification.png")
    args = ap.parse_args()

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
    order = q2_frame_order(all_frames)
    result = q3_true_sample_rate(reports)

    make_figure(reports, all_frames, result, Path(args.out))

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  frame order        : {order}")
    print(f"  v1 rate            : {result['host']['rate_hz']:.1f} Hz")
    print(f"  v2 rate            : {result['imu_rate_hz']:.1f} Hz")
    print(f"  host jitter CV     : {result['host']['cv_percent']:.1f} %")
    print(f"  packet loss        : {result['loss_percent']:.2f} %")
    print()
    print("  Next: scripts/02_record_raw.py to capture the three conditions")
    print("        (stationary / pointing / flick) for spectral analysis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
