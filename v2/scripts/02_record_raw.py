#!/usr/bin/env python3
"""Record the three reference conditions for spectral analysis.

Each condition isolates one thing the study needs to know, and none of them can
be substituted by the others:

``stationary``
    Device at rest on a stable surface, untouched. Characterises the *sensor*:
    white noise level, bias drift, and the Allan variance from which principled
    Kalman tuning values are derived. Must be genuinely untouched -- a hand
    resting on the table transmits building vibration.

``pointing``
    Held in the hand, aiming at a fixed distant point, deliberately trying to
    stay on target. Contains voluntary micro-corrections plus physiological
    tremor with no large motion to mask it. **This is the recording that decides
    whether the tremor stage stays in the study**: if the 8-12 Hz band holds a
    negligible share of the power here, F9 should be dropped and the negative
    result reported.

``flick``
    Rapid target-to-target movements with pauses between them. Contains the
    ballistic transients that stress the jitter/lag trade-off, and is where the
    75 ms perceptual threshold actually bites.

Usage
-----
    python scripts/02_record_raw.py --device left
    python scripts/02_record_raw.py --device left --only pointing --duration 60

Longer is better for the stationary condition: Allan variance needs several
minutes to reach the bias-instability minimum.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from am.acquisition.joycon import JoyConSource  # noqa: E402
from am.analysis.spectral import band_report, is_stationary  # noqa: E402
from am.core.timebase import uniformity  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONDITIONS = {
    "stationary": {
        "duration": 180.0,
        "instruction": (
            "Place the controller on a stable surface and DO NOT TOUCH IT.\n"
            "     Step away if you can. This measures the sensor itself."
        ),
    },
    "pointing": {
        "duration": 60.0,
        "instruction": (
            "Hold it naturally and aim at a fixed distant point.\n"
            "     Try to stay on target. Small corrections are expected --\n"
            "     that, plus tremor, is exactly what we are measuring."
        ),
    },
    "flick": {
        "duration": 60.0,
        "instruction": (
            "Move rapidly between two targets, pausing ~1 s on each.\n"
            "     Fast, deliberate movements with clear stops."
        ),
    },
}


def countdown(seconds: int = 5) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"\r  Starting in {remaining} ...", end="", flush=True)
        time.sleep(1.0)
    print("\r  RECORDING            ")


def record(source: JoyConSource, duration_s: float) -> list:
    samples = []
    start = time.perf_counter()
    last_print = start
    for sample in source.samples():
        samples.append(sample)
        now = time.perf_counter()
        if now - last_print >= 1.0:
            elapsed = now - start
            print(f"\r  {elapsed:5.1f}s / {duration_s:.0f}s  "
                  f"({len(samples)} samples)", end="", flush=True)
            last_print = now
        if now - start >= duration_s:
            break
    print()
    return samples


def write_csv(samples: list, path: Path, condition: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "t_device", "t_host", "gyro_x", "gyro_y", "gyro_z",
            "accel_x", "accel_y", "accel_z", "dropped_before", "buttons", "seq",
        ])
        for s in samples:
            writer.writerow([
                f"{s.t_device:.6f}", f"{s.t_host:.6f}",
                f"{s.gyro[0]:.5f}", f"{s.gyro[1]:.5f}", f"{s.gyro[2]:.5f}",
                f"{s.accel[0]:.5f}", f"{s.accel[1]:.5f}", f"{s.accel[2]:.5f}",
                s.dropped_before, s.buttons, s.seq,
            ])
    print(f"  Written: {path}")


def summarise(samples: list, condition: str) -> None:
    t = np.array([s.t_device for s in samples])
    gyro = np.stack([s.gyro for s in samples])
    dropped = sum(s.dropped_before for s in samples)

    u = uniformity(np.diff(t))
    rate = u["rate_hz"]

    print(f"  samples      : {len(samples)}  ({dropped} dropped, "
          f"{dropped / max(len(samples) + dropped, 1) * 100:.2f} %)")
    print(f"  rate         : {rate:.1f} Hz   (dt CV {u['cv_percent']:.2f} %)")
    print(f"  gyro std     : x {gyro[:, 0].std():7.3f}  "
          f"y {gyro[:, 1].std():7.3f}  z {gyro[:, 2].std():7.3f} dps")

    if len(samples) < 512:
        return

    yaw = gyro[:, 2]
    stationary = is_stationary(yaw, rate)
    print(f"  looks still  : {'yes' if stationary else 'no'}", end="")
    if condition == "stationary" and not stationary:
        print("   <-- WARNING: expected a still recording. Redo it.")
    elif condition != "stationary" and stationary:
        print("   <-- WARNING: expected motion. Were you moving?")
    else:
        print()

    report = band_report(yaw, rate)
    if report:
        print(f"  band power   : voluntary(0-5Hz) {report['voluntary_0_5Hz']:5.1f} %   "
              f"tremor(8-12Hz) {report['tremor_8_12Hz']:5.1f} %   "
              f"noise(>15Hz) {report['noise_15Hz_up']:5.1f} %")
        if "tremor_peak_hz" in report:
            print(f"  tremor peak  : {report['tremor_peak_hz']:.2f} Hz")
        if condition == "pointing":
            share = report["tremor_8_12Hz"]
            print()
            if share >= 5.0:
                print(f"  -> Tremor is {share:.1f} % of power. Narrowband suppression")
                print("     is worth pursuing; keep F9 in the study.")
            elif share >= 1.0:
                print(f"  -> Tremor is only {share:.1f} % of power. Marginal. Compare")
                print("     F9 against the fixed notch before committing to it.")
            else:
                print(f"  -> Tremor is {share:.1f} % of power, i.e. negligible.")
                print("     Drop F9 and report this as a negative finding: for this")
                print("     device and task, tremor is not the limiting factor.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="left", choices=["left", "right", "pro"])
    ap.add_argument("--only", choices=sorted(CONDITIONS), help="record one condition")
    ap.add_argument("--duration", type=float, help="override the default duration")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--no-calibrate", action="store_true",
                    help="skip bias calibration (keeps the raw bias in the data)")
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else PROJECT_ROOT / "data" / "recordings"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    todo = [args.only] if args.only else list(CONDITIONS)

    source = JoyConSource(device_type=args.device)
    if not source.open():
        print(f"No {args.device} controller found. Pair it over Bluetooth first.")
        return 1

    try:
        if not args.no_calibrate:
            print("Bias calibration -- put the controller down and keep it still.")
            countdown(3)
            source.calibrate(duration_s=3.0)
            print()

        for condition in todo:
            spec = CONDITIONS[condition]
            duration = args.duration or spec["duration"]

            print("=" * 72)
            print(f"CONDITION: {condition}   ({duration:.0f} s)")
            print("=" * 72)
            print(f"  {spec['instruction']}")
            print()
            input("  Press Enter when ready ...")
            countdown(5)

            samples = record(source, duration)
            if len(samples) < 100:
                print(f"  Only {len(samples)} samples -- something is wrong. Skipping.")
                continue

            path = outdir / f"{stamp}_{condition}.csv"
            write_csv(samples, path, condition)
            summarise(samples, condition)
            print()
    finally:
        source.close()

    print("=" * 72)
    print("Next:")
    print(f"  python scripts/03_evaluate_filters.py --csv "
          f"{outdir / f'{stamp}_pointing.csv'}")
    print()
    print("  The pointing recording decides whether F9 stays in the condition")
    print("  set. The stationary recording gives the Allan variance and the")
    print("  Kalman tuning values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
