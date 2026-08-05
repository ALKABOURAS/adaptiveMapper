"""Joy-Con / Pro Controller acquisition at the full hardware sample rate.

What changed from v1 and why
----------------------------
v1 read bytes 19..24 of input report 0x30 and emitted one sample per report,
timestamped with ``time.time()`` at reception, on a non-blocking handle. Measured
result: ~33 Hz effective, 21 % timing jitter, 67 % of the sensor data discarded.

Report 0x30 actually carries **three** complete 6-axis IMU frames at offsets 13,
25 and 37, sampled 5 ms apart, in a report emitted at ~60 Hz. All three are
emitted here, giving ~180-200 Hz, and each is timestamped from the device's own
counter at byte 1 rather than from the host clock.

Report 0x30 layout
------------------
====== =========================================================
offset contents
====== =========================================================
0      report id (0x30)
1      timer / packet counter, wraps at 256
2      battery level and connection info
3..5   button state
6..8   left stick (12-bit packed)
9..11  right stick (12-bit packed)
12     vibrator input report
13..24 IMU frame 0  (accel xyz, gyro xyz -- six int16 LE)
25..36 IMU frame 1
37..48 IMU frame 2
====== =========================================================

The temporal ordering of the three frames (oldest-first vs newest-first) is
**not** assumed. :data:`FrameOrder` selects it and
``scripts/01_verify_acquisition.py`` determines it empirically by choosing the
order that minimises signal discontinuity across report boundaries.
"""

from __future__ import annotations

import struct
import time
from enum import Enum
from typing import Iterator

import numpy as np

from am.acquisition.base import ImuSource
from am.core.timebase import CounterTimebase
from am.core.types import (
    JOYCON_FRAMES_PER_REPORT,
    JOYCON_SAMPLE_PERIOD_S,
    ImuSample,
)

VENDOR_ID = 0x057E
PRODUCT_IDS = {"left": 0x2006, "right": 0x2007, "pro": 0x2009}

REPORT_ID_FULL = 0x30
IMU_BLOCK_OFFSET = 13
IMU_FRAME_SIZE = 12

#: Gyroscope full-scale +-2000 dps over 16 bits -> 4000/65535 dps per LSB.
GYRO_DPS_PER_LSB = 0.06103
#: Accelerometer full-scale +-8 g over 16 bits -> 16/65535 g per LSB.
ACCEL_G_PER_LSB = 0.000244

_FRAME_STRUCT = struct.Struct("<6h")


class FrameOrder(str, Enum):
    """Temporal ordering of the three IMU frames inside one report."""

    OLDEST_FIRST = "oldest_first"  # frame 0 is t-10ms, frame 2 is t
    NEWEST_FIRST = "newest_first"  # frame 0 is t, frame 2 is t-10ms


class JoyConSource(ImuSource):
    """Full-rate Joy-Con acquisition with hardware-derived timestamps.

    Parameters
    ----------
    device_type:
        ``"left"``, ``"right"`` or ``"pro"``.
    frame_order:
        Temporal ordering of the three IMU frames per report. The default was
        determined empirically, not taken from documentation: on a measured
        20 s recording the NEWEST_FIRST reconstruction was 117 % smoother by
        total variation than OLDEST_FIRST. Frame 0 therefore holds the most
        recent sample and frame 2 the oldest. Re-run the verification script
        if the firmware or report mode changes.
    gyro_dps_per_lsb, accel_g_per_lsb:
        Scale factors. Override after calibrating against a known angular rate
        (research design B3) instead of trusting the nominal value.
    read_timeout_ms:
        Blocking read timeout. Blocking (not polling) is what removes the host
        scheduling jitter that dominated v1.
    """

    def __init__(
        self,
        device_type: str = "left",
        *,
        frame_order: FrameOrder = FrameOrder.NEWEST_FIRST,
        gyro_dps_per_lsb: float = GYRO_DPS_PER_LSB,
        accel_g_per_lsb: float = ACCEL_G_PER_LSB,
        read_timeout_ms: int = 200,
    ) -> None:
        super().__init__()
        if device_type not in PRODUCT_IDS:
            raise ValueError(
                f"device_type must be one of {sorted(PRODUCT_IDS)}, got {device_type!r}"
            )
        self.device_type = device_type
        self.product_id = PRODUCT_IDS[device_type]
        self.frame_order = frame_order
        self.gyro_scale = gyro_dps_per_lsb
        self.accel_scale = accel_g_per_lsb
        self.read_timeout_ms = read_timeout_ms

        self._device = None
        self._packet_number = 0
        self._seq = 0
        self._t0_host: float | None = None
        self._timebase = CounterTimebase(
            modulus=256,
            ticks_per_report=JOYCON_FRAMES_PER_REPORT,
            seconds_per_tick=JOYCON_SAMPLE_PERIOD_S,
        )
        #: Gyro bias in dps, subtracted from every sample. Set by ``calibrate``.
        self.gyro_bias = np.zeros(3, dtype=np.float64)

    # ---------------------------------------------------------------- lifecycle

    @property
    def nominal_rate_hz(self) -> float:
        return JOYCON_FRAMES_PER_REPORT / (
            JOYCON_FRAMES_PER_REPORT * JOYCON_SAMPLE_PERIOD_S
        )

    def open(self) -> bool:
        import hid

        for info in hid.enumerate(VENDOR_ID):
            if info["product_id"] != self.product_id:
                continue
            dev = hid.device()
            dev.open_path(info["path"])
            # Blocking reads: the host waits for the device instead of the
            # device waiting for the host. This is the single change that
            # removes most of the v1 timing jitter.
            dev.set_nonblocking(False)
            self._device = dev
            self._enable_imu()
            self._timebase.reset()
            self._t0_host = None
            self._seq = 0
            return True
        return False

    def close(self) -> None:
        if self._device is not None:
            try:
                self._device.close()
            finally:
                self._device = None

    def _send_subcommand(self, subcommand: int, argument: list[int]) -> None:
        self._packet_number = (self._packet_number + 1) % 16
        neutral_rumble = [0x00, 0x01, 0x40, 0x40, 0x00, 0x01, 0x40, 0x40]
        payload = [0x01, self._packet_number] + neutral_rumble + [subcommand] + argument
        self._device.write(bytes(payload))
        time.sleep(0.05)

    def _enable_imu(self) -> None:
        self._send_subcommand(0x40, [0x01])  # enable IMU
        self._send_subcommand(0x03, [0x30])  # standard full input report mode
        time.sleep(0.2)

    # ------------------------------------------------------------------ parsing

    def _parse_frame(self, report: bytes, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Decode IMU frame ``index`` (0..2) into (gyro_dps, accel_g)."""
        off = IMU_BLOCK_OFFSET + index * IMU_FRAME_SIZE
        ax, ay, az, gx, gy, gz = _FRAME_STRUCT.unpack_from(report, off)
        accel = np.array([ax, ay, az], dtype=np.float64) * self.accel_scale
        gyro = np.array([gx, gy, gz], dtype=np.float64) * self.gyro_scale
        return gyro, accel

    def _frame_indices(self) -> tuple[int, ...]:
        """Frame indices in chronological order."""
        if self.frame_order is FrameOrder.OLDEST_FIRST:
            return (0, 1, 2)
        return (2, 1, 0)

    def read_report(self) -> bytes | None:
        """Read one raw input report, or None on timeout / wrong report id."""
        raw = self._device.read(64, timeout_ms=self.read_timeout_ms)
        if not raw:
            return None
        self.stats.reports_received += 1
        if raw[0] != REPORT_ID_FULL or len(raw) < IMU_BLOCK_OFFSET + 3 * IMU_FRAME_SIZE:
            self.stats.reports_malformed += 1
            return None
        return bytes(raw)

    # ------------------------------------------------------------------- stream

    def samples(self) -> Iterator[ImuSample]:
        if self._device is None:
            raise RuntimeError("JoyConSource.samples() called before open()")

        while True:
            report = self.read_report()
            if report is None:
                continue

            t_host = time.perf_counter()
            if self._t0_host is None:
                self._t0_host = t_host
            t_host -= self._t0_host

            t_report, dropped_reports = self._timebase.update(report[1])
            dropped_samples = dropped_reports * JOYCON_FRAMES_PER_REPORT
            self.stats.samples_dropped += dropped_samples

            buttons = (report[3] << 16) | (report[4] << 8) | report[5]

            for position, frame_index in enumerate(self._frame_indices()):
                gyro, accel = self._parse_frame(report, frame_index)
                gyro -= self.gyro_bias

                t_device = t_report + position * JOYCON_SAMPLE_PERIOD_S
                sample = ImuSample(
                    t_device=t_device,
                    t_host=t_host,
                    gyro=gyro,
                    accel=accel,
                    dropped_before=dropped_samples if position == 0 else 0,
                    buttons=buttons,
                    seq=self._seq,
                )
                self._seq += 1

                if self.stats.t_first is None:
                    self.stats.t_first = t_device
                self.stats.t_last = t_device
                self.stats.samples_emitted += 1
                self.stats._host_deltas.append(t_host)

                yield sample

    # -------------------------------------------------------------- calibration

    def calibrate(self, duration_s: float = 3.0, *, verbose: bool = True) -> np.ndarray:
        """Estimate gyroscope bias from a stationary interval.

        Uses the **median** rather than the mean: a single transport glitch or a
        small bump during calibration would otherwise bias every subsequent
        sample for the whole session. Also reports the interquartile range so
        that a calibration taken while the device was moving is detectable
        rather than silently accepted.
        """
        if verbose:
            print(f"Calibrating for {duration_s:.1f} s -- keep the device still.")

        collected: list[np.ndarray] = []
        saved_bias = self.gyro_bias
        self.gyro_bias = np.zeros(3, dtype=np.float64)
        try:
            start = time.perf_counter()
            for sample in self.samples():
                collected.append(sample.gyro.copy())
                if time.perf_counter() - start >= duration_s:
                    break
        finally:
            self.gyro_bias = saved_bias

        if len(collected) < 10:
            raise RuntimeError(f"Calibration collected only {len(collected)} samples")

        data = np.stack(collected)
        bias = np.median(data, axis=0)
        iqr = np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0)

        self.gyro_bias = bias
        if verbose:
            print(f"  n           = {len(collected)}")
            print(f"  bias (dps)  = {np.round(bias, 4)}")
            print(f"  IQR  (dps)  = {np.round(iqr, 4)}")
            if np.any(iqr > 1.0):
                print(
                    "  WARNING: IQR > 1 dps on at least one axis. The device was "
                    "probably not stationary; repeat the calibration."
                )
        return bias
