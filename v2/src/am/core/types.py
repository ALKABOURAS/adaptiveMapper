"""Core data types shared across acquisition, filtering and analysis.

Design rules
------------
1. An ``ImuSample`` carries **two** clocks: ``t_device`` (reconstructed from the
   hardware counter, uniform) and ``t_host`` (wall clock at reception, jittery).
   Analysis always uses ``t_device``. Latency measurement uses the difference.
2. Missing data is represented explicitly by ``dropped_before``, never by a
   silent gap. Downstream code must be able to distinguish "no motion" from
   "no data".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np

# Joy-Con / Pro Controller IMU timing (standard full input report 0x30).
#
# These are measured values, not the figures usually quoted. The report period
# is 15 ms exactly, i.e. 66.7 Hz -- the "60 Hz" often cited in reverse
# engineering notes is a rounding of it. Measured over a 20 s capture: report
# dt 14.94 +- 0.52 ms from the device counter, giving 200.7 Hz at the IMU.
JOYCON_SAMPLE_PERIOD_S = 0.005  # 5 ms between the 3 IMU frames in one report
JOYCON_FRAMES_PER_REPORT = 3
JOYCON_REPORT_PERIOD_S = JOYCON_FRAMES_PER_REPORT * JOYCON_SAMPLE_PERIOD_S  # 15 ms
JOYCON_REPORT_RATE_HZ = 1.0 / JOYCON_REPORT_PERIOD_S  # 66.67 Hz
JOYCON_NOMINAL_IMU_RATE_HZ = 1.0 / JOYCON_SAMPLE_PERIOD_S  # 200 Hz


class Axis(str, Enum):
    """Gyroscope axes in device frame."""

    X = "x"  # roll
    Y = "y"  # pitch
    Z = "z"  # yaw


@dataclass(slots=True)
class ImuSample:
    """A single 6-axis inertial measurement.

    Attributes
    ----------
    t_device:
        Seconds since stream start, reconstructed from the device hardware
        counter. Uniform by construction. **Use this for all analysis.**
    t_host:
        Seconds since stream start measured by the host clock at the moment the
        containing packet was parsed. Includes transport and OS scheduling
        latency. Use only to characterise the transport, never as a time base.
    gyro:
        Angular rate, degrees per second, device frame, bias-corrected.
    accel:
        Specific force, g, device frame.
    dropped_before:
        Number of IMU samples known to be missing immediately before this one,
        inferred from a discontinuity in the hardware counter. Zero in normal
        operation.
    buttons:
        Raw button bitfield, transport specific.
    seq:
        Monotonic sample index assigned by the source, counting dropped samples.
    """

    t_device: float
    t_host: float
    gyro: np.ndarray  # shape (3,), float64, dps
    accel: np.ndarray  # shape (3,), float64, g
    dropped_before: int = 0
    buttons: int = 0
    seq: int = 0

    @property
    def gx(self) -> float:
        return float(self.gyro[0])

    @property
    def gy(self) -> float:
        return float(self.gyro[1])

    @property
    def gz(self) -> float:
        return float(self.gyro[2])


@dataclass(slots=True)
class StreamStats:
    """Running quality statistics for an acquisition source.

    These are first-class experimental data, not debug output: packet loss is a
    confound that must be reported per trial (see research design §5).
    """

    samples_emitted: int = 0
    samples_dropped: int = 0
    reports_received: int = 0
    reports_malformed: int = 0
    t_first: float | None = None
    t_last: float | None = None
    _host_deltas: list[float] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        if self.t_first is None or self.t_last is None:
            return 0.0
        return self.t_last - self.t_first

    @property
    def effective_rate_hz(self) -> float:
        d = self.duration_s
        return self.samples_emitted / d if d > 0 else 0.0

    @property
    def loss_ratio(self) -> float:
        total = self.samples_emitted + self.samples_dropped
        return self.samples_dropped / total if total else 0.0

    def host_jitter(self) -> tuple[float, float]:
        """Return (mean, std) of host inter-arrival times in seconds."""
        if len(self._host_deltas) < 2:
            return (0.0, 0.0)
        a = np.asarray(self._host_deltas)
        return (float(a.mean()), float(a.std()))

    def summary(self) -> str:
        mean_dt, std_dt = self.host_jitter()
        cv = (std_dt / mean_dt * 100.0) if mean_dt > 0 else 0.0
        return (
            f"samples={self.samples_emitted} dropped={self.samples_dropped} "
            f"({self.loss_ratio * 100:.2f}%) reports={self.reports_received} "
            f"malformed={self.reports_malformed} "
            f"duration={self.duration_s:.2f}s "
            f"effective_rate={self.effective_rate_hz:.1f}Hz "
            f"host_dt={mean_dt * 1000:.2f}+-{std_dt * 1000:.2f}ms (CV {cv:.1f}%)"
        )


def samples_to_arrays(
    samples: Sequence[ImuSample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorise a sample sequence into (t_device, gyro[N,3], accel[N,3])."""
    if not samples:
        empty = np.empty((0, 3), dtype=np.float64)
        return np.empty(0, dtype=np.float64), empty, empty
    t = np.fromiter((s.t_device for s in samples), dtype=np.float64, count=len(samples))
    gyro = np.stack([s.gyro for s in samples])
    accel = np.stack([s.accel for s in samples])
    return t, gyro, accel
