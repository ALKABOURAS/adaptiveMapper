"""Reconstruction of a uniform time base from a wrapping hardware counter.

Why this module exists
----------------------
v1 timestamped each sample with ``time.time()`` at the moment of reception. That
measures *when the host got around to reading the packet*, which includes
Bluetooth retransmission, USB HID polling and OS scheduler latency. The measured
result was 21 % jitter on dt, which makes any frequency-domain analysis invalid,
since the FFT assumes a uniform sampling grid.

The Joy-Con input report 0x30 carries a 1-byte counter at offset 1 that the
device increments deterministically per report. Differencing that counter yields
the true elapsed device time and simultaneously reveals dropped packets: a jump
larger than the expected increment means reports were lost in transit.

This module is transport agnostic. Any source with a monotonic wrapping counter
can use it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CounterTimebase:
    """Convert a wrapping integer hardware counter into monotonic seconds.

    Parameters
    ----------
    modulus:
        Counter wraps at this value (256 for a 1-byte counter).
    ticks_per_report:
        Expected counter increment between two consecutive reports. For the
        Joy-Con this is 3, because each report carries 3 IMU frames and the
        counter advances one tick per frame.
    seconds_per_tick:
        Physical duration of one counter tick. For the Joy-Con, 5 ms.
    max_plausible_gap_ticks:
        Increments larger than this are treated as counter desynchronisation
        rather than packet loss, and trigger a resynchronisation instead of
        inflating the drop count. Guards against a wrap being misread as a
        multi-second gap.
    """

    modulus: int = 256
    ticks_per_report: int = 3
    seconds_per_tick: float = 0.005
    max_plausible_gap_ticks: int = 120  # 0.6 s at 5 ms/tick

    _last_raw: int | None = None
    _total_ticks: int = 0
    _resyncs: int = 0

    @property
    def resync_count(self) -> int:
        """Number of times the counter was judged desynchronised."""
        return self._resyncs

    @property
    def total_ticks(self) -> int:
        return self._total_ticks

    def reset(self) -> None:
        self._last_raw = None
        self._total_ticks = 0
        self._resyncs = 0

    def update(self, raw_counter: int) -> tuple[float, int]:
        """Feed the counter from a newly received report.

        Returns
        -------
        (t_report, dropped_reports)
            ``t_report`` is the device time in seconds of the *first* IMU frame
            in this report. ``dropped_reports`` is the number of whole reports
            inferred missing since the previous call.
        """
        raw = int(raw_counter) % self.modulus

        if self._last_raw is None:
            self._last_raw = raw
            return (0.0, 0)

        delta = (raw - self._last_raw) % self.modulus
        self._last_raw = raw

        if delta == 0:
            # Duplicate report. Do not advance time; caller should discard.
            return (self._total_ticks * self.seconds_per_tick, 0)

        if delta > self.max_plausible_gap_ticks:
            # Counter desync (device reset, mode change, or a very long stall).
            # Advance by the nominal amount rather than fabricating a huge gap.
            self._resyncs += 1
            delta = self.ticks_per_report

        dropped_reports = max(0, delta // self.ticks_per_report - 1)
        self._total_ticks += delta
        return (self._total_ticks * self.seconds_per_tick, dropped_reports)


def uniformity(dt_seconds) -> dict[str, float]:
    """Summarise how uniform a sequence of inter-sample intervals is.

    ``cv_percent`` is the headline number: the coefficient of variation of dt.
    Below roughly 2 % the grid can be treated as uniform for spectral analysis;
    the v1 pipeline measured 21 %.
    """
    import numpy as np

    a = np.asarray(dt_seconds, dtype=np.float64)
    a = a[np.isfinite(a) & (a > 0)]
    if a.size < 2:
        return {"n": float(a.size), "mean_ms": 0.0, "std_ms": 0.0, "cv_percent": 0.0}
    mean = float(a.mean())
    std = float(a.std())
    return {
        "n": float(a.size),
        "mean_ms": mean * 1e3,
        "median_ms": float(np.median(a)) * 1e3,
        "std_ms": std * 1e3,
        "min_ms": float(a.min()) * 1e3,
        "max_ms": float(a.max()) * 1e3,
        "cv_percent": (std / mean * 100.0) if mean > 0 else 0.0,
        "rate_hz": 1.0 / mean if mean > 0 else 0.0,
        "nyquist_hz": 0.5 / mean if mean > 0 else 0.0,
    }
