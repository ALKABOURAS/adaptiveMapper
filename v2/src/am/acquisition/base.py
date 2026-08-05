"""Common interface for every source of inertial samples.

Any source -- real hardware, a recorded file, or the synthetic generator -- looks
identical to the rest of the system. That is what makes the synthetic generator
usable as a unit-test fixture for filters, and what makes recorded sessions
replayable so that every algorithm is evaluated on *identical* input rather than
on a fresh hand movement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from am.core.types import ImuSample, StreamStats


class ImuSource(ABC):
    """Abstract source of :class:`ImuSample` values."""

    def __init__(self) -> None:
        self.stats = StreamStats()

    @abstractmethod
    def open(self) -> bool:
        """Acquire the underlying resource. Return True on success."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying resource."""

    @abstractmethod
    def samples(self) -> Iterator[ImuSample]:
        """Yield samples in device-time order until the source is exhausted.

        Implementations must yield **every** IMU frame the hardware provides,
        must never silently coalesce or discard frames, and must set
        ``dropped_before`` when a gap is detected.
        """

    @property
    @abstractmethod
    def nominal_rate_hz(self) -> float:
        """Expected sample rate, for filter design and sanity checks."""

    def __enter__(self):
        if not self.open():
            raise RuntimeError(f"{type(self).__name__}: failed to open source")
        return self

    def __exit__(self, *exc) -> None:
        self.close()
