"""Adaptive suppression of physiological tremor (WFLC).

The frequency-domain argument, correctly aimed
----------------------------------------------
An earlier hypothesis was to filter interference from the wireless link, by
analogy with 50 Hz mains interference. That analogy does not hold. Bluetooth
carries an integrity-checked digital payload, so link interference produces
*lost and delayed packets*, not an additive sinusoid in the gyroscope values;
and a battery-powered digital MEMS device has no coupling path to a mains
distribution network. There is no spectral line to notch. (See research design
§3.1.)

There *is* a narrowband disturbance in this signal, and it is biological.
Physiological tremor is an involuntary oscillation present in every healthy
hand, centred around 8-12 Hz, well separated from voluntary hand motion, which
lies below about 5 Hz. That separation is what makes narrowband rejection
possible without touching the intentional signal -- which a low-pass filter
cannot do without also attenuating fast voluntary movement.

Why adaptive rather than a fixed notch
--------------------------------------
Tremor frequency is not constant: it varies between people, with posture,
fatigue and limb loading. A fixed notch centred on a population average is
mistuned for most individuals most of the time. The Weighted-Frequency Fourier
Linear Combiner (Riviere, Reger and Thakor, 1998) estimates the frequency online
and tracks it, and is the algorithm used in the Micron handheld surgical
instrument for exactly this problem.

Prerequisite
------------
Observing an 8-12 Hz band requires a sample rate comfortably above 24 Hz. The
v1 pipeline ran at ~33 Hz, placing the tremor band at the very edge of Nyquist
and contaminated by aliasing. This filter is only meaningful on the 200 Hz
acquisition path.
"""

from __future__ import annotations

import math

import numpy as np

from am.filters.base import ScalarFilter


class WFLCTremorFilter(ScalarFilter):
    """Weighted-Frequency Fourier Linear Combiner tremor canceller.

    Models the disturbance as a sum of ``harmonics`` sinusoids whose fundamental
    frequency and amplitudes are estimated by LMS adaptation, then subtracts it.
    The filter output is the residual, i.e. the tremor-free signal.

    Parameters
    ----------
    rate_hz:
        Nominal sample rate, used to convert between Hz and rad/sample.
    f0_hz:
        Initial frequency estimate. Start it near the middle of the expected
        tremor band.
    band_hz:
        ``(low, high)`` constraint on the tracked frequency. **Essential**: an
        unconstrained WFLC will happily lock onto the large low-frequency
        voluntary motion and cancel the very signal being measured. The band
        encodes the prior that tremor is 8-12 Hz.
    mu_freq:
        Frequency adaptation rate. Too large and the estimate wanders; too
        small and it never leaves ``f0_hz``.
    mu_amp:
        Amplitude adaptation rate. The defaults were calibrated on the
        synthetic fixture: at 5e-3 the filter converges too slowly to attenuate
        anything over a 40 s recording (measured 0.3 dB), while at 5e-2 it
        reaches 15.7 dB of tremor attenuation for -0.03 dB on the voluntary
        band. Re-tune per device rather than trusting these numbers.
    bandpass_q:
        Quality factor of the internal adaptation-path band-pass.
    harmonics:
        Number of harmonics modelled. 1 is usually sufficient for physiological
        tremor; 2 captures mild waveform asymmetry.

    Attributes
    ----------
    frequency_hz:
        Current frequency estimate. Log this per trial -- the tracked tremor
        frequency per participant is a reportable result in its own right.
    """

    def __init__(
        self,
        rate_hz: float = 200.0,
        f0_hz: float = 9.0,
        band_hz: tuple[float, float] = (6.0, 14.0),
        mu_freq: float = 1e-3,
        mu_amp: float = 5e-2,
        harmonics: int = 1,
        bandpass_q: float = 3.0,
        *,
        name: str | None = None,
    ) -> None:
        if harmonics < 1:
            raise ValueError("harmonics must be at least 1")
        if not band_hz[0] < f0_hz < band_hz[1]:
            raise ValueError("f0_hz must lie inside band_hz")
        if band_hz[1] >= rate_hz / 2:
            raise ValueError(
                f"band upper bound {band_hz[1]} Hz is at or above Nyquist "
                f"({rate_hz / 2} Hz) -- the tremor band is not observable at "
                f"{rate_hz} Hz. This is exactly the v1 acquisition problem."
            )
        self.rate_hz = rate_hz
        self.f0_hz = f0_hz
        self.band_hz = band_hz
        self.mu_freq = mu_freq
        self.mu_amp = mu_amp
        self.harmonics = harmonics
        self.bandpass_q = bandpass_q
        self.name = name or f"F9-WFLC(f0={f0_hz:g},M={harmonics})"
        self.reset()

    def reset(self) -> None:
        self._w0 = 2.0 * math.pi * self.f0_hz / self.rate_hz  # rad/sample
        self._w = np.zeros(2 * self.harmonics, dtype=np.float64)
        self._theta = 0.0
        self._bp_z1 = 0.0
        self._bp_z2 = 0.0
        self._design_bandpass()
        #: Running power estimate of the band-limited signal, for normalised LMS.
        self._power = 1.0

    # -- adaptation-path band-pass ------------------------------------------
    #
    # Adapting directly on the raw signal does not work, and the failure mode is
    # instructive: voluntary motion carries the overwhelming majority of the
    # signal power, so the LMS gradients are dominated by a component the filter
    # is not trying to model, and the frequency estimate is driven straight into
    # whichever band limit it is clamped by. Measured on the synthetic fixture,
    # the estimate railed to the lower bound for every choice of step size.
    #
    # The fix is to drive the adaptation from a band-pass centred on the current
    # frequency estimate, while still subtracting the resulting tremor estimate
    # from the *original* signal. A second-order band-pass has exactly zero
    # phase shift at its centre frequency, and the centre is re-tuned to the
    # tracked frequency on every update, so the estimate stays phase-aligned
    # with the tremor in the unfiltered signal.

    def _design_bandpass(self) -> None:
        f0 = max(self.band_hz[0], min(self.frequency_hz, self.band_hz[1]))
        w0 = 2.0 * math.pi * f0 / self.rate_hz
        alpha = math.sin(w0) / (2.0 * self.bandpass_q)
        cos_w0 = math.cos(w0)
        # Constant-peak-gain band-pass (RBJ cookbook): unity gain at f0.
        b0, b1, b2 = alpha, 0.0, -alpha
        a0, a1, a2 = 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha
        self._bp_b = (b0 / a0, b1 / a0, b2 / a0)
        self._bp_a = (a1 / a0, a2 / a0)

    def _bandpass(self, x: float) -> float:
        b0, b1, b2 = self._bp_b
        a1, a2 = self._bp_a
        y = b0 * x + self._bp_z1
        self._bp_z1 = b1 * x - a1 * y + self._bp_z2
        self._bp_z2 = b2 * x - a2 * y
        return y

    @property
    def frequency_hz(self) -> float:
        return self._w0 * self.rate_hz / (2.0 * math.pi)

    @property
    def amplitude(self) -> float:
        """Estimated tremor amplitude (fundamental)."""
        return float(math.hypot(self._w[0], self._w[self.harmonics]))

    def _clamp_frequency(self) -> None:
        lo = 2.0 * math.pi * self.band_hz[0] / self.rate_hz
        hi = 2.0 * math.pi * self.band_hz[1] / self.rate_hz
        self._w0 = min(max(self._w0, lo), hi)

    def update(self, x: float, dt: float) -> float:
        m = self.harmonics

        # Band-limited copy used only to drive the adaptation.
        x_bp = self._bandpass(x)

        # Accumulated phase. Using the running sum of w0 (rather than w0 * k)
        # is what lets the frequency change without a phase discontinuity.
        self._theta += self._w0
        if self._theta > 2.0 * math.pi:
            self._theta -= 2.0 * math.pi

        r = np.arange(1, m + 1, dtype=np.float64)
        sin_terms = np.sin(r * self._theta)
        cos_terms = np.cos(r * self._theta)
        basis = np.concatenate([sin_terms, cos_terms])

        tremor_estimate = float(self._w @ basis)
        error = x_bp - tremor_estimate  # adaptation error, band-limited

        # Normalised LMS: dividing by the running input power makes the step
        # size independent of signal amplitude, so one setting works across
        # participants with different tremor magnitudes.
        self._power = 0.999 * self._power + 0.001 * (x_bp * x_bp)
        norm = 1.0 / (self._power + 1e-6)

        grad = float(np.sum(r * (self._w[:m] * cos_terms - self._w[m:] * sin_terms)))
        self._w0 += 2.0 * self.mu_freq * error * grad * norm
        self._clamp_frequency()
        self._design_bandpass()

        self._w += 2.0 * self.mu_amp * error * basis

        # Subtract the tremor estimate from the ORIGINAL signal, not from the
        # band-passed one -- the voluntary motion must survive untouched.
        return x - tremor_estimate


class NotchFilter(ScalarFilter):
    """Fixed second-order IIR notch, as the non-adaptive comparison to WFLC.

    Including it lets the thesis answer "does the adaptation actually buy
    anything, or would a fixed notch at the population mean have done?" -- a
    question a reviewer will ask, and one worth answering with data.

    Parameters
    ----------
    notch_hz:
        Centre frequency.
    q:
        Quality factor. Higher is narrower; too narrow misses a mistuned tremor,
        too wide starts eating voluntary motion.
    """

    def __init__(self, notch_hz: float = 9.5, rate_hz: float = 200.0, q: float = 5.0):
        if notch_hz >= rate_hz / 2:
            raise ValueError("notch frequency must be below Nyquist")
        self.notch_hz = notch_hz
        self.rate_hz = rate_hz
        self.q = q
        self.name = f"notch({notch_hz:g}Hz,Q={q:g})"
        self._design()
        self.reset()

    def _design(self) -> None:
        w0 = 2.0 * math.pi * self.notch_hz / self.rate_hz
        alpha = math.sin(w0) / (2.0 * self.q)
        cos_w0 = math.cos(w0)
        b0, b1, b2 = 1.0, -2.0 * cos_w0, 1.0
        a0, a1, a2 = 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha
        self._b = (b0 / a0, b1 / a0, b2 / a0)
        self._a = (a1 / a0, a2 / a0)

    def reset(self) -> None:
        self._z1 = 0.0
        self._z2 = 0.0

    def update(self, x: float, dt: float) -> float:
        b0, b1, b2 = self._b
        a1, a2 = self._a
        y = b0 * x + self._z1
        self._z1 = b1 * x - a1 * y + self._z2
        self._z2 = b2 * x - a2 * y
        return y
