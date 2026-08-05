"""Filter suite for the adaptiveMapper study.

The registry below is the canonical list of experimental conditions referenced
in the research design (§3.3). Evaluation scripts iterate over it so that no
algorithm is silently omitted from a comparison.
"""

from am.filters.adaptive import (
    CascadedOneEuro,
    DoubleExponentialSmoothing,
    JerkOneEuro,
    OneEuroFilter,
)
from am.filters.base import FilterChain, Passthrough, ScalarFilter, VectorFilter
from am.filters.classical import BiquadLowpass, HampelFilter, MedianFilter
from am.filters.kalman import (
    AdaptiveKalman,
    ConstantVelocityKalman,
    ScalarRandomWalkKalman,
)
from am.filters.tremor import NotchFilter, WFLCTremorFilter

__all__ = [
    "ScalarFilter",
    "Passthrough",
    "FilterChain",
    "VectorFilter",
    "HampelFilter",
    "MedianFilter",
    "BiquadLowpass",
    "OneEuroFilter",
    "CascadedOneEuro",
    "JerkOneEuro",
    "DoubleExponentialSmoothing",
    "ScalarRandomWalkKalman",
    "ConstantVelocityKalman",
    "AdaptiveKalman",
    "WFLCTremorFilter",
    "NotchFilter",
    "build_registry",
]


def build_registry(
    rate_hz: float = 200.0, measurement_var: float = 0.007
) -> dict[str, ScalarFilter]:
    """Instantiate every evaluation condition F0..F10.

    Parameter values here are *starting points*, not tuned values. Tuning must
    be done on a calibration set that is disjoint from the evaluation set, and
    the final values must be reported in the thesis (research design §5).

    Parameters
    ----------
    measurement_var:
        Gyroscope measurement noise variance, dps^2. The default is the value
        measured on this hardware by Allan variance on a 3-minute stationary
        recording (ARW 0.34 deg/sqrt(h), giving ~0.007 dps^2). Do not guess it:
        an earlier default of 4.0 -- carried over from the synthetic fixture --
        overstated the noise by roughly 600x and made every Kalman condition
        smooth far harder than the sensor warrants. Re-measure per device with
        scripts/04_analyze_recordings.py.
    """
    hampel = lambda: HampelFilter(window=7, n_sigmas=3.0)  # noqa: E731

    return {
        "F0": Passthrough(),
        "F1": hampel(),
        "F2": BiquadLowpass(cutoff_hz=8.0, rate_hz=rate_hz),
        "F3": OneEuroFilter(min_cutoff=1.0, beta=0.007),
        "F4": CascadedOneEuro(min_cutoff=1.0, beta=0.007),
        "F5": JerkOneEuro(min_cutoff=1.0, beta=0.007, gamma=0.0005),
        "F6": DoubleExponentialSmoothing(alpha=0.25, gamma=0.1, predict_steps=1.0),
        "F7": ConstantVelocityKalman(process_var=1e3, measurement_var=measurement_var),
        "F8": AdaptiveKalman(
            process_var=1e2, measurement_var=measurement_var, speed_coupling=0.05
        ),
        "F9": WFLCTremorFilter(rate_hz=rate_hz, f0_hz=9.0, mu_freq=1e-3, mu_amp=5e-2),
        "F10": FilterChain(
            hampel(),
            WFLCTremorFilter(rate_hz=rate_hz, f0_hz=9.0, mu_freq=1e-3, mu_amp=5e-2),
            OneEuroFilter(min_cutoff=1.0, beta=0.007),
            name="F10-proposed",
        ),
        # Reproduces the v1 comparison so the thesis can show the structural
        # ramp error rather than assert it.
        "K0": ScalarRandomWalkKalman(process_noise=1e-5, measurement_noise=0.1),
    }
