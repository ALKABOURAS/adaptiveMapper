# adaptiveMapper v2

Rebuild of the thesis codebase. The research goal is unchanged; the acquisition
path, filter suite, baselines and evaluation method are rebuilt from scratch.

**Read [`docs/00_research_design.md`](docs/00_research_design.md) first.** It is
the normative document — code and thesis chapters both follow from it.

---

## Why v2 exists

Analysis of the v1 recording (`../docs/images/2026-01-09_20-51-59_live_data.csv`)
found the pipeline was running at **≈ 33 Hz with 21 % timing jitter**, using one
of the three IMU frames in each Bluetooth report and timestamping with the host
clock.

Three v1 conclusions do not survive that:

| v1 conclusion | Status |
|---|---|
| "The 1 € filter barely reduces noise" | Confounded — at 33 Hz with `min_cutoff = 1.0` the smoothing coefficient is ≈ 0.16, so the filter had almost no effect by construction |
| "The Kalman filter lags" | Invalid comparison — the v1 estimator was a scalar random walk with no velocity state, which has a *structural* steady-state error on a ramp |
| Anything about frequency content | Unobservable — Nyquist was 16 Hz and the 8–12 Hz tremor band sat at the edge, aliased |

None of these are failures of the research idea. They are measurement problems,
and RQ0 (acquisition adequacy) is now a reportable finding in its own right.

---

## Layout

```
v2/
├── docs/00_research_design.md    normative: baselines, conditions, metrics, stats
├── src/am/
│   ├── core/          ImuSample, hardware-counter timebase reconstruction
│   ├── acquisition/   Joy-Con driver (200 Hz), synthetic generator with ground truth
│   ├── filters/       F0–F10 condition set, one interface
│   ├── analysis/      Welch PSD, spectrogram, Allan variance, lag & throughput metrics
│   ├── processing/    crosstalk suppression, CD gain            (next)
│   └── experiment/    participant/block/trial management         (next)
├── scripts/
│   ├── 01_verify_acquisition.py  ← run this first, with the controller
│   └── 03_evaluate_filters.py    Level 1 comparison table
└── tests/test_filters.py         22 tests, all passing
```

## Setup

```bash
cd v2
pip install -r requirements.txt
pytest tests/ -v
python scripts/03_evaluate_filters.py          # works with no hardware
python scripts/01_verify_acquisition.py --device right --duration 20
```

Only `numpy` is needed for the analysis; `matplotlib` for figures, `hidapi` for
the controller. There is **no scipy dependency** — Welch, the spectrogram and
the Allan deviation are implemented directly so the study reproduces from a
minimal environment.

---

## Design decisions worth knowing

**Two clocks per sample.** `t_device` is reconstructed from the controller's own
counter and is uniform by construction; `t_host` is wall-clock arrival time.
Analysis uses `t_device`. Their difference *is* the transport latency
measurement.

**Packet loss is data, not noise.** Gaps are marked explicitly via
`dropped_before` and reported per trial. Bluetooth interference shows up here,
as loss and delay — not as a spectral line in the gyroscope signal. There is
nothing to notch, and the mains-interference analogy does not transfer to a
battery-powered digital MEMS device.

**Nonlinear before linear.** The spikes in the v1 recordings are impulsive
outliers. A linear filter cannot remove an impulse — by linearity it only
redistributes it over its own impulse response. Hampel runs first in every
chain.

**Honest baselines.** `ScalarRandomWalkKalman` (the v1 estimator) is kept as
condition K0 so the thesis can *show* the ramp error rather than assert it, and
`ConstantVelocityKalman` / `AdaptiveKalman` are the fair comparisons the v1
study lacked.

**`dt` is always passed in.** No filter calls `time.time()`. That is what makes
recorded sessions replayable, so every algorithm is scored on identical input
instead of on a fresh hand movement.

---

## Verified results

`pytest tests/` — 22 passing. The spectral estimators are checked against
analytic references (Parseval: 12.5013 vs 12.5 exact; white-noise Allan slope
−0.478 vs −0.5 theoretical).

Two findings already worth recording:

**Cascading the 1 € filter is not a free lunch.** A second stage gives
−12 dB/octave and measurably lower stationary noise, but doubles the lag
(30 → 60 ms on the synthetic fixture). Which side wins is a task-level question,
which is why Level 2 exists.

**WFLC is conditional, and the harness says so.** On a favourable signal (tremor
8 dps, stable frequency) it achieves **15.7 dB** of tremor attenuation for
−0.03 dB on the voluntary band, converging to 9.49 Hz against a true 9.5 Hz. On
the harder default fixture — tremor at 1.44 % of total power, frequency drifting
±1 Hz — it achieves **0.2 dB**, i.e. nothing.

That gap is the whole point of running the spectral analysis on real recordings
before committing to the condition set. If the Joy-Con signal in this task looks
like the second case, F9 should be **dropped and the negative result reported**.

---

## Status

Done: research design, core types, timebase, Joy-Con driver, synthetic
generator, filter suite F0–F10, spectral and metric analysis, verification
script, Level 1 harness, tests.

Next: record real data and settle the tremor question · crosstalk and gain
stages · Unity testbed with participant management · ISO 9241-9 throughput ·
power analysis · rewrite of §2.4, §3.3 and Chapters 4–5.
