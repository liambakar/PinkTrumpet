# Pink Trumpet ↔ Pink Trombone comparison

Run date: 2026-08-13  
Reference: [Pink Trombone v1.1](https://dood.al/pinktrombone/) by Neil Thapen

## Result

Pink Trumpet now matches the deterministic, voiced core of Pink Trombone extremely closely under identical controls.

Across 16 steady tract states, the mean diagnostic parity index improved from **59.7 to 99.95 / 100**. Mean aligned waveform correlation improved from **0.718 to 0.9998**, mean log-spectral error fell from **11.57 dB to 0.0096 dB**, and median absolute gain error fell from **2.95 dB to 0.0003 dB**.

| Metric | Before alignment | After alignment |
| --- | ---: | ---: |
| Mean parity index | 59.72 / 100 | **99.95 / 100** |
| Median parity index | 60.22 / 100 | **99.98 / 100** |
| Mean waveform correlation | 0.7182 | **0.9998** |
| Mean spectral-shape cosine | 0.7752 | **1.0000** |
| Mean log-spectral error | 11.5714 dB | **0.0096 dB** |
| Median absolute gain error | 2.9532 dB | **0.0003 dB** |

The parity index is a regression diagnostic, not a human perceptual percentage. It combines normalized spectral error (50%), aligned waveform correlation (20%), amplitude-envelope similarity (15%), and spectral cosine similarity (15%).

## Per-state results

| State | Parity / 100 | Waveform corr. | Spectrum cosine | Spectral error | Gain Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Neutral | 100.000 | 1.0000 | 1.0000 | 0.0000 dB | 0.0000 dB |
| Open /ɑ/, 105 Hz | 99.981 | 0.9998 | 1.0000 | 0.0019 dB | 0.0000 dB |
| Open /ɑ/, 220 Hz | 99.944 | 0.9986 | 1.0000 | 0.0039 dB | +0.0001 dB |
| Bright /i/ | 99.978 | 0.9999 | 1.0000 | 0.0041 dB | 0.0000 dB |
| Rounded /u/ | 99.514 | 1.0000 | 1.0000 | 0.1167 dB | +0.0014 dB |
| Mid /e/ | 99.960 | 0.9993 | 1.0000 | 0.0042 dB | 0.0000 dB |
| Low/back | 99.971 | 0.9995 | 1.0000 | 0.0019 dB | 0.0000 dB |
| Nasal /m/ | 99.993 | 1.0000 | 1.0000 | 0.0004 dB | 0.0000 dB |
| Seeded states (mean) | 99.976 | 0.9999 | 1.0000 | 0.0026 dB | — |

## Listening pairs

Each file is the analyzed 0.6-second segment, peak-normalized independently.

| State | Original | Pink Trumpet |
| --- | --- | --- |
| Neutral | [listen](./audio/neutral-original.wav) | [listen](./audio/neutral-current.wav) |
| Bright /i/ | [listen](./audio/bright_i-original.wav) | [listen](./audio/bright_i-current.wav) |
| Seeded sample 05 | [listen](./audio/seeded_05-original.wav) | [listen](./audio/seeded_05-current.wav) |
| Seeded sample 06 | [listen](./audio/seeded_06-original.wav) | [listen](./audio/seeded_06-current.wav) |

## Changes that closed the gap

1. Restored the original tongue curve, including its `gridOffset` and endpoint shaping.
2. Replaced the nonlinear `tanh` output stage with the original linear `0.125` scaling.
3. Emulated Pink Trombone's 512-sample logical synthesis block across the browser's 128-sample AudioWorklet callbacks.
4. Restored tract movement speed to 15 and separated mouth (`0.999`) from nasal (`1.0`) propagation damping.
5. Matched the original nasal-reflection initialization order. This removed the final nasal-state outlier.
6. Restored the original 500 Hz aspiration and 1 kHz turbulence filters, modulation amplitudes, and initial control values for the live sound path.

These changes modify the only Pink Trumpet engine. There is no alternate or compatibility mode, and the public parameter API is unchanged.

## What was compared

- The original page's own inline synthesis code was executed headlessly; the reference DSP was not reimplemented.
- Both engines ran at 48 kHz with identical pitch, tenseness, intensity, tongue, constriction, and velum targets.
- Each state settled for 0.45 seconds before a 0.6-second segment was measured.
- The sweep included eight named tract shapes and eight reproducible pseudo-random states (`seed = 20260813`).
- Vibrato, wobble, aspiration, turbulence, and random modulation were disabled to isolate the glottal source and vocal-tract waveguide.
- Signals were gain-normalized and aligned within ±20 ms for shape metrics. Gain difference was measured before normalization.

The reference HTML SHA-256 was `94f92caa426e9e7bd928ab82c7d0d75d297e1090411867f62dd09081f4e1cec2`; the tested Pink Trumpet DSP SHA-256 was `eb403771d2ddeae7129db16b68c7d998a2fdeac158f50b59f911d995da25c951`.

## Remaining scope

This result establishes near-exact parity for steady deterministic voicing, including oral and nasal tract shapes. The live engine now uses the original filter frequencies and modulation amplitudes, but this report does not yet establish statistical parity for random aspiration, turbulence, vibrato/wobble, or rapidly changing gestures. Those require seeded statistical and transition tests because the original uses nondeterministic Web Audio noise.

Exploratory LPC formants remain outside the parity index. Estimates were available for both engines in 75% of states and agreed closely when stable, but the estimator can mistake the fundamental for F1 in synthetic spectra.

## Next validation pass

1. Compare vowel transitions, stop closures, and release transients.
2. Compare seeded spectral distributions for the restored 500 Hz aspiration and 1 kHz turbulence paths.
3. Compare pitch modulation over longer recordings.
4. Use this deterministic suite as a regression gate while developing the phoneme discriminator and feedback loop.

## Reproduce

Download the original page, then run:

```bash
npm run compare:original -- --original /path/to/pink-trombone-original.html --artifacts comparison
```

The complete per-state parameters and measurements are in [`results.json`](./results.json). The harness is in [`compare.py`](./compare.py), [`render-original.mjs`](./render-original.mjs), and [`render-current.mjs`](./render-current.mjs).
