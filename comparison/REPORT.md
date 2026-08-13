# Pink Trumpet ↔ Pink Trombone comparison

Run date: 2026-08-13  
Reference: [Pink Trombone v1.1](https://dood.al/pinktrombone/) by Neil Thapen  
Pink Trumpet revision: `da70c06`

## Result

Pink Trumpet is recognizably derived from the same synthesizer, but it is **not yet acoustically equivalent** to the original under matched controls.

Across 16 deterministic voiced states, the mean diagnostic parity index was **59.7 / 100** (median **60.2**, range **42.8–73.2**). The average aligned waveform correlation was **0.718**, average spectral-shape cosine similarity was **0.775**, and mean log-spectral error was **11.57 dB**. Loudness also differed by a median absolute **2.95 dB**.

The parity index is a regression-oriented diagnostic, not a human perceptual percentage. It combines normalized spectral error (50%), aligned waveform correlation (20%), amplitude-envelope similarity (15%), and spectral cosine similarity (15%). It deliberately excludes the unreliable formant estimates described below.

| State | Parity / 100 | Waveform corr. | Spectrum cosine | Spectral error | Gain Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Neutral | 49.0 | 0.606 | 0.657 | 16.29 dB | +6.05 dB |
| Open /ɑ/, 105 Hz | 60.2 | 0.703 | 0.760 | 10.52 dB | +4.69 dB |
| Open /ɑ/, 220 Hz | 53.7 | 0.530 | 0.627 | 11.14 dB | +0.32 dB |
| Bright /i/ | 69.4 | 0.900 | 0.965 | 9.55 dB | +2.05 dB |
| Rounded /u/ | 66.2 | 0.821 | 0.903 | 9.94 dB | +2.82 dB |
| Mid /e/ | 46.8 | 0.577 | 0.637 | 17.02 dB | +2.10 dB |
| Low/back | 60.3 | 0.688 | 0.776 | 10.72 dB | +2.32 dB |
| Nasal /m/ | 55.9 | 0.588 | 0.617 | 10.54 dB | +5.03 dB |
| Seeded states (mean) | 61.8 | 0.760 | 0.803 | 11.18 dB | — |

## Listening pairs

Each file is the analyzed 0.6-second segment, peak-normalized independently so the comparison emphasizes timbre. The raw gain difference remains in `results.json`.

| State | Original | Pink Trumpet |
| --- | --- | --- |
| Neutral | [listen](./audio/neutral-original.wav) | [listen](./audio/neutral-current.wav) |
| Bright /i/ | [listen](./audio/bright_i-original.wav) | [listen](./audio/bright_i-current.wav) |
| Worst measured state | [listen](./audio/seeded_05-original.wav) | [listen](./audio/seeded_05-current.wav) |
| Best measured state | [listen](./audio/seeded_06-original.wav) | [listen](./audio/seeded_06-current.wav) |

## What was compared

- The original page's own inline synthesis code was executed headlessly; the reference DSP was not reimplemented for this test.
- Both engines ran at 48 kHz with identical pitch, tenseness, intensity, tongue, constriction, and velum targets.
- Each state settled for 0.45 seconds before a 0.6-second segment was measured.
- The sweep included eight named tract shapes and eight reproducible pseudo-random states (`seed = 20260813`).
- Vibrato, wobble, aspiration, turbulence, and random modulation were disabled to isolate the glottal source and vocal-tract waveguide.
- Signals were gain-normalized and aligned within ±20 ms for shape metrics. Gain difference was measured before normalization.

The reference HTML used for the run had SHA-256 `94f92caa426e9e7bd928ab82c7d0d75d297e1090411867f62dd09081f4e1cec2`. It is not committed; the harness takes a downloaded copy as input.

## Likely causes of the gap

1. **Tongue geometry differs.** The original tongue equation includes a `gridOffset` of `1.7`; Pink Trumpet currently omits it. This changes multiple tube diameters at once and is the strongest candidate for the vowel-spectrum drift.
2. **Output stages differ.** The original linearly scales the two tract passes by `0.125`; Pink Trumpet applies `tanh(value × 0.16)`. This explains much of the level mismatch and adds state-dependent harmonic distortion.
3. **Control cadence differs.** The original updates its tract every 512 samples, while the `AudioWorklet` updates every 128 samples. Movement speed is also 15 in the original and 18 in Pink Trumpet. These matter most during transitions, which this steady-state run did not measure.
4. **Nasal damping differs slightly.** Pink Trumpet applies `0.999` propagation loss throughout; the original mouth path uses `0.999`, while its nasal path uses a separate `fade` value of `1.0`.
5. **Pointer semantics are not one-to-one.** The original turns pointer diameter into an internal constriction by subtracting `0.3` and only applies it below a threshold. This harness intentionally compared common *internal* tract targets, because Pink Trumpet exposes model parameters rather than pointer coordinates.

## Formant-analysis caveat

Exploratory LPC estimates were retained in the raw results but were available for both engines in only **37.5%** of states. The synthetic spectra often did not provide three stable, speech-like resonances under the estimator's bandwidth rules, and some low estimates tracked the fundamental rather than F1. Formants therefore do not contribute to the parity index. A stronger follow-up should estimate resonances from the tract's impulse response or use cepstral/source-filter separation.

## Recommended next pass

1. Restore the original tongue geometry and linear output stage behind a `compatibilityMode` flag.
2. Re-run this exact sweep and use the parity index as a regression target; aim first for mean spectral error below **6 dB** and a mean waveform correlation above **0.9**.
3. Add dynamic tests for vowel transitions and stop releases.
4. Add a separately seeded noise suite for aspiration and `/s/`-like turbulence; filtered noise behavior cannot be judged from this deterministic pass.
5. Only after DSP parity improves, compare both engines against the `skfda` phoneme spectra. That will distinguish “matches Pink Trombone” from “matches the target phoneme dataset.”

## Reproduce

Download the original page, then run:

```bash
npm run compare:original -- --original /path/to/pink-trombone-original.html --artifacts comparison
```

The complete per-state parameters and measurements are in [`results.json`](./results.json). The comparison harness is in [`compare.py`](./compare.py), [`render-original.mjs`](./render-original.mjs), and [`render-current.mjs`](./render-current.mjs).
