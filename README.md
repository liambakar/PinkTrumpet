# Pink Trumpet

A parameter-first, model-ready recreation of [Neil Thapen's Pink Trombone](https://dood.al/pinktrombone/). The original synthesizer's glottal source and 44-section digital waveguide have been moved into an `AudioWorklet`, while its mouse-only canvas state has become an explicit, guarded JavaScript API.

**Hosted synthesizer:** [liambakar.github.io/PinkTrumpet](https://liambakar.github.io/PinkTrumpet/)

The hosted GitHub Pages demo supports interactive synthesis and WAV rendering. Adversarial phoneme search requires the local Python scoring service described below because GitHub Pages is static hosting.

## Run it

Create the local Python environment once. It contains the spectral scorer and keeps its packages isolated from the rest of your machine.

```bash
npm run setup
npm run dev
```

Open [http://localhost:4173](http://localhost:4173), then select **Start voice**. Browsers require that user gesture before audio may play.

Run the parameter tests with:

```bash
npm test
```

The latest reproducible comparison against the original implementation is in
[`comparison/REPORT.md`](./comparison/REPORT.md), with matched listening samples and raw metrics.

The setup command downloads `skfda.datasets.fetch_phoneme`, trains a cached discriminator, and reports held-out accuracy. The first run needs network access; later runs use `.cache/phoneme-discriminator.joblib`.

## Programmatic control

The page exposes its controller as `window.pinkTrumpet`:

```js
await pinkTrumpet.start();

pinkTrumpet.setParameters({
  pitchHz: 180,
  tenseness: 0.72,
  tongueIndex: 25.5,
  tongueDiameter: 3.1,
  constrictionIndex: 36,
  constrictionDiameter: 0.65,
  fricativeIntensity: 0.4,
  velum: 0.08,
}, { rampMs: 80 });

console.log(pinkTrumpet.state);
console.log(pinkTrumpet.vector);
console.log(pinkTrumpet.feedback); // RMS level, audio time, active state
```

Capture the exact synthesized PCM or a playable WAV:

```js
const frame = await pinkTrumpet.captureFrame({ durationMs: 32 });
// frame.samples: Float32Array, frame.sampleRate: browser sample rate

const wav = await pinkTrumpet.captureWav({ durationMs: 500 });
const url = URL.createObjectURL(wav);
```

Unknown fields are ignored, non-numeric values fall back safely, and every numeric output is clamped to its declared range before it reaches the audio thread.

## Parameter contract

| Name | Range | Meaning |
| --- | ---: | --- |
| `pitchHz` | 70–420 | Fundamental frequency |
| `intensity` | 0–1 | Glottal source amplitude |
| `tenseness` | 0–1 | Breathiness-to-tightness of the glottis |
| `loudness` | 0–1.5 | Final source scaling |
| `voicing` | 0–1 | Voiced/unvoiced blend |
| `aspiration` | 0–1 | Breath noise |
| `vibrato` | 0–1 | Regular pitch modulation |
| `wobble` | 0–1 | Irregular low-frequency pitch modulation |
| `tongueIndex` | 12–29 | Tongue position from back to front |
| `tongueDiameter` | 2.05–3.5 | Tongue height / airway shaping |
| `constrictionIndex` | 2–42 | Extra constriction position along the tract |
| `constrictionDiameter` | 0–3.5 | Extra constriction opening; zero is closed |
| `fricativeIntensity` | 0–1 | Turbulence injected at the constriction |
| `velum` | 0.01–0.45 | Nasal passage opening |

The canonical names, stable vector order, defaults, and ranges live in [`src/parameters.js`](./src/parameters.js). Import `PARAMETER_SCHEMA`, `PARAMETER_NAMES`, `parameterVector`, or `parametersFromVector` instead of duplicating the contract in training or inference code.

## Model feedback loop

`createControlLoop` accepts synchronous or asynchronous policies. Each policy receives the current named state, stable numeric vector, feedback metrics, schema, step count, and an abort signal. It can return either a partial parameter object or a complete vector.

```js
const loop = pinkTrumpet.createControlLoop({
  intervalMs: 120,
  rampMs: 100,
  policy: async ({ state, vector, feedback, schema, signal }) => {
    const response = await model.predict({ state, vector, feedback, schema, signal });
    return response.parameters; // partial objects are fine
  },
});

loop.addEventListener("step", ({ detail }) => console.log(detail));
loop.addEventListener("error", ({ detail }) => console.error(detail));

await loop.start();
// later: loop.stop();
```

The loop never overlaps policy calls: the next iteration is scheduled only after the prior prediction resolves. Stopping it aborts the provided signal and prevents a late result from changing the voice. This is suitable for a later reward loop in which a microphone model, spectral target, or human preference signal supplies the feedback.

### scikit-fda phoneme search

The included dataset has five targets: `aa`, `ao`, `dcl`, `iy`, and `sh`. It contains 4,509 log-periodograms derived from 32 ms, 512-sample frames at 16 kHz. It does **not** contain the original playable TIMIT waveforms. Pink Trumpet therefore captures its output, resamples it to 16 kHz, and converts it to the same 256-bin spectral representation before scoring.

```js
const search = pinkTrumpet.createPhonemeSearch({
  phoneme: "aa",
  rewardThreshold: 0.9, // stop successfully at 90%; null disables threshold success
  maxIterations: 240, // hard cap on real audio evaluations
  capturesPerPromisingCandidate: 3, // average the strongest candidates
});

search.addEventListener("score", ({ detail }) => {
  console.log(detail.current.score, detail.best.parameters);
});

search.addEventListener("complete", ({ detail }) => {
  console.log(detail.outcome); // "threshold-success" or "best-available"
  console.log(detail.best.score, detail.best.parameters);
});

await search.start();
// later: search.stop(); // manual stop also restores the highest-scoring parameters
```

The scorer combines a five-way MLP discriminator probability (70%) with cosine similarity to the target phoneme's mean normalized spectrum (30%). The search begins with a random safe mean and uses CMA-ES to score a population, learn productive parameter directions and correlations, and generate the next population. Each generation captures every candidate once, then captures its three strongest candidates two more times and uses their average rewards. Calls never overlap.

Searches stop either when the reward threshold is reached or when the maximum evaluation budget is exhausted. Both outcomes restore the best measured parameters. The completion report distinguishes `threshold-success` from `best-available` and includes the evaluation count, completed generations, population size, and best candidate.

This is an adversarial-reward baseline, not a differentiable GAN: Web Audio parameters cannot receive gradients from the Python discriminator. A future raw-waveform dataset can add WavLM and multi-resolution STFT rewards without changing the controller or capture API.

For training, use `PARAMETER_NAMES` as the output-head order and either normalize each field with its `min`/`max` or emit named values. Keep the sanitizer between the model and synthesizer even if the model output layer is already bounded.

## Architecture

- `src/pink-trombone-worklet.js` — real-time glottal source and vocal-tract waveguide
- `src/voice-controller.js` — audio lifecycle, parameter API, feedback metrics, and sequential policy loop
- `src/phoneme-search.js` — captured-frame scoring and black-box parameter optimization
- `src/cma-es.js` — bounded population sampling and covariance adaptation
- `src/parameters.js` — the single parameter schema, presets, clamps, and vector conversion
- `src/tract-visualizer.js` — responsive canvas display and pointer-to-parameter mapping
- `src/app.js` — UI wiring only; model code does not need it
- `ml/phoneme_discriminator.py` — scikit-fda loading, matched spectral features, discriminator, and cached model
- `server.py` — static development server and local `/api/score` endpoint

## Attribution

The synthesis method and portions of the DSP are derived from **Pink Trombone v1.1** by Neil Thapen, used under the MIT License. This project retains the license and attribution in [`LICENSE`](./LICENSE).
