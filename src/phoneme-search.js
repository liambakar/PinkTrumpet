import { PARAMETER_SCHEMA, sanitizeParameters } from "./parameters.js";

export const PHONEMES = Object.freeze(["aa", "ao", "dcl", "iy", "sh"]);

export function normalizeRewardThreshold(value) {
  if (value == null || value === "") return null;
  const threshold = Number(value);
  if (!Number.isFinite(threshold) || threshold <= 0 || threshold > 1) {
    throw new RangeError("rewardThreshold must be greater than 0 and no greater than 1, or null to disable it.");
  }
  return threshold;
}

const SEARCH_PARAMETERS = Object.freeze([
  "pitchHz",
  "intensity",
  "tenseness",
  "voicing",
  "aspiration",
  "tongueIndex",
  "tongueDiameter",
  "constrictionIndex",
  "constrictionDiameter",
  "fricativeIntensity",
  "velum",
]);

const gaussian = () => {
  const a = Math.max(Number.EPSILON, Math.random());
  const b = Math.random();
  return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
};

const abortableDelay = (durationMs, signal) => new Promise((resolve, reject) => {
  const timer = setTimeout(resolve, durationMs);
  signal?.addEventListener("abort", () => {
    clearTimeout(timer);
    reject(new DOMException("Search stopped", "AbortError"));
  }, { once: true });
});

function randomState(base) {
  const random = {};
  for (const name of SEARCH_PARAMETERS) {
    const spec = PARAMETER_SCHEMA[name];
    random[name] = spec.min + Math.random() * (spec.max - spec.min);
  }
  random.pitchHz = 80 + Math.random() * 150;
  random.loudness = 1;
  random.vibrato = 0;
  random.wobble = 0;
  return sanitizeParameters(random, base);
}

function mutateState(base, temperature) {
  const mutation = {};
  for (const name of SEARCH_PARAMETERS) {
    const spec = PARAMETER_SCHEMA[name];
    mutation[name] = base[name] + gaussian() * (spec.max - spec.min) * temperature;
  }
  return sanitizeParameters(mutation, base);
}

export class PhonemeSearch extends EventTarget {
  #voice;
  #phoneme;
  #endpoint;
  #settleMs;
  #loop;
  #best = null;
  #last = null;
  #temperature;
  #rewardThreshold;

  constructor(voice, {
    phoneme = "aa",
    endpoint = "/api/score",
    settleMs = 140,
    intervalMs = 220,
    rampMs = 90,
    temperature = 0.12,
    rewardThreshold = 0.9,
  } = {}) {
    super();
    if (!PHONEMES.includes(phoneme)) {
      throw new TypeError(`Unsupported phoneme '${phoneme}'. Choose one of: ${PHONEMES.join(", ")}.`);
    }
    this.#voice = voice;
    this.#phoneme = phoneme;
    this.#endpoint = endpoint;
    this.#settleMs = Math.max(20, settleMs);
    this.#temperature = Math.min(0.5, Math.max(0.005, temperature));
    this.#rewardThreshold = normalizeRewardThreshold(rewardThreshold);
    this.#loop = voice.createControlLoop({
      intervalMs,
      rampMs,
      policy: (context) => this.#step(context),
    });
    this.#loop.addEventListener("status", ({ detail }) => {
      this.dispatchEvent(new CustomEvent("status", { detail: { ...detail, phoneme: this.#phoneme } }));
    });
    this.#loop.addEventListener("error", ({ detail }) => {
      this.dispatchEvent(new CustomEvent("error", { detail }));
    });
  }

  get running() { return this.#loop.running; }
  get phoneme() { return this.#phoneme; }
  get best() { return this.#best && structuredClone(this.#best); }
  get last() { return this.#last && structuredClone(this.#last); }
  get rewardThreshold() { return this.#rewardThreshold; }

  async start() {
    if (this.running) return;
    this.#best = null;
    this.#last = null;
    this.#voice.setParameters(randomState(this.#voice.state), { rampMs: 1, source: "phoneme-random-start" });
    await this.#loop.start();
  }

  stop({ restoreBest = true } = {}) {
    this.#loop.stop();
    if (restoreBest && this.#best) {
      this.#voice.setParameters(this.#best.parameters, { rampMs: 120, source: "phoneme-best" });
    }
  }

  async #step({ state, step, signal }) {
    await abortableDelay(this.#settleMs, signal);
    const frame = await this.#voice.captureFrame({ durationMs: 32 });
    const response = await fetch(this.#endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        phoneme: this.#phoneme,
        sampleRate: frame.sampleRate,
        samples: Array.from(frame.samples),
      }),
      signal,
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Scoring failed with HTTP ${response.status}.`);
    const evaluation = Object.freeze({
      ...result,
      parameters: { ...state },
      iteration: step,
    });
    this.#last = evaluation;
    if (!this.#best || evaluation.score > this.#best.score) this.#best = evaluation;
    this.dispatchEvent(new CustomEvent("score", {
      detail: { current: structuredClone(evaluation), best: this.best },
    }));

    if (this.#rewardThreshold != null && this.#best.score >= this.#rewardThreshold) {
      const completion = Object.freeze({
        reason: "reward-threshold",
        phoneme: this.#phoneme,
        threshold: this.#rewardThreshold,
        best: this.best,
      });
      this.stop();
      this.dispatchEvent(new CustomEvent("complete", { detail: completion }));
      return null;
    }

    const cooling = Math.max(0.025, this.#temperature / Math.sqrt(1 + step * 0.035));
    const restart = step > 0 && step % 24 === 0;
    return restart
      ? randomState(this.#best.parameters)
      : mutateState(this.#best.parameters, cooling);
  }
}
