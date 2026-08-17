import { CmaEsOptimizer } from "./cma-es.js";
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

export function normalizeMaxIterations(value) {
  const iterations = Number(value);
  if (!Number.isFinite(iterations) || iterations < 1 || iterations > 10_000) {
    throw new RangeError("maxIterations must be between 1 and 10,000.");
  }
  return Math.round(iterations);
}

export function normalizeCaptureCount(value) {
  const captures = Number(value);
  if (!Number.isFinite(captures) || captures < 1 || captures > 10) {
    throw new RangeError("capturesPerPromisingCandidate must be between 1 and 10.");
  }
  return Math.round(captures);
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

function stateToPoint(state) {
  return SEARCH_PARAMETERS.map((name) => {
    const spec = PARAMETER_SCHEMA[name];
    return (state[name] - spec.min) / (spec.max - spec.min);
  });
}

function pointToState(point, base) {
  const parameters = Object.fromEntries(SEARCH_PARAMETERS.map((name, index) => {
    const spec = PARAMETER_SCHEMA[name];
    return [name, spec.min + point[index] * (spec.max - spec.min)];
  }));
  return sanitizeParameters(parameters, base);
}

function mostFrequent(values) {
  const counts = new Map();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
  return [...counts].sort((left, right) => right[1] - left[1])[0]?.[0];
}

export class PhonemeSearch extends EventTarget {
  #voice;
  #phoneme;
  #endpoint;
  #settleMs;
  #loop;
  #best = null;
  #committedBest = null;
  #last = null;
  #temperature;
  #rewardThreshold;
  #maxIterations;
  #capturesPerPromisingCandidate;
  #promisingCandidates;
  #populationSize;
  #optimizer = null;
  #population = [];
  #jobs = [];
  #activeJob = null;
  #baseState = null;
  #resampling = false;
  #evaluations = 0;
  #report = null;

  constructor(voice, {
    phoneme = "aa",
    endpoint = "/api/score",
    settleMs = 140,
    intervalMs = 220,
    rampMs = 90,
    temperature = 0.18,
    rewardThreshold = 0.9,
    maxIterations = 240,
    capturesPerPromisingCandidate = 3,
    promisingCandidates = 3,
    populationSize,
  } = {}) {
    super();
    if (!PHONEMES.includes(phoneme)) {
      throw new TypeError(`Unsupported phoneme '${phoneme}'. Choose one of: ${PHONEMES.join(", ")}.`);
    }
    this.#voice = voice;
    this.#phoneme = phoneme;
    this.#endpoint = endpoint;
    this.#settleMs = Math.max(20, settleMs);
    this.#temperature = Math.min(0.6, Math.max(0.01, temperature));
    this.#rewardThreshold = normalizeRewardThreshold(rewardThreshold);
    this.#maxIterations = normalizeMaxIterations(maxIterations);
    this.#capturesPerPromisingCandidate = normalizeCaptureCount(capturesPerPromisingCandidate);
    this.#promisingCandidates = Math.max(1, Math.round(promisingCandidates));
    this.#populationSize = populationSize == null ? undefined : Math.max(4, Math.round(populationSize));
    this.#loop = voice.createControlLoop({
      intervalMs,
      rampMs,
      policy: (context) => this.#step(context),
    });
    this.#loop.addEventListener("status", ({ detail }) => {
      this.dispatchEvent(new CustomEvent("status", { detail: {
        ...detail,
        phoneme: this.#phoneme,
        evaluations: this.#evaluations,
        maxIterations: this.#maxIterations,
      } }));
    });
    this.#loop.addEventListener("error", ({ detail }) => {
      this.dispatchEvent(new CustomEvent("error", { detail }));
    });
  }

  get running() { return this.#loop.running; }
  get phoneme() { return this.#phoneme; }
  get best() { return this.#best && structuredClone(this.#best); }
  get last() { return this.#last && structuredClone(this.#last); }
  get report() { return this.#report && structuredClone(this.#report); }
  get rewardThreshold() { return this.#rewardThreshold; }
  get maxIterations() { return this.#maxIterations; }
  get evaluations() { return this.#evaluations; }

  async start() {
    if (this.running) return;
    this.#best = null;
    this.#committedBest = null;
    this.#last = null;
    this.#report = null;
    this.#evaluations = 0;
    this.#baseState = randomState(this.#voice.state);
    this.#optimizer = new CmaEsOptimizer({
      mean: stateToPoint(this.#baseState),
      sigma: this.#temperature,
      populationSize: this.#populationSize,
    });
    this.#beginGeneration();
    this.#activeJob = this.#jobs.shift();
    this.#voice.setParameters(this.#activeJob.member.parameters, { rampMs: 1, source: "phoneme-cma-start" });
    await this.#loop.start();
  }

  stop({ restoreBest = true } = {}) {
    this.#loop.stop();
    if (restoreBest && this.#best) {
      this.#voice.setParameters(this.#best.parameters, { rampMs: 120, source: "phoneme-best" });
    }
  }

  #beginGeneration() {
    this.#resampling = false;
    this.#population = this.#optimizer.ask().map((candidate, index) => ({
      candidate,
      index,
      generation: candidate.generation + 1,
      parameters: pointToState(candidate.values, this.#baseState),
      samples: [],
      lastEvaluation: 0,
    }));
    this.#jobs = this.#population.map((member) => ({ member, phase: "population" }));
  }

  #aggregate(member) {
    const count = member.samples.length;
    if (!count) return null;
    const average = (name) => member.samples.reduce((sum, sample) => sum + sample[name], 0) / count;
    return Object.freeze({
      phoneme: this.#phoneme,
      score: average("score"),
      discriminatorProbability: average("discriminatorProbability"),
      centroidSimilarity: average("centroidSimilarity"),
      predictedPhoneme: mostFrequent(member.samples.map((sample) => sample.predictedPhoneme)),
      parameters: { ...member.parameters },
      iteration: member.lastEvaluation - 1,
      evaluation: member.lastEvaluation,
      generation: member.generation,
      candidate: member.index + 1,
      populationSize: this.#optimizer.populationSize,
      captures: count,
    });
  }

  #refreshBest() {
    const candidates = [this.#committedBest, ...this.#population.map((member) => this.#aggregate(member))]
      .filter(Boolean);
    this.#best = candidates.reduce((best, candidate) => {
      if (!best || candidate.score > best.score + 1e-12) return candidate;
      if (Math.abs(candidate.score - best.score) <= 1e-12 && candidate.captures > best.captures) return candidate;
      return best;
    }, null);
  }

  async #captureAndScore(signal) {
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
    return result;
  }

  #scheduleResampling() {
    if (this.#capturesPerPromisingCandidate <= 1) return false;
    const promising = [...this.#population]
      .sort((left, right) => right.samples[0].score - left.samples[0].score)
      .slice(0, Math.min(this.#promisingCandidates, this.#population.length));
    this.#jobs = promising.flatMap((member) => Array.from(
      { length: this.#capturesPerPromisingCandidate - 1 },
      () => ({ member, phase: "resample" }),
    ));
    this.#resampling = this.#jobs.length > 0;
    return this.#resampling;
  }

  #complete(reason) {
    this.#refreshBest();
    const thresholdReached = reason === "reward-threshold";
    this.#report = Object.freeze({
      reason,
      outcome: thresholdReached ? "threshold-success" : "best-available",
      thresholdReached,
      phoneme: this.#phoneme,
      threshold: this.#rewardThreshold,
      evaluations: this.#evaluations,
      maxIterations: this.#maxIterations,
      completedGenerations: this.#optimizer.generation,
      populationSize: this.#optimizer.populationSize,
      promisingCandidates: Math.min(this.#promisingCandidates, this.#optimizer.populationSize),
      capturesPerPromisingCandidate: this.#capturesPerPromisingCandidate,
      best: this.best,
    });
    this.stop();
    this.dispatchEvent(new CustomEvent("complete", { detail: this.report }));
  }

  #finishGeneration() {
    const evaluations = this.#population.map((member) => ({
      candidate: member.candidate,
      score: this.#aggregate(member).score,
    }));
    this.#optimizer.tell(evaluations);
    this.#refreshBest();
    this.#committedBest = this.#best;
    this.dispatchEvent(new CustomEvent("generation", { detail: {
      generation: this.#optimizer.generation,
      evaluations: this.#evaluations,
      best: this.best,
      sigma: this.#optimizer.sigma,
    } }));
  }

  async #step({ signal }) {
    const result = await this.#captureAndScore(signal);
    this.#evaluations += 1;
    this.#activeJob.member.samples.push(result);
    this.#activeJob.member.lastEvaluation = this.#evaluations;
    const current = this.#aggregate(this.#activeJob.member);
    this.#last = current;
    this.#refreshBest();
    this.dispatchEvent(new CustomEvent("score", {
      detail: {
        current: structuredClone(current),
        best: this.best,
        phase: this.#activeJob.phase,
        evaluations: this.#evaluations,
        maxIterations: this.#maxIterations,
      },
    }));

    const generationReady = !this.#jobs.length
      && (this.#resampling || this.#capturesPerPromisingCandidate === 1);
    if (generationReady) {
      this.#finishGeneration();
      if (this.#rewardThreshold != null && this.#best.score >= this.#rewardThreshold) {
        this.#complete("reward-threshold");
        return null;
      }
      if (this.#evaluations >= this.#maxIterations) {
        this.#complete("max-iterations");
        return null;
      }
      this.#beginGeneration();
    } else if (!this.#jobs.length && this.#evaluations < this.#maxIterations) {
      this.#scheduleResampling();
    }

    if (this.#evaluations >= this.#maxIterations) {
      this.#complete("max-iterations");
      return null;
    }

    this.#activeJob = this.#jobs.shift();
    return this.#activeJob.member.parameters;
  }
}
