import test from "node:test";
import assert from "node:assert/strict";
import { PhonemeSearch } from "../src/phoneme-search.js";
import { DEFAULT_PARAMETERS, sanitizeParameters } from "../src/parameters.js";

class FakeControlLoop extends EventTarget {
  constructor(voice, policy) {
    super();
    this.voice = voice;
    this.policy = policy;
    this.running = false;
    this.step = 0;
  }

  async start() {
    this.running = true;
    this.abortController = new AbortController();
    this.dispatchEvent(new CustomEvent("status", { detail: { running: true } }));
    while (this.running) {
      const action = await this.policy({
        state: this.voice.state,
        step: this.step,
        signal: this.abortController.signal,
      });
      if (!this.running || action == null) break;
      this.voice.setParameters(action);
      this.step += 1;
    }
  }

  stop() {
    if (!this.running) return;
    this.running = false;
    this.abortController.abort();
    this.dispatchEvent(new CustomEvent("status", { detail: { running: false } }));
  }
}

class FakeVoice {
  constructor() {
    this.state = { ...DEFAULT_PARAMETERS };
  }

  createControlLoop({ policy }) {
    return new FakeControlLoop(this, policy);
  }

  setParameters(partial) {
    this.state = sanitizeParameters(partial, this.state);
  }

  async captureFrame() {
    return { samples: new Float32Array(512), sampleRate: 16_000 };
  }
}

function fakeResponse(score) {
  return {
    ok: true,
    async json() {
      return {
        phoneme: "aa",
        score,
        discriminatorProbability: score,
        centroidSimilarity: score,
        predictedPhoneme: "aa",
      };
    },
  };
}

test("population search reports threshold success after repeated elite captures", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => fakeResponse(0.95);
  try {
    const search = new PhonemeSearch(new FakeVoice(), {
      phoneme: "aa",
      rewardThreshold: 0.9,
      maxIterations: 6,
      populationSize: 4,
      promisingCandidates: 1,
      capturesPerPromisingCandidate: 3,
      settleMs: 20,
    });
    let completion;
    search.addEventListener("complete", ({ detail }) => { completion = detail; });
    await search.start();
    assert.equal(completion.outcome, "threshold-success");
    assert.equal(completion.thresholdReached, true);
    assert.equal(completion.evaluations, 6);
    assert.equal(completion.completedGenerations, 1);
    assert.equal(completion.promisingCandidates, 1);
    assert.equal(completion.capturesPerPromisingCandidate, 3);
    assert.equal(completion.best.captures, 3);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("population search reports best available at its evaluation limit", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => fakeResponse(0.4);
  try {
    const search = new PhonemeSearch(new FakeVoice(), {
      phoneme: "aa",
      rewardThreshold: 0.9,
      maxIterations: 5,
      populationSize: 4,
      capturesPerPromisingCandidate: 1,
      settleMs: 20,
    });
    let completion;
    search.addEventListener("complete", ({ detail }) => { completion = detail; });
    await search.start();
    assert.equal(completion.outcome, "best-available");
    assert.equal(completion.thresholdReached, false);
    assert.equal(completion.evaluations, 5);
    assert.equal(completion.completedGenerations, 1);
    assert.equal(completion.best.score, 0.4);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
