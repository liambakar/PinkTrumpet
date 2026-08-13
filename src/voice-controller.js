import {
  DEFAULT_PARAMETERS,
  PARAMETER_SCHEMA,
  parameterVector,
  parametersFromVector,
  sanitizeParameters,
} from "./parameters.js";
import { PhonemeSearch } from "./phoneme-search.js";

export class VoiceController extends EventTarget {
  #context = null;
  #node = null;
  #gain = null;
  #parameters = { ...DEFAULT_PARAMETERS };
  #rms = 0;
  #started = false;
  #captureSequence = 0;
  #pendingCaptures = new Map();

  get started() {
    return this.#started;
  }

  get state() {
    return Object.freeze({ ...this.#parameters });
  }

  get vector() {
    return parameterVector(this.#parameters);
  }

  get feedback() {
    return Object.freeze({
      rms: this.#rms,
      active: this.#started && this.#context?.state === "running",
      audioTime: this.#context?.currentTime ?? 0,
      timestamp: performance.now(),
    });
  }

  async start() {
    if (!this.#context) await this.#createAudioGraph();
    await this.#context.resume();
    this.#gain.gain.cancelScheduledValues(this.#context.currentTime);
    this.#gain.gain.setTargetAtTime(1, this.#context.currentTime, 0.018);
    this.#started = true;
    this.dispatchEvent(new CustomEvent("status", { detail: { active: true } }));
  }

  async stop() {
    if (!this.#context) return;
    this.#gain.gain.cancelScheduledValues(this.#context.currentTime);
    this.#gain.gain.setTargetAtTime(0, this.#context.currentTime, 0.015);
    this.#started = false;
    this.dispatchEvent(new CustomEvent("status", { detail: { active: false } }));
  }

  async toggle() {
    if (this.#started) await this.stop();
    else await this.start();
    return this.#started;
  }

  setParameters(partial, { rampMs = 45, source = "api" } = {}) {
    const next = sanitizeParameters(partial, this.#parameters);
    const changed = Object.fromEntries(
      Object.entries(next).filter(([name, value]) => value !== this.#parameters[name]),
    );
    if (!Object.keys(changed).length) return this.state;
    this.#parameters = next;
    this.#node?.port.postMessage({ type: "parameters", parameters: changed, rampMs });
    this.dispatchEvent(new CustomEvent("parameters", { detail: { parameters: this.state, changed, source } }));
    return this.state;
  }

  setVector(vector, options) {
    return this.setParameters(parametersFromVector(vector), options);
  }

  reset(options) {
    return this.setParameters(DEFAULT_PARAMETERS, { ...options, source: options?.source ?? "reset" });
  }

  createControlLoop(options) {
    return new VoiceControlLoop(this, options);
  }

  createPhonemeSearch(options) {
    return new PhonemeSearch(this, options);
  }

  async captureFrame({ durationMs = 32 } = {}) {
    if (!this.#context || this.#context.state !== "running") await this.start();
    const safeDuration = Math.min(10_000, Math.max(8, Number(durationMs) || 32));
    const requestId = `capture-${++this.#captureSequence}`;
    const sampleCount = Math.round(this.#context.sampleRate * safeDuration / 1000);
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.#pendingCaptures.delete(requestId);
        reject(new Error("Audio capture timed out."));
      }, safeDuration + 2_000);
      this.#pendingCaptures.set(requestId, { resolve, reject, timeout });
      this.#node.port.postMessage({ type: "capture", requestId, sampleCount });
    });
  }

  async captureWav(options) {
    const frame = await this.captureFrame(options);
    return encodeMonoWav(frame.samples, frame.sampleRate);
  }

  async #createAudioGraph() {
    const Context = globalThis.AudioContext ?? globalThis.webkitAudioContext;
    if (!Context) throw new Error("This browser does not support the Web Audio API.");
    this.#context = new Context({ latencyHint: "interactive" });
    await this.#context.audioWorklet.addModule(new URL("./pink-trombone-worklet.js", import.meta.url));
    this.#node = new AudioWorkletNode(this.#context, "pink-trombone-processor", {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    this.#gain = this.#context.createGain();
    this.#gain.gain.value = 0;
    this.#node.connect(this.#gain).connect(this.#context.destination);
    this.#node.port.onmessage = ({ data }) => {
      if (data?.type === "meter") {
        this.#rms += (data.rms - this.#rms) * 0.35;
        this.dispatchEvent(new CustomEvent("feedback", { detail: this.feedback }));
      } else if (data?.type === "capture") {
        const pending = this.#pendingCaptures.get(data.requestId);
        if (!pending) return;
        clearTimeout(pending.timeout);
        this.#pendingCaptures.delete(data.requestId);
        pending.resolve({
          samples: data.samples,
          sampleRate: data.sampleRate,
          durationMs: data.samples.length / data.sampleRate * 1000,
        });
      }
    };
    this.#node.port.postMessage({ type: "parameters", parameters: this.#parameters, rampMs: 1 });
  }
}

export function encodeMonoWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeAscii = (offset, value) => {
    for (let index = 0; index < value.length; index += 1) view.setUint8(offset + index, value.charCodeAt(index));
  };
  writeAscii(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeAscii(8, "WAVE");
  writeAscii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let index = 0; index < samples.length; index += 1) {
    const value = Math.min(1, Math.max(-1, samples[index]));
    view.setInt16(44 + index * 2, value < 0 ? value * 0x8000 : value * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

export class VoiceControlLoop extends EventTarget {
  #voice;
  #policy;
  #intervalMs;
  #rampMs;
  #timer = null;
  #running = false;
  #step = 0;
  #abortController = null;

  constructor(voice, { policy, intervalMs = 120, rampMs = 90, autoStartVoice = true } = {}) {
    super();
    if (typeof policy !== "function") throw new TypeError("A control loop requires a policy function.");
    this.#voice = voice;
    this.#policy = policy;
    this.#intervalMs = Math.max(16, Number(intervalMs) || 120);
    this.#rampMs = Math.max(0, Number(rampMs) || 0);
    this.autoStartVoice = autoStartVoice;
  }

  get running() { return this.#running; }
  get stepCount() { return this.#step; }

  async start() {
    if (this.#running) return;
    this.#running = true;
    this.#abortController = new AbortController();
    if (this.autoStartVoice && !this.#voice.started) await this.#voice.start();
    this.dispatchEvent(new CustomEvent("status", { detail: { running: true } }));
    this.#tick();
  }

  stop() {
    if (!this.#running) return;
    this.#running = false;
    clearTimeout(this.#timer);
    this.#abortController?.abort();
    this.dispatchEvent(new CustomEvent("status", { detail: { running: false } }));
  }

  async #tick() {
    if (!this.#running) return;
    const startedAt = performance.now();
    const context = Object.freeze({
      state: this.#voice.state,
      vector: this.#voice.vector,
      feedback: this.#voice.feedback,
      schema: PARAMETER_SCHEMA,
      step: this.#step,
      signal: this.#abortController.signal,
    });
    try {
      const action = await this.#policy(context);
      if (!this.#running || action == null) return;
      if (Array.isArray(action) || ArrayBuffer.isView(action)) {
        this.#voice.setVector(action, { rampMs: this.#rampMs, source: "control-loop" });
      } else {
        this.#voice.setParameters(action, { rampMs: this.#rampMs, source: "control-loop" });
      }
      this.#step += 1;
      this.dispatchEvent(new CustomEvent("step", { detail: { action, step: this.#step, feedback: context.feedback } }));
    } catch (error) {
      if (error?.name !== "AbortError") this.dispatchEvent(new CustomEvent("error", { detail: error }));
    }
    if (!this.#running) return;
    const delay = Math.max(0, this.#intervalMs - (performance.now() - startedAt));
    this.#timer = setTimeout(() => this.#tick(), delay);
  }
}
