import process from "node:process";

const [encodedParameters] = process.argv.slice(2);
if (!encodedParameters) throw new Error("Usage: node render-current.mjs BASE64_JSON");
const parameters = JSON.parse(Buffer.from(encodedParameters, "base64url").toString("utf8"));

globalThis.sampleRate = 48_000;
let Processor;
globalThis.AudioWorkletProcessor = class {
  constructor() {
    this.port = { onmessage: null, postMessage() {} };
  }
};
globalThis.registerProcessor = (_name, processorClass) => { Processor = processorClass; };
await import("../src/pink-trombone-worklet.js?comparison");

const processor = new Processor();
processor.glottis.noise.sample = () => 0;
Object.assign(processor.parameters, parameters, {
  aspiration: 0,
  fricativeIntensity: 0,
  vibrato: 0,
  wobble: 0,
});
processor.targets = { ...processor.parameters };
processor.smoothing = 1;

function render(sampleCount) {
  const output = new Float32Array(sampleCount);
  const blockLength = 128;
  let cursor = 0;
  while (cursor < sampleCount) {
    const block = new Float32Array(blockLength);
    processor.process([], [[block]]);
    const count = Math.min(blockLength, sampleCount - cursor);
    output.set(block.subarray(0, count), cursor);
    cursor += count;
  }
  return output;
}

render(Math.round(sampleRate * parameters.warmupSeconds));
const audio = render(Math.round(sampleRate * parameters.durationSeconds));
process.stdout.write(Buffer.from(audio.buffer, audio.byteOffset, audio.byteLength));
