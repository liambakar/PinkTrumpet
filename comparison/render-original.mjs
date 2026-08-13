import fs from "node:fs";
import process from "node:process";
import vm from "node:vm";

const [htmlPath, encodedParameters] = process.argv.slice(2);
if (!htmlPath || !encodedParameters) {
  throw new Error("Usage: node render-original.mjs ORIGINAL_HTML BASE64_JSON");
}

const parameters = JSON.parse(Buffer.from(encodedParameters, "base64url").toString("utf8"));
const html = fs.readFileSync(htmlPath, "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/i)?.[1];
if (!script) throw new Error("Could not find the original inline Pink Trombone script.");

function makeCanvasContext() {
  const methods = new Set([
    "arc", "beginPath", "clearRect", "closePath", "fill", "fillRect", "fillText",
    "lineTo", "moveTo", "rect", "restore", "rotate", "save", "scale", "stroke",
    "strokeRect", "strokeText", "translate",
  ]);
  return new Proxy({}, {
    get(target, property) {
      if (property === "measureText") return () => ({ width: 0 });
      if (methods.has(property)) return () => {};
      return target[property];
    },
    set(target, property, value) {
      target[property] = value;
      return true;
    },
  });
}

const canvases = new Map();
function makeCanvas() {
  const context = makeCanvasContext();
  return {
    width: 600,
    height: 600,
    style: {},
    offsetLeft: 0,
    offsetTop: 0,
    addEventListener() {},
    getContext() { return context; },
  };
}

class FakeAudioNode {
  connect() { return this; }
  disconnect() {}
  start() {}
}

class FakeAudioContext {
  constructor() {
    this.sampleRate = 48_000;
    this.destination = new FakeAudioNode();
  }
  createScriptProcessor() { return new FakeAudioNode(); }
  createBiquadFilter() {
    return Object.assign(new FakeAudioNode(), { frequency: { value: 0 }, Q: { value: 0 } });
  }
  createBuffer(_channels, frameCount) {
    return { getChannelData: () => new Float32Array(frameCount) };
  }
  createBufferSource() { return new FakeAudioNode(); }
}

const context = {
  console,
  Date,
  Float32Array,
  Float64Array,
  Math: Object.create(Math),
  navigator: { userAgent: "node comparison harness" },
  requestAnimationFrame() {},
  document: {
    body: { style: {} },
    addEventListener() {},
    getElementById(id) {
      if (!canvases.has(id)) canvases.set(id, makeCanvas());
      return canvases.get(id);
    },
  },
  innerWidth: 600,
  innerHeight: 600,
  AudioContext: FakeAudioContext,
  webkitAudioContext: FakeAudioContext,
};
context.window = context;
vm.createContext(context);
vm.runInContext(script, context, { filename: "pink-trombone-original.js" });

const glottis = context.Glottis;
const tract = context.Tract;
const tractUI = context.TractUI;
const blockLength = 512;
const sampleRate = 48_000;

context.autoWobble = false;
context.alwaysVoice = true;
context.noise.simplex1 = () => 0;
glottis.oldFrequency = parameters.pitchHz;
glottis.newFrequency = parameters.pitchHz;
glottis.UIFrequency = parameters.pitchHz;
glottis.smoothFrequency = parameters.pitchHz;
glottis.oldTenseness = parameters.tenseness;
glottis.newTenseness = parameters.tenseness;
glottis.UITenseness = parameters.tenseness;
glottis.vibratoAmount = 0;
glottis.intensity = parameters.intensity;
glottis.loudness = parameters.loudness * parameters.voicing;
glottis.isTouched = true;
glottis.timeInWaveform = 0;
glottis.totalTime = 0;
glottis.setupWaveform(0);

tractUI.tongueIndex = parameters.tongueIndex;
tractUI.tongueDiameter = parameters.tongueDiameter;
tractUI.setRestDiameter();
for (let i = 0; i < tract.n; i += 1) tract.targetDiameter[i] = tract.restDiameter[i];

const index = Math.min(tract.n - 3, Math.max(2, parameters.constrictionIndex));
const diameter = Math.min(3.5, Math.max(0, parameters.constrictionDiameter));
const center = Math.round(index);
const width = index < 25 ? 10 : index >= tract.tipStart ? 5 : 10 - 5 * (index - 25) / (tract.tipStart - 25);
for (let offset = -Math.ceil(width) - 1; offset < width + 1; offset += 1) {
  const section = center + offset;
  if (section < 0 || section >= tract.n) continue;
  const relative = Math.abs(section - index) - 0.5;
  const shrink = relative <= 0 ? 0 : relative > width ? 1 : 0.5 * (1 - Math.cos(Math.PI * relative / width));
  if (diameter < tract.targetDiameter[section]) {
    tract.targetDiameter[section] = diameter + (tract.targetDiameter[section] - diameter) * shrink;
  }
}
tract.velumTarget = parameters.velum;

function render(sampleCount) {
  const output = new Float32Array(sampleCount);
  let cursor = 0;
  while (cursor < sampleCount) {
    const count = Math.min(blockLength, sampleCount - cursor);
    for (let j = 0; j < count; j += 1) {
      const lambda1 = j / blockLength;
      const lambda2 = (j + 0.5) / blockLength;
      const glottalOutput = glottis.runStep(lambda1, 0);
      tract.runStep(glottalOutput, 0, lambda1);
      let value = tract.lipOutput + tract.noseOutput;
      tract.runStep(glottalOutput, 0, lambda2);
      value += tract.lipOutput + tract.noseOutput;
      output[cursor + j] = value * 0.125;
    }
    glottis.finishBlock();
    tract.finishBlock();
    cursor += count;
  }
  return output;
}

render(Math.round(sampleRate * parameters.warmupSeconds));
const audio = render(Math.round(sampleRate * parameters.durationSeconds));
process.stdout.write(Buffer.from(audio.buffer, audio.byteOffset, audio.byteLength));
