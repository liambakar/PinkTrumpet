/*
 * Vocal-tract DSP derived from Pink Trombone v1.1 by Neil Thapen.
 * Original work and this refactor are available under the MIT License.
 */

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const moveTowards = (current, target, up, down = up) =>
  current < target ? Math.min(current + up, target) : Math.max(current - down, target);

class SmoothNoise {
  constructor(seed = 0.731) {
    this.seed = seed;
  }

  sample(x) {
    return (
      Math.sin((x + this.seed) * 1.73) * 0.47 +
      Math.sin((x + this.seed * 7) * 0.73) * 0.31 +
      Math.sin((x + 1.9) * 0.19) * 0.22
    );
  }
}

class Glottis {
  constructor(rate) {
    this.sampleRate = rate;
    this.timeInWaveform = 0;
    this.totalTime = 0;
    this.frequency = 140;
    this.tenseness = 0.62;
    this.intensity = 0.82;
    this.loudness = 1;
    this.voicing = 1;
    this.aspiration = 0.18;
    this.vibrato = 0.18;
    this.wobble = 0;
    this.noise = new SmoothNoise();
    this.setupWaveform();
  }

  update(parameters) {
    const vibrato = parameters.vibrato * 0.012 * Math.sin(2 * Math.PI * this.totalTime * 5.8);
    const wobble = parameters.wobble * 0.035 * this.noise.sample(this.totalTime * 2.1);
    this.frequency = parameters.pitchHz * (1 + vibrato + wobble);
    this.tenseness = parameters.tenseness;
    this.intensity = parameters.intensity;
    this.loudness = parameters.loudness;
    this.voicing = parameters.voicing;
    this.aspiration = parameters.aspiration;
    this.vibrato = parameters.vibrato;
    this.wobble = parameters.wobble;
  }

  setupWaveform() {
    this.waveformLength = 1 / Math.max(40, this.frequency);
    let rd = clamp(3 * (1 - this.tenseness), 0.5, 2.7);
    const ra = -0.01 + 0.048 * rd;
    const rk = 0.224 + 0.118 * rd;
    const rg = (rk / 4) * (0.5 + 1.2 * rk) / (0.11 * rd - ra * (0.5 + 1.2 * rk));
    const ta = ra;
    const tp = 1 / (2 * rg);
    const te = tp + tp * rk;
    const epsilon = 1 / ta;
    const shift = Math.exp(-epsilon * (1 - te));
    const delta = 1 - shift;
    const rhsIntegral = ((shift - 1) / epsilon + (1 - te) * shift) / delta;
    const lowerIntegral = -(te - tp) / 2 + rhsIntegral;
    const upperIntegral = -lowerIntegral;
    const omega = Math.PI / tp;
    const s = Math.sin(omega * te);
    const y = Math.max(1e-6, (-Math.PI * s * upperIntegral) / (tp * 2));
    const alpha = Math.log(y) / (tp / 2 - te);
    this.alpha = alpha;
    this.e0 = -1 / (s * Math.exp(alpha * te));
    this.epsilon = epsilon;
    this.shift = shift;
    this.delta = delta;
    this.te = te;
    this.omega = omega;
  }

  noiseModulator() {
    const voicedPulse = 0.1 + 0.2 * Math.max(0, Math.sin(2 * Math.PI * this.timeInWaveform / this.waveformLength));
    return this.tenseness * this.intensity * voicedPulse + (1 - this.tenseness * this.intensity) * 0.3;
  }

  run(whiteNoise) {
    const step = 1 / this.sampleRate;
    this.timeInWaveform += step;
    this.totalTime += step;
    if (this.timeInWaveform >= this.waveformLength) {
      this.timeInWaveform %= this.waveformLength;
      this.setupWaveform();
    }

    const t = this.timeInWaveform / this.waveformLength;
    const lf = t > this.te
      ? (-Math.exp(-this.epsilon * (t - this.te)) + this.shift) / this.delta
      : this.e0 * Math.exp(this.alpha * t) * Math.sin(this.omega * t);
    const voiced = lf * this.intensity * this.loudness * this.voicing;
    const breath = whiteNoise * this.aspiration * this.intensity * (1 - Math.sqrt(this.tenseness)) * this.noiseModulator() * 0.42;
    return voiced + breath;
  }
}

class VocalTract {
  constructor(rate) {
    this.sampleRate = rate;
    this.n = 44;
    this.bladeStart = 10;
    this.tipStart = 32;
    this.lipStart = 39;
    this.noseLength = 28;
    this.noseStart = this.n - this.noseLength + 1;
    this.glottalReflection = 0.75;
    this.lipReflection = -0.85;
    this.fade = 0.999;
    this.movementSpeed = 18;
    this.velumTarget = 0.01;
    this.lipOutput = 0;
    this.noseOutput = 0;
    this.lastObstruction = -1;
    this.transients = [];
    this.allocate();
    this.initializeDiameters();
    this.calculateNoseReflections();
    this.calculateReflections();
  }

  allocate() {
    const n = this.n;
    const nn = this.noseLength;
    for (const name of ["R", "L", "diameter", "restDiameter", "targetDiameter", "A"])
      this[name] = new Float64Array(n);
    for (const name of ["reflection", "newReflection", "junctionOutputR", "junctionOutputL"])
      this[name] = new Float64Array(n + 1);
    for (const name of ["noseR", "noseL", "noseDiameter", "noseA"])
      this[name] = new Float64Array(nn);
    for (const name of ["noseReflection", "noseJunctionOutputR", "noseJunctionOutputL"])
      this[name] = new Float64Array(nn + 1);
  }

  initializeDiameters() {
    for (let i = 0; i < this.n; i += 1) {
      const diameter = i < 6.5 ? 0.6 : i < 12 ? 1.1 : 1.5;
      this.diameter[i] = this.restDiameter[i] = this.targetDiameter[i] = diameter;
    }
    for (let i = 0; i < this.noseLength; i += 1) {
      const d = 2 * i / this.noseLength;
      this.noseDiameter[i] = Math.min(d < 1 ? 0.4 + 1.6 * d : 0.5 + 1.5 * (2 - d), 1.9);
    }
    this.noseDiameter[0] = this.velumTarget;
  }

  setTargets(p) {
    for (let i = 0; i < this.n; i += 1) {
      let base = i < 6.5 ? 0.6 : i < 12 ? 1.1 : 1.5;
      if (i >= this.bladeStart && i < this.lipStart) {
        const t = 1.1 * Math.PI * (p.tongueIndex - i) / (this.tipStart - this.bladeStart);
        const fixed = 2 + (p.tongueDiameter - 2) / 1.5;
        let curve = (1.5 - fixed) * Math.cos(t);
        if (i === this.bladeStart || i === this.lipStart - 2) curve *= 0.94;
        base = 1.5 - curve;
      }
      this.restDiameter[i] = this.targetDiameter[i] = base;
    }

    const index = clamp(p.constrictionIndex, 2, this.n - 3);
    const diameter = clamp(p.constrictionDiameter, 0, 3.5);
    const center = Math.round(index);
    const width = index < 25 ? 10 : index >= this.tipStart ? 5 : 10 - 5 * (index - 25) / (this.tipStart - 25);
    for (let offset = -Math.ceil(width) - 1; offset <= width + 1; offset += 1) {
      const section = center + offset;
      if (section < 0 || section >= this.n) continue;
      const relative = Math.abs(section - index) - 0.5;
      const shrink = relative <= 0 ? 0 : relative > width ? 1 : 0.5 * (1 - Math.cos(Math.PI * relative / width));
      if (diameter < this.targetDiameter[section]) {
        this.targetDiameter[section] = diameter + (this.targetDiameter[section] - diameter) * shrink;
      }
    }
    this.velumTarget = p.velum;
  }

  reshape(deltaTime) {
    const amount = deltaTime * this.movementSpeed;
    let obstruction = -1;
    for (let i = 0; i < this.n; i += 1) {
      if (this.diameter[i] <= 0) obstruction = i;
      const slow = i < this.noseStart ? 0.6 : i >= this.tipStart ? 1 : 0.6 + 0.4 * (i - this.noseStart) / (this.tipStart - this.noseStart);
      this.diameter[i] = moveTowards(this.diameter[i], this.targetDiameter[i], slow * amount, 2 * amount);
    }
    if (this.lastObstruction > -1 && obstruction === -1 && this.noseA[0] < 0.05) this.addTransient(this.lastObstruction);
    this.lastObstruction = obstruction;
    this.noseDiameter[0] = moveTowards(this.noseDiameter[0], this.velumTarget, amount * 0.25, amount * 0.1);
    this.noseA[0] = this.noseDiameter[0] ** 2;
  }

  calculateReflections() {
    for (let i = 0; i < this.n; i += 1) this.A[i] = this.diameter[i] ** 2;
    for (let i = 1; i < this.n; i += 1) {
      this.reflection[i] = this.newReflection[i];
      this.newReflection[i] = this.A[i] === 0 ? 0.999 : (this.A[i - 1] - this.A[i]) / (this.A[i - 1] + this.A[i]);
    }
    this.reflectionLeft = this.newReflectionLeft ?? 0;
    this.reflectionRight = this.newReflectionRight ?? 0;
    this.reflectionNose = this.newReflectionNose ?? 0;
    const sum = Math.max(1e-6, this.A[this.noseStart] + this.A[this.noseStart + 1] + this.noseA[0]);
    this.newReflectionLeft = (2 * this.A[this.noseStart] - sum) / sum;
    this.newReflectionRight = (2 * this.A[this.noseStart + 1] - sum) / sum;
    this.newReflectionNose = (2 * this.noseA[0] - sum) / sum;
  }

  calculateNoseReflections() {
    for (let i = 0; i < this.noseLength; i += 1) this.noseA[i] = this.noseDiameter[i] ** 2;
    for (let i = 1; i < this.noseLength; i += 1) {
      this.noseReflection[i] = (this.noseA[i - 1] - this.noseA[i]) / (this.noseA[i - 1] + this.noseA[i]);
    }
  }

  addTransient(position) {
    this.transients.push({ position: clamp(position, 0, this.n - 1), age: 0 });
  }

  processTransients() {
    for (const transient of this.transients) {
      const amplitude = 0.3 * 2 ** (-200 * transient.age);
      this.R[transient.position] += amplitude / 2;
      this.L[transient.position] += amplitude / 2;
      transient.age += 1 / (this.sampleRate * 2);
    }
    this.transients = this.transients.filter((item) => item.age <= 0.2);
  }

  addTurbulence(noise, p, modulator) {
    if (p.fricativeIntensity <= 0 || p.constrictionDiameter <= 0) return;
    const index = clamp(p.constrictionIndex, 2, this.n - 3);
    const i = Math.floor(index);
    const delta = index - i;
    const thinness = clamp(8 * (0.7 - p.constrictionDiameter), 0, 1);
    const openness = clamp(30 * (p.constrictionDiameter - 0.3), 0, 1);
    const value = 0.66 * noise * p.fricativeIntensity * modulator * thinness * openness;
    const a = value * (1 - delta) / 2;
    const b = value * delta / 2;
    this.R[i + 1] += a; this.L[i + 1] += a;
    this.R[i + 2] += b; this.L[i + 2] += b;
  }

  run(glottalOutput, turbulenceNoise, lambda, parameters, noiseModulator) {
    this.processTransients();
    this.addTurbulence(turbulenceNoise, parameters, noiseModulator);
    this.junctionOutputR[0] = this.L[0] * this.glottalReflection + glottalOutput;
    this.junctionOutputL[this.n] = this.R[this.n - 1] * this.lipReflection;
    for (let i = 1; i < this.n; i += 1) {
      const r = this.reflection[i] * (1 - lambda) + this.newReflection[i] * lambda;
      const w = r * (this.R[i - 1] + this.L[i]);
      this.junctionOutputR[i] = this.R[i - 1] - w;
      this.junctionOutputL[i] = this.L[i] + w;
    }

    const junction = this.noseStart;
    let r = this.newReflectionLeft * (1 - lambda) + this.reflectionLeft * lambda;
    this.junctionOutputL[junction] = r * this.R[junction - 1] + (1 + r) * (this.noseL[0] + this.L[junction]);
    r = this.newReflectionRight * (1 - lambda) + this.reflectionRight * lambda;
    this.junctionOutputR[junction] = r * this.L[junction] + (1 + r) * (this.R[junction - 1] + this.noseL[0]);
    r = this.newReflectionNose * (1 - lambda) + this.reflectionNose * lambda;
    this.noseJunctionOutputR[0] = r * this.noseL[0] + (1 + r) * (this.L[junction] + this.R[junction - 1]);

    for (let i = 0; i < this.n; i += 1) {
      this.R[i] = this.junctionOutputR[i] * this.fade;
      this.L[i] = this.junctionOutputL[i + 1] * this.fade;
    }
    this.lipOutput = this.R[this.n - 1];

    this.noseJunctionOutputL[this.noseLength] = this.noseR[this.noseLength - 1] * this.lipReflection;
    for (let i = 1; i < this.noseLength; i += 1) {
      const w = this.noseReflection[i] * (this.noseR[i - 1] + this.noseL[i]);
      this.noseJunctionOutputR[i] = this.noseR[i - 1] - w;
      this.noseJunctionOutputL[i] = this.noseL[i] + w;
    }
    for (let i = 0; i < this.noseLength; i += 1) {
      this.noseR[i] = this.noseJunctionOutputR[i] * this.fade;
      this.noseL[i] = this.noseJunctionOutputL[i + 1] * this.fade;
    }
    this.noseOutput = this.noseR[this.noseLength - 1];
  }

  finishBlock(blockTime) {
    this.reshape(blockTime);
    this.calculateReflections();
  }
}

class PinkTromboneProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.parameters = {
      pitchHz: 140, intensity: 0.82, tenseness: 0.62, loudness: 1, voicing: 1,
      aspiration: 0.18, vibrato: 0.18, wobble: 0, tongueIndex: 17.5,
      tongueDiameter: 2.75, constrictionIndex: 32, constrictionDiameter: 3.5,
      fricativeIntensity: 0, velum: 0.01,
    };
    this.targets = { ...this.parameters };
    this.smoothing = 0.008;
    this.glottis = new Glottis(sampleRate);
    this.tract = new VocalTract(sampleRate);
    this.blockCounter = 0;
    this.capture = null;
    this.port.onmessage = ({ data }) => {
      if (data?.type === "parameters") {
        Object.assign(this.targets, data.parameters);
        const rampSeconds = clamp((data.rampMs ?? 40) / 1000, 0.001, 2);
        this.smoothing = 1 - Math.exp(-1 / (sampleRate * rampSeconds));
      } else if (data?.type === "capture") {
        const sampleCount = Math.round(clamp(data.sampleCount, 128, sampleRate * 10));
        this.capture = {
          requestId: data.requestId,
          samples: new Float32Array(sampleCount),
          offset: 0,
        };
      }
    };
  }

  process(_inputs, outputs) {
    const output = outputs[0]?.[0];
    if (!output) return true;
    let squareSum = 0;
    for (let index = 0; index < output.length; index += 1) {
      for (const key of Object.keys(this.parameters)) {
        this.parameters[key] += (this.targets[key] - this.parameters[key]) * this.smoothing;
      }
      this.glottis.update(this.parameters);
      this.tract.setTargets(this.parameters);
      const whiteNoise = Math.random() * 2 - 1;
      const glottal = this.glottis.run(whiteNoise);
      let value = 0;
      this.tract.run(glottal, whiteNoise, index / output.length, this.parameters, this.glottis.noiseModulator());
      value += this.tract.lipOutput + this.tract.noseOutput;
      this.tract.run(glottal, whiteNoise, (index + 0.5) / output.length, this.parameters, this.glottis.noiseModulator());
      value += this.tract.lipOutput + this.tract.noseOutput;
      value = Math.tanh(value * 0.16);
      output[index] = value;
      squareSum += value * value;
    }
    if (this.capture) {
      const remaining = this.capture.samples.length - this.capture.offset;
      const count = Math.min(remaining, output.length);
      this.capture.samples.set(output.subarray(0, count), this.capture.offset);
      this.capture.offset += count;
      if (this.capture.offset >= this.capture.samples.length) {
        const { requestId, samples } = this.capture;
        this.capture = null;
        this.port.postMessage(
          { type: "capture", requestId, sampleRate, samples },
          [samples.buffer],
        );
      }
    }
    this.tract.finishBlock(output.length / sampleRate);
    if (++this.blockCounter % 8 === 0) {
      this.port.postMessage({ type: "meter", rms: Math.sqrt(squareSum / output.length) });
    }
    return true;
  }
}

registerProcessor("pink-trombone-processor", PinkTromboneProcessor);
