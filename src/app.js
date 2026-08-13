import { DEFAULT_PARAMETERS, PARAMETER_SCHEMA, PRESETS } from "./parameters.js";
import { TractVisualizer } from "./tract-visualizer.js";
import { VoiceController } from "./voice-controller.js";

const voice = new VoiceController();
globalThis.pinkTrumpet = voice;

const $ = (selector) => document.querySelector(selector);
const controlsRoot = $("#controlGroups");
const pitchInput = $("#pitch");
const pitchValue = $("#pitchValue");
const parameterJson = $("#parameterJson");
const audioButton = $("#audioButton");
const status = $("#status");
const statusText = $("#statusText");
const loopButton = $("#loopButton");
const meterFill = $("#meterFill");
const searchButton = $("#searchButton");
const searchState = $("#searchState");
let phonemeSearch = null;
let renderedWavUrl = null;

async function checkScoringBackend() {
  try {
    const response = await fetch(new URL("api/health", document.baseURI), { cache: "no-store" });
    if (!response.ok) throw new Error("Scoring backend unavailable");
    return;
  } catch {
    searchButton.disabled = true;
    searchButton.textContent = "Local server required";
    searchButton.title = "Run npm run dev locally to use adversarial phoneme search.";
    $("#scoreNote").textContent = "The hosted demo supports synthesis and WAV rendering. Run the local Python server to use discriminator search.";
  }
}

const visualizer = new TractVisualizer($("#tractCanvas"), (partial) => {
  voice.setParameters(partial, { rampMs: 25, source: "tract" });
  clearActivePreset();
});

const visibleControls = {
  Source: ["intensity", "tenseness", "voicing", "aspiration", "vibrato", "wobble"],
  Tract: ["tongueIndex", "tongueDiameter", "constrictionIndex", "constrictionDiameter", "fricativeIntensity", "velum"],
};

for (const [group, names] of Object.entries(visibleControls)) {
  const section = document.createElement("section");
  section.className = "control-group";
  section.innerHTML = `<h3>${group}</h3>`;
  for (const name of names) {
    const spec = PARAMETER_SCHEMA[name];
    const control = document.createElement("div");
    control.className = "control";
    control.innerHTML = `
      <label for="control-${name}"><span>${spec.label}</span><output></output></label>
      <input id="control-${name}" data-parameter="${name}" type="range"
        min="${spec.min}" max="${spec.max}" step="${(spec.max - spec.min) <= 1.5 ? 0.01 : 0.1}" />`;
    section.append(control);
  }
  controlsRoot.append(section);
}

function formatValue(name, value) {
  const spec = PARAMETER_SCHEMA[name];
  if (spec.unit) return `${Math.round(value)} ${spec.unit}`;
  if (["intensity", "tenseness", "voicing", "aspiration", "vibrato", "wobble", "fricativeIntensity"].includes(name)) {
    return `${Math.round(value * 100)}%`;
  }
  return value.toFixed(2);
}

function updateRange(input, value) {
  const min = Number(input.min);
  const max = Number(input.max);
  const progress = ((value - min) / (max - min)) * 100;
  input.value = String(value);
  input.style.setProperty("--range-progress", `${progress}%`);
}

function render(parameters) {
  updateRange(pitchInput, parameters.pitchHz);
  pitchValue.value = `${Math.round(parameters.pitchHz)} Hz`;
  for (const input of controlsRoot.querySelectorAll("input[data-parameter]")) {
    const name = input.dataset.parameter;
    updateRange(input, parameters[name]);
    input.closest(".control").querySelector("output").value = formatValue(name, parameters[name]);
  }
  parameterJson.textContent = JSON.stringify(parameters, null, 2);
  visualizer.setParameters(parameters);
}

function clearActivePreset() {
  stopAutomations();
  document.querySelectorAll(".preset").forEach((button) => button.classList.remove("active"));
  $("#presetTitle").textContent = "Custom voice";
}

function stopAutomations() {
  if (demoLoop?.running) demoLoop.stop();
  if (phonemeSearch?.running) phonemeSearch.stop({ restoreBest: false });
}

pitchInput.addEventListener("input", () => {
  voice.setParameters({ pitchHz: pitchInput.value }, { rampMs: 25, source: "ui" });
  clearActivePreset();
});

controlsRoot.addEventListener("input", (event) => {
  const input = event.target.closest("input[data-parameter]");
  if (!input) return;
  voice.setParameters({ [input.dataset.parameter]: input.value }, { rampMs: 30, source: "ui" });
  clearActivePreset();
});

document.querySelectorAll(".preset").forEach((button) => {
  button.addEventListener("click", () => {
    stopAutomations();
    const preset = PRESETS[button.dataset.preset];
    voice.setParameters(preset.parameters, { rampMs: 110, source: "preset" });
    document.querySelectorAll(".preset").forEach((item) => item.classList.toggle("active", item === button));
    $("#presetTitle").textContent = preset.label;
  });
});

audioButton.addEventListener("click", async () => {
  audioButton.disabled = true;
  try {
    const active = await voice.toggle();
    audioButton.classList.toggle("active", active);
    audioButton.lastChild.textContent = active ? " Stop voice" : " Start voice";
  } catch (error) {
    status.dataset.state = "error";
    statusText.textContent = "Audio unavailable";
    console.error(error);
  } finally {
    audioButton.disabled = false;
  }
});

voice.addEventListener("status", ({ detail }) => {
  status.dataset.state = detail.active ? "running" : "idle";
  statusText.textContent = detail.active ? "Engine running" : "Engine idle";
  audioButton.classList.toggle("active", detail.active);
  audioButton.lastChild.textContent = detail.active ? " Stop voice" : " Start voice";
});

voice.addEventListener("parameters", ({ detail }) => render(detail.parameters));
voice.addEventListener("feedback", ({ detail }) => {
  meterFill.style.width = `${Math.min(100, detail.rms * 520)}%`;
});

$("#resetButton").addEventListener("click", () => {
  voice.reset({ rampMs: 100 });
  clearActivePreset();
});

$("#jsonToggle").addEventListener("click", (event) => {
  const content = $("#jsonContent");
  const expanded = event.currentTarget.getAttribute("aria-expanded") === "true";
  event.currentTarget.setAttribute("aria-expanded", String(!expanded));
  content.hidden = expanded;
});

$("#copyButton").addEventListener("click", async (event) => {
  await navigator.clipboard.writeText(parameterJson.textContent);
  event.currentTarget.textContent = "Copied";
  setTimeout(() => { event.currentTarget.textContent = "Copy JSON"; }, 1000);
});

const demoLoop = voice.createControlLoop({
  intervalMs: 90,
  rampMs: 120,
  policy: ({ step }) => {
    const time = step * 0.055;
    return {
      tongueIndex: 20.5 + Math.sin(time * 0.73) * 7.1,
      tongueDiameter: 2.74 + Math.cos(time * 0.51) * 0.55,
      constrictionIndex: 35 + Math.sin(time * 0.31) * 5,
      constrictionDiameter: 2.8 + Math.sin(time * 0.47) * 0.55,
      pitchHz: 132 + Math.sin(time * 0.21) * 24,
      velum: 0.01 + Math.max(0, Math.sin(time * 0.17)) ** 8 * 0.24,
    };
  },
});

loopButton.addEventListener("click", async () => {
  if (demoLoop.running) demoLoop.stop();
  else {
    clearActivePreset();
    await demoLoop.start();
  }
});

demoLoop.addEventListener("status", ({ detail }) => {
  loopButton.classList.toggle("active", detail.running);
  loopButton.textContent = detail.running ? "Stop unscored demo" : "Run unscored demo";
});

demoLoop.addEventListener("error", ({ detail }) => console.error("Control loop error", detail));

function percentage(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function renderScore({ current, best }) {
  $("#bestScore").textContent = percentage(best.score);
  $("#discriminatorScore").value = percentage(current.discriminatorProbability);
  $("#centroidScore").value = percentage(current.centroidSimilarity);
  $("#discriminatorBar").style.width = percentage(current.discriminatorProbability);
  $("#centroidBar").style.width = percentage(current.centroidSimilarity);
  $("#scoreNote").textContent = `Iteration ${current.iteration + 1} · predicted ${current.predictedPhoneme} · target ${current.phoneme}`;
}

searchButton.addEventListener("click", async () => {
  if (phonemeSearch?.running) {
    phonemeSearch.stop();
    return;
  }
  if (demoLoop.running) demoLoop.stop();
  clearActivePreset();
  const thresholdInput = $("#rewardThreshold");
  const thresholdPercent = Math.min(100, Math.max(1, Number(thresholdInput.value) || 90));
  thresholdInput.value = String(thresholdPercent);
  const rewardThreshold = thresholdPercent / 100;
  phonemeSearch = voice.createPhonemeSearch({
    phoneme: $("#phonemeSelect").value,
    rewardThreshold,
  });
  phonemeSearch.addEventListener("score", ({ detail }) => renderScore(detail));
  phonemeSearch.addEventListener("status", ({ detail }) => {
    searchButton.classList.toggle("active", detail.running);
    searchButton.textContent = detail.running ? "Stop and keep best" : "Start phoneme search";
    searchState.classList.toggle("running", detail.running);
    searchState.classList.remove("complete");
    searchState.lastChild.textContent = detail.running ? ` Searching for /${detail.phoneme}/` : " Search stopped";
    thresholdInput.disabled = detail.running;
  });
  phonemeSearch.addEventListener("complete", ({ detail }) => {
    searchState.classList.add("complete");
    searchState.lastChild.textContent = ` Target reached at ${percentage(detail.best.score)}`;
    $("#scoreNote").textContent = `Stopped automatically: best reward met the ${percentage(detail.threshold)} threshold.`;
  });
  phonemeSearch.addEventListener("error", ({ detail }) => {
    $("#scoreNote").textContent = detail.message;
    phonemeSearch.stop();
    console.error("Phoneme search error", detail);
  });
  await phonemeSearch.start();
});

$("#renderWavButton").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  if (phonemeSearch?.running) phonemeSearch.stop();
  button.disabled = true;
  button.textContent = "Rendering…";
  try {
    const wav = await voice.captureWav({ durationMs: 700 });
    if (renderedWavUrl) URL.revokeObjectURL(renderedWavUrl);
    renderedWavUrl = URL.createObjectURL(wav);
    const audio = $("#candidateAudio");
    audio.src = renderedWavUrl;
    audio.hidden = false;
    const download = $("#downloadWav");
    download.href = renderedWavUrl;
    download.download = `pink-trumpet-${$("#phonemeSelect").value}.wav`;
    download.hidden = false;
    $("#scoreNote").textContent = "Rendered 700 ms from the retained parameters.";
  } finally {
    button.disabled = false;
    button.textContent = "Render current WAV";
  }
});

render(DEFAULT_PARAMETERS);
checkScoringBackend();
