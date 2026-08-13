const number = (min, max, defaultValue, label, group, unit = "") => ({
  type: "number",
  min,
  max,
  default: defaultValue,
  label,
  group,
  unit,
});

export const PARAMETER_SCHEMA = Object.freeze({
  pitchHz: number(70, 420, 140, "Pitch", "Source", "Hz"),
  intensity: number(0, 1, 1, "Intensity", "Source"),
  tenseness: number(0, 1, 0.6, "Tenseness", "Source"),
  loudness: number(0, 1.5, 1, "Loudness", "Source"),
  voicing: number(0, 1, 1, "Voicing", "Source"),
  aspiration: number(0, 1, 1, "Breath", "Source"),
  vibrato: number(0, 1, 1, "Vibrato", "Source"),
  wobble: number(0, 1, 1, "Wobble", "Source"),
  tongueIndex: number(12, 29, 12.9, "Tongue position", "Tract"),
  tongueDiameter: number(2.05, 3.5, 2.43, "Tongue height", "Tract"),
  constrictionIndex: number(2, 42, 32, "Constriction position", "Tract"),
  constrictionDiameter: number(0, 3.5, 3.5, "Airway opening", "Tract"),
  fricativeIntensity: number(0, 1, 0, "Frication", "Tract"),
  velum: number(0.01, 0.45, 0.01, "Velum opening", "Tract"),
});

export const DEFAULT_PARAMETERS = Object.freeze(
  Object.fromEntries(Object.entries(PARAMETER_SCHEMA).map(([key, spec]) => [key, spec.default])),
);

export const PRESETS = Object.freeze({
  openA: {
    label: "Open /ɑ/",
    parameters: { tongueIndex: 13, tongueDiameter: 2.55, constrictionDiameter: 3.5, velum: 0.01, fricativeIntensity: 0, voicing: 1 },
  },
  brightE: {
    label: "Bright /i/",
    parameters: { tongueIndex: 27.4, tongueDiameter: 3.15, constrictionIndex: 40, constrictionDiameter: 2.4, velum: 0.01, fricativeIntensity: 0, voicing: 1 },
  },
  roundedU: {
    label: "Rounded /u/",
    parameters: { tongueIndex: 23, tongueDiameter: 3.3, constrictionIndex: 41, constrictionDiameter: 0.9, velum: 0.01, fricativeIntensity: 0, voicing: 1 },
  },
  nasalM: {
    label: "Nasal /m/",
    parameters: { tongueIndex: 17, tongueDiameter: 2.7, constrictionIndex: 41, constrictionDiameter: 0.05, velum: 0.4, fricativeIntensity: 0, voicing: 1 },
  },
  fricativeS: {
    label: "Fricative /s/",
    parameters: { tongueIndex: 25, tongueDiameter: 2.85, constrictionIndex: 35.5, constrictionDiameter: 0.55, velum: 0.01, fricativeIntensity: 0.95, voicing: 0.06, tenseness: 0.7 },
  },
});

export function clampParameter(name, value) {
  const spec = PARAMETER_SCHEMA[name];
  if (!spec) throw new TypeError(`Unknown voice parameter: ${name}`);
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return spec.default;
  return Math.min(spec.max, Math.max(spec.min, numeric));
}

export function sanitizeParameters(partial, base = DEFAULT_PARAMETERS) {
  if (partial == null || typeof partial !== "object" || Array.isArray(partial)) {
    throw new TypeError("Voice parameters must be an object.");
  }

  const next = { ...base };
  for (const [name, value] of Object.entries(partial)) {
    if (!(name in PARAMETER_SCHEMA)) continue;
    next[name] = clampParameter(name, value);
  }
  return next;
}

export function parameterVector(parameters) {
  const safe = sanitizeParameters(parameters);
  return Object.keys(PARAMETER_SCHEMA).map((name) => safe[name]);
}

export function parametersFromVector(vector) {
  if (!Array.isArray(vector) && !ArrayBuffer.isView(vector)) {
    throw new TypeError("Parameter vector must be an array or typed array.");
  }
  const names = Object.keys(PARAMETER_SCHEMA);
  return sanitizeParameters(Object.fromEntries(names.map((name, index) => [name, vector[index]])));
}

export const PARAMETER_NAMES = Object.freeze(Object.keys(PARAMETER_SCHEMA));
