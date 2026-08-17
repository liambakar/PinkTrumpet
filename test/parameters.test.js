import test from "node:test";
import assert from "node:assert/strict";
import {
  DEFAULT_PARAMETERS,
  PARAMETER_NAMES,
  clampParameter,
  parameterVector,
  parametersFromVector,
  sanitizeParameters,
} from "../src/parameters.js";

test("sanitizeParameters clamps model output to safe ranges", () => {
  const result = sanitizeParameters({ pitchHz: 9999, velum: -5, notAParameter: 4 });
  assert.equal(result.pitchHz, 420);
  assert.equal(result.velum, 0.01);
  assert.equal("notAParameter" in result, false);
});

test("invalid numeric values fall back to the parameter default", () => {
  assert.equal(clampParameter("tenseness", Number.NaN), DEFAULT_PARAMETERS.tenseness);
  assert.equal(clampParameter("pitchHz", "not a number"), DEFAULT_PARAMETERS.pitchHz);
});

test("partial updates preserve the supplied base state", () => {
  const base = { ...DEFAULT_PARAMETERS, pitchHz: 220 };
  const result = sanitizeParameters({ tenseness: 0.2 }, base);
  assert.equal(result.pitchHz, 220);
  assert.equal(result.tenseness, 0.2);
});

test("parameter vectors round-trip in a stable order", () => {
  const state = sanitizeParameters({ pitchHz: 201, tongueIndex: 24.25, velum: 0.32 });
  const vector = parameterVector(state);
  assert.equal(vector.length, PARAMETER_NAMES.length);
  assert.deepEqual(parametersFromVector(vector), state);
});

test("unknown parameter access is rejected", () => {
  assert.throws(() => clampParameter("madeUp", 1), /Unknown voice parameter/);
});
