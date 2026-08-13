import test from "node:test";
import assert from "node:assert/strict";
import {
  normalizeCaptureCount,
  normalizeMaxIterations,
  normalizeRewardThreshold,
} from "../src/phoneme-search.js";

test("reward thresholds accept normalized rewards and allow disabling", () => {
  assert.equal(normalizeRewardThreshold(0.9), 0.9);
  assert.equal(normalizeRewardThreshold("0.75"), 0.75);
  assert.equal(normalizeRewardThreshold(null), null);
  assert.equal(normalizeRewardThreshold(""), null);
});

test("reward thresholds reject values outside zero to one", () => {
  assert.throws(() => normalizeRewardThreshold(0), /rewardThreshold/);
  assert.throws(() => normalizeRewardThreshold(1.01), /rewardThreshold/);
  assert.throws(() => normalizeRewardThreshold("invalid"), /rewardThreshold/);
});

test("search budgets and repeated capture counts are bounded", () => {
  assert.equal(normalizeMaxIterations("240"), 240);
  assert.equal(normalizeCaptureCount(3), 3);
  assert.throws(() => normalizeMaxIterations(0), /maxIterations/);
  assert.throws(() => normalizeMaxIterations(10_001), /maxIterations/);
  assert.throws(() => normalizeCaptureCount(0), /capturesPerPromisingCandidate/);
  assert.throws(() => normalizeCaptureCount(11), /capturesPerPromisingCandidate/);
});
