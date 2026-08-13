import test from "node:test";
import assert from "node:assert/strict";
import { normalizeRewardThreshold } from "../src/phoneme-search.js";

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
