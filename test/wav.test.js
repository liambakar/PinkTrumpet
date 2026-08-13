import test from "node:test";
import assert from "node:assert/strict";
import { encodeMonoWav } from "../src/voice-controller.js";

test("encodeMonoWav writes a valid mono 16-bit WAV container", async () => {
  const samples = new Float32Array([-1, -0.5, 0, 0.5, 1]);
  const blob = encodeMonoWav(samples, 16_000);
  const bytes = new Uint8Array(await blob.arrayBuffer());
  const view = new DataView(bytes.buffer);
  const text = (start, length) => String.fromCharCode(...bytes.slice(start, start + length));
  assert.equal(blob.type, "audio/wav");
  assert.equal(text(0, 4), "RIFF");
  assert.equal(text(8, 4), "WAVE");
  assert.equal(text(36, 4), "data");
  assert.equal(view.getUint16(22, true), 1);
  assert.equal(view.getUint32(24, true), 16_000);
  assert.equal(view.getUint16(34, true), 16);
  assert.equal(view.getUint32(40, true), samples.length * 2);
});
