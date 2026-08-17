import test from "node:test";
import assert from "node:assert/strict";
import { CmaEsOptimizer } from "../src/cma-es.js";

function seededRandom(seed = 1) {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

test("CMA-ES candidates stay inside normalized parameter bounds", () => {
  const optimizer = new CmaEsOptimizer({ mean: [0.1, 0.5, 0.9], sigma: 0.4, random: seededRandom(7) });
  const population = optimizer.ask();
  assert.equal(population.length, optimizer.populationSize);
  for (const candidate of population) {
    assert.equal(candidate.values.length, 3);
    assert.ok(candidate.values.every((value) => value >= 0 && value <= 1));
  }
});

test("CMA-ES learns a direction on a smooth objective", () => {
  const target = [0.18, 0.72, 0.36, 0.84];
  const optimizer = new CmaEsOptimizer({
    mean: [0.85, 0.15, 0.8, 0.2],
    sigma: 0.24,
    populationSize: 10,
    random: seededRandom(19),
  });
  const distance = (values) => values.reduce((sum, value, index) => sum + (value - target[index]) ** 2, 0);
  const initialDistance = distance(optimizer.mean);
  for (let generation = 0; generation < 45; generation += 1) {
    const population = optimizer.ask();
    optimizer.tell(population.map((candidate) => ({ candidate, score: -distance(candidate.values) })));
  }
  assert.ok(distance(optimizer.mean) < initialDistance * 0.01);
});

test("CMA-ES rejects incomplete generations", () => {
  const optimizer = new CmaEsOptimizer({ mean: [0.5, 0.5], random: seededRandom(3) });
  const population = optimizer.ask();
  assert.throws(() => optimizer.tell(population.slice(1).map((candidate) => ({ candidate, score: 0 }))), /expected/);
});
