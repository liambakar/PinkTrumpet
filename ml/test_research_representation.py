from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from research_representation import engine
from research_representation.analyze_v2 import analyze, direction_metrics
from research_representation.experiment import (
    INTERVENTIONS,
    PARAMETER_NAMES,
    experiment_variants,
    make_base_configurations,
    parse_args,
    parameters_for_variant,
)


class ExperimentTwoTests(unittest.TestCase):
    def test_hubert_cli_has_an_automatic_device_default(self):
        import sys

        original = sys.argv
        try:
            sys.argv = ["experiment.py"]
            self.assertEqual(parse_args().device, "auto")
        finally:
            sys.argv = original

    def test_seeded_engine_is_reproducible(self):
        parameters = {
            "pitchHz": 140,
            "tenseness": 0.6,
            "intensity": 1,
            "loudness": 1,
            "voicing": 1,
            "tongueIndex": 18,
            "tongueDiameter": 2.7,
            "constrictionIndex": 32,
            "constrictionDiameter": 3.5,
        }
        first = engine.generate_audio(parameters, 0.01, seed=17)
        second = engine.generate_audio(parameters, 0.01, seed=17)
        different = engine.generate_audio(parameters, 0.01, seed=18)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, different))

    def test_interventions_stay_inside_supported_ranges(self):
        configurations = make_base_configurations(100, seed=7)
        variants = experiment_variants()
        bounds = {
            "tongueIndex": (12, 29),
            "tongueDiameter": (2.05, 3.5),
            "constrictionIndex": (2, 42),
            "constrictionDiameter": (0, 3.5),
            "pitchHz": (70, 420),
        }
        self.assertEqual(tuple(item.name for item in INTERVENTIONS), PARAMETER_NAMES)
        for base in configurations:
            for variant in variants:
                parameters = parameters_for_variant(base, variant)
                for name, (minimum, maximum) in bounds.items():
                    self.assertGreaterEqual(parameters[name], minimum)
                    self.assertLessEqual(parameters[name], maximum)

    def test_direction_metrics_detect_a_shared_direction(self):
        rng = np.random.default_rng(4)
        direction = np.zeros(16)
        direction[3] = 1
        vectors = direction + rng.normal(0, 0.01, size=(24, 16))
        metrics = direction_metrics(vectors, seed=9)
        self.assertGreater(metrics["mean_pairwise_cosine"], 0.99)
        self.assertGreater(metrics["mean_leave_one_out_alignment"], 0.99)
        self.assertGreater(metrics["resultant_strength"], 0.99)

    def test_analysis_uses_held_out_base_groups(self):
        sample_count = 10
        dimension = 12
        directions = {
            "pitchHz": np.eye(dimension)[0],
            "tongueIndex": np.eye(dimension)[1],
            "tongueDiameter": np.eye(dimension)[2],
        }
        parameters = np.asarray([
            "base",
            *[name for name in directions for _ in range(4)],
        ])
        deltas = np.asarray([0, *[-1, -0.5, 0.5, 1] * len(directions)], dtype=float)
        rng = np.random.default_rng(11)
        base = rng.normal(size=(sample_count, dimension))
        embeddings = np.empty((sample_count, parameters.size, dimension), dtype=np.float32)
        embeddings[:, 0] = base
        for index in range(1, parameters.size):
            embeddings[:, index] = base + deltas[index] * directions[parameters[index]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.npz"
            np.savez_compressed(
                path,
                embeddings=embeddings,
                base_parameters=np.zeros((sample_count, 5)),
                parameter_names=np.asarray(PARAMETER_NAMES),
                variant_parameters=parameters,
                variant_deltas=deltas,
                completed=np.ones(sample_count, dtype=bool),
            )
            result = analyze(path, seed=13)
        self.assertAlmostEqual(result["transformation_classifier"]["balanced_accuracy"], 1.0)
        for metrics in result["magnitude_regression"].values():
            self.assertGreater(metrics["r2"], 0.9)


if __name__ == "__main__":
    unittest.main()
