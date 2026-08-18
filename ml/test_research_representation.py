from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from research_representation import engine
from research_representation.analyze_v2 import analyze, direction_metrics
from research_representation.analyze_v3 import (
    cross_validated_state_predictors,
    local_linear_field,
    smoothness_metrics,
    unit_rows,
)
from research_representation.analyze_v4 import (
    projected_features,
    separate_direction_magnitude_prediction,
)
from research_representation.analyze_v5 import dimensionality_sweep
from research_representation.analyze_v6 import (
    acoustic_measurements,
    cross_fitted_pls_score,
)
from research_representation.experiment import (
    INTERVENTIONS,
    PARAMETER_NAMES,
    experiment_variants,
    make_base_configurations,
    parse_args,
    parameters_for_variant,
)
from research_representation.experiment_v4 import MEL_BINS, log_mel_feature


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


class ExperimentThreeTests(unittest.TestCase):
    def test_smoothness_detects_a_state_aligned_field(self):
        rng = np.random.default_rng(31)
        starting = rng.normal(size=(24, 8))
        displacement = unit_rows(starting)
        metrics = smoothness_metrics(
            starting,
            displacement,
            seed=5,
            permutation_draws=200,
        )
        self.assertGreater(metrics["spearman_rho"], 0.99)
        self.assertGreater(
            metrics["nearest_state_mean_direction_cosine"],
            metrics["all_pair_mean_direction_cosine"],
        )

    def test_state_predictor_beats_a_fixed_vector_on_linear_data(self):
        rng = np.random.default_rng(37)
        starting = rng.normal(size=(30, 8))
        transform = rng.normal(size=(8, 12))
        targets = unit_rows(starting) @ transform
        result = cross_validated_state_predictors(
            starting,
            targets,
            seed=7,
            permutation_draws=200,
        )
        self.assertGreater(result["linear_state"]["mean_direction_cosine"], 0.95)
        self.assertGreater(result["linear_state"]["mse_improvement_over_fixed_vector"], 0.8)

    def test_local_linear_field_recovers_signed_slopes(self):
        rng = np.random.default_rng(41)
        field = rng.normal(size=(10, 6))
        deltas = np.asarray([-0.5, -0.2, 0.2, 0.5])
        displacements = deltas[None, :, None] * field[:, None, :]
        np.testing.assert_allclose(local_linear_field(displacements, deltas), field)


class ExperimentFourTests(unittest.TestCase):
    def test_log_mel_feature_is_finite_and_repeatable(self):
        time = np.arange(round(engine.SAMPLE_RATE * 0.2)) / engine.SAMPLE_RATE
        audio = np.sin(2 * np.pi * 220 * time)
        first = log_mel_feature(audio)
        second = log_mel_feature(audio)
        self.assertEqual(first.size % MEL_BINS, 0)
        self.assertTrue(np.all(np.isfinite(first)))
        np.testing.assert_array_equal(first, second)

    def test_projected_features_include_state_delta_interactions(self):
        rng = np.random.default_rng(43)
        starting = rng.normal(size=(10, 8))
        deltas = np.asarray([-1.0, 1.0])
        features = projected_features(
            starting,
            np.arange(8),
            np.arange(8, 10),
            deltas,
            components=3,
        )
        self.assertEqual(features.shape, (4, 11))
        self.assertFalse(np.allclose(features[0], features[1]))

    def test_separate_predictor_learns_a_state_delta_field(self):
        rng = np.random.default_rng(47)
        starting = rng.normal(size=(24, 8))
        transform = rng.normal(size=(8, 12))
        field = unit_rows(starting) @ transform
        deltas = np.asarray([-1.0, -0.5, 0.5, 1.0])
        vectors = deltas[None, :, None] * field[:, None, :]
        result = separate_direction_magnitude_prediction(
            starting,
            vectors,
            deltas,
            seed=11,
            permutation_draws=200,
            forest_trees=20,
        )
        self.assertGreater(result["direction"]["linear_unit_target_mean_cosine"], 0.9)
        self.assertGreater(result["direction"]["linear_unit_target_mean_cosine"], result["direction"]["fixed_vector_mean_cosine"])


class ExperimentFiveTests(unittest.TestCase):
    def test_supervised_projection_recovers_a_hidden_one_dimensional_state(self):
        rng = np.random.default_rng(53)
        sample_count = 120
        starting = rng.normal(size=(sample_count, 10))
        hidden_axis = rng.normal(size=10)
        response_axis = rng.normal(size=14)
        hidden_state = unit_rows(starting) @ hidden_axis
        targets = hidden_state[:, None] * response_axis[None, :]

        result = dimensionality_sweep(
            starting,
            targets,
            dimensions=(1, 2),
            random_repetitions=2,
            seed=17,
        )

        learned = result["curves"]["learned_pls"]["1"]["mean_direction_cosine"]
        pca = result["curves"]["pca"]["1"]["mean_direction_cosine"]
        random = result["curves"]["random"]["1"]["mean_direction_cosine"]
        self.assertGreater(learned, 0.9)
        self.assertGreater(learned, pca + 0.2)
        self.assertGreater(learned, random + 0.2)

    def test_dimension_sweep_rejects_unidentifiable_projection_sizes(self):
        starting = np.ones((10, 12))
        targets = np.ones((10, 4))
        with self.assertRaises(ValueError):
            dimensionality_sweep(starting, targets, dimensions=(8,), random_repetitions=1)


class ExperimentSixTests(unittest.TestCase):
    def test_cross_fitted_scalar_recovers_a_stable_hidden_axis(self):
        rng = np.random.default_rng(59)
        starting = rng.normal(size=(60, 10))
        hidden_axis = rng.normal(size=10)
        response_axis = rng.normal(size=14)
        hidden_state = unit_rows(starting) @ hidden_axis
        targets = hidden_state[:, None] * response_axis[None, :]

        result = cross_fitted_pls_score(starting, targets, seed=19)

        self.assertGreater(result["mean_fold_axis_absolute_cosine_to_full"], 0.9)
        self.assertGreater(result["cross_fitted_vs_full_score_spearman"], 0.9)

    def test_acoustic_measurements_are_ordered_and_finite(self):
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
        audio = engine.generate_audio(parameters, 0.1, seed=23)
        measurements = acoustic_measurements(audio)
        self.assertTrue(all(np.isfinite(value) for value in measurements.values()))
        self.assertGreater(measurements["formantF2Hz"], measurements["formantF1Hz"])
        self.assertGreater(measurements["spectralCentroidHz"], 0)


if __name__ == "__main__":
    unittest.main()
