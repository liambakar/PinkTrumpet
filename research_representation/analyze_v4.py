"""Experiment 4: explain why tongue-position displacement is harder to predict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

try:
    from .analyze_v2 import DISPLAY_NAMES, holm_adjust, row_cosine
    from .analyze_v3 import (
        jackknife_spearman_interval,
        mantel_spearman_pvalue,
        paired_sign_flip_pvalue,
        unit_rows,
        upper_triangle,
    )
except ImportError:  # Direct script execution.
    from analyze_v2 import DISPLAY_NAMES, holm_adjust, row_cosine
    from analyze_v3 import (
        jackknife_spearman_interval,
        mantel_spearman_pvalue,
        paired_sign_flip_pvalue,
        unit_rows,
        upper_triangle,
    )


DEFAULT_EMBEDDINGS = Path(__file__).with_name("experiment_v2.npz")
DEFAULT_ACOUSTICS = Path(__file__).with_name("acoustic_v4.npz")
ALPHAS = (1e-6, 1e-4, 1e-2, 1.0, 100.0, 10_000.0)


def coefficient_of_variation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1) / max(values.mean(), 1e-12))


def robust_coefficient_of_variation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    first, median, third = np.quantile(values, (0.25, 0.5, 0.75))
    return float((third - first) / max(median, 1e-12))


def bootstrap_spearman_interval(
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
    draws: int = 5_000,
) -> list[float]:
    values = []
    for _ in range(draws):
        indices = rng.integers(0, left.size, size=left.size)
        statistic = spearmanr(left[indices], right[indices]).statistic
        if np.isfinite(statistic):
            values.append(statistic)
    return [float(value) for value in np.quantile(values, (0.025, 0.975))]


def block_permutation_spearman(
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
    draws: int = 10_000,
) -> tuple[float, float]:
    """Correlate matrices while permuting whole starting configurations."""
    observed = float(spearmanr(left.ravel(), right.ravel()).statistic)
    extreme = 0
    for _ in range(draws):
        order = rng.permutation(left.shape[0])
        statistic = spearmanr(left[order].ravel(), right.ravel()).statistic
        extreme += abs(statistic) >= abs(observed)
    return observed, float((extreme + 1) / (draws + 1))


def experiment_arrays(embedding_path: Path) -> dict:
    checkpoint = np.load(embedding_path, allow_pickle=False)
    if not np.all(checkpoint["completed"].astype(bool)):
        raise ValueError("Experiment 2 checkpoint is incomplete.")
    embeddings = checkpoint["embeddings"].astype(np.float64)
    parameters = checkpoint["variant_parameters"].astype(str)
    deltas = checkpoint["variant_deltas"].astype(float)
    starting = embeddings[:, 0, :]
    return {
        "starting": starting,
        "displacements": embeddings[:, 1:, :] - starting[:, None, :],
        "parameters": parameters[1:],
        "deltas": deltas[1:],
        "base_parameters": checkpoint["base_parameters"].astype(np.float64),
        "parameter_names": checkpoint["parameter_names"].astype(str),
    }


def largest_displacements(experiment: dict, parameter: str) -> tuple[np.ndarray, float]:
    keep = experiment["parameters"] == parameter
    deltas = experiment["deltas"][keep]
    index = int(np.argmax(deltas))
    return experiment["displacements"][:, keep, :][:, index, :], float(deltas[index])


def acoustic_variability_analysis(
    experiment: dict,
    acoustics: np.lib.npyio.NpzFile,
    seed: int,
    permutation_draws: int,
) -> dict:
    acoustic_parameters = acoustics["intervention_parameters"].astype(str)
    base_features = acoustics["base_features"].astype(np.float64)
    intervention_features = acoustics["intervention_features"].astype(np.float64)
    output = {}
    acoustic_norm_matrix = []
    hubert_norm_matrix = []
    for index, parameter in enumerate(acoustic_parameters):
        acoustic_vectors = intervention_features[:, index, :] - base_features
        hubert_vectors, delta = largest_displacements(experiment, parameter)
        acoustic_norms = np.linalg.norm(acoustic_vectors, axis=1) / np.sqrt(acoustic_vectors.shape[1])
        hubert_norms = np.linalg.norm(hubert_vectors, axis=1)
        correlation = spearmanr(acoustic_norms, hubert_norms)
        acoustic_norm_matrix.append((acoustic_norms - acoustic_norms.mean()) / max(acoustic_norms.std(), 1e-12))
        hubert_norm_matrix.append((hubert_norms - hubert_norms.mean()) / max(hubert_norms.std(), 1e-12))
        output[parameter] = {
            "delta": delta,
            "acoustic_norm_mean": float(acoustic_norms.mean()),
            "acoustic_norm_std": float(acoustic_norms.std(ddof=1)),
            "acoustic_norm_cv": coefficient_of_variation(acoustic_norms),
            "acoustic_norm_robust_cv": robust_coefficient_of_variation(acoustic_norms),
            "hubert_norm_mean": float(hubert_norms.mean()),
            "hubert_norm_std": float(hubert_norms.std(ddof=1)),
            "hubert_norm_cv": coefficient_of_variation(hubert_norms),
            "hubert_norm_robust_cv": robust_coefficient_of_variation(hubert_norms),
            "acoustic_hubert_norm_spearman": float(correlation.statistic),
            "acoustic_hubert_norm_spearman_95ci": bootstrap_spearman_interval(
                acoustic_norms,
                hubert_norms,
                np.random.default_rng(seed + index),
            ),
            "acoustic_hubert_norm_pvalue": float(correlation.pvalue),
            "acoustic_mean_pairwise_direction_cosine": float(upper_triangle(
                unit_rows(acoustic_vectors) @ unit_rows(acoustic_vectors).T
            ).mean()),
            "hubert_mean_pairwise_direction_cosine": float(upper_triangle(
                unit_rows(hubert_vectors) @ unit_rows(hubert_vectors).T
            ).mean()),
            "acoustic_norms": acoustic_norms.tolist(),
            "hubert_norms": hubert_norms.tolist(),
        }
    acoustic_norm_matrix = np.column_stack(acoustic_norm_matrix)
    hubert_norm_matrix = np.column_stack(hubert_norm_matrix)
    pooled_rho, pooled_p = block_permutation_spearman(
        acoustic_norm_matrix,
        hubert_norm_matrix,
        np.random.default_rng(seed + 100),
        permutation_draws,
    )
    acoustic_cvs = np.asarray([output[name]["acoustic_norm_cv"] for name in acoustic_parameters])
    hubert_cvs = np.asarray([output[name]["hubert_norm_cv"] for name in acoustic_parameters])
    return {
        "per_control": output,
        "pooled_within_control_norm_spearman": pooled_rho,
        "pooled_base_block_permutation_pvalue": pooled_p,
        "across_control_cv_spearman_exploratory": float(spearmanr(acoustic_cvs, hubert_cvs).statistic),
    }


def similarity_matrices(experiment: dict, base_acoustics: np.ndarray) -> dict[str, np.ndarray]:
    hubert = unit_rows(experiment["starting"])
    physical = experiment["base_parameters"]
    physical = (physical - physical.mean(axis=0)) / np.maximum(physical.std(axis=0), 1e-12)
    frames = base_acoustics.shape[1] // 40
    acoustic_summary = base_acoustics.reshape(base_acoustics.shape[0], frames, 40).mean(axis=1)
    acoustic_summary = (
        acoustic_summary - acoustic_summary.mean(axis=0)
    ) / np.maximum(acoustic_summary.std(axis=0), 1e-12)
    return {
        "hubert": hubert @ hubert.T,
        "articulatory": -cdist(physical, physical) / np.sqrt(physical.shape[1]),
        "acoustic": -cdist(acoustic_summary, acoustic_summary) / np.sqrt(acoustic_summary.shape[1]),
    }


def neighborhood_analysis(
    experiment: dict,
    base_acoustics: np.ndarray,
    seed: int,
    permutation_draws: int,
) -> dict:
    similarities = similarity_matrices(experiment, base_acoustics)
    output = {}
    raw_pvalues = {}
    comparison_pvalues = {}
    for parameter_index, parameter in enumerate(sorted(set(experiment["parameters"].tolist()))):
        vectors, delta = largest_displacements(experiment, parameter)
        direction_matrix = unit_rows(vectors) @ unit_rows(vectors).T
        output[parameter] = {"delta": delta, "metrics": {}}
        for metric_index, (metric, state_matrix) in enumerate(similarities.items()):
            rho = float(spearmanr(upper_triangle(state_matrix), upper_triangle(direction_matrix)).statistic)
            nearest_matrix = np.array(state_matrix, copy=True)
            np.fill_diagonal(nearest_matrix, -np.inf)
            nearest = np.argmax(nearest_matrix, axis=1)
            nearest_cosines = row_cosine(vectors, vectors[nearest])
            pvalue = mantel_spearman_pvalue(
                state_matrix,
                direction_matrix,
                np.random.default_rng(seed + parameter_index * 10 + metric_index),
                permutation_draws,
            )
            key = f"{parameter}:{metric}"
            raw_pvalues[key] = pvalue
            output[parameter]["metrics"][metric] = {
                "spearman_rho": rho,
                "spearman_jackknife_95ci": jackknife_spearman_interval(state_matrix, direction_matrix),
                "mantel_permutation_pvalue": pvalue,
                "nearest_neighbor_mean_direction_cosine": float(nearest_cosines.mean()),
                "nearest_neighbor_median_direction_cosine": float(np.median(nearest_cosines)),
                "nearest_neighbor_cosines": nearest_cosines.tolist(),
            }
        output[parameter]["comparisons"] = {}
        for left, right in (
            ("hubert", "articulatory"),
            ("hubert", "acoustic"),
            ("acoustic", "articulatory"),
        ):
            left_values = np.asarray(output[parameter]["metrics"][left]["nearest_neighbor_cosines"])
            right_values = np.asarray(output[parameter]["metrics"][right]["nearest_neighbor_cosines"])
            difference = left_values - right_values
            comparison_key = f"{parameter}:{left}-vs-{right}"
            pvalue = paired_sign_flip_pvalue(
                difference,
                np.random.default_rng(seed + 500 + parameter_index * 10 + len(output[parameter]["comparisons"])),
                permutation_draws,
            )
            comparison_pvalues[comparison_key] = pvalue
            output[parameter]["comparisons"][f"{left}_vs_{right}"] = {
                "mean_direction_cosine_difference": float(difference.mean()),
                "paired_sign_flip_pvalue": pvalue,
            }
    adjusted = holm_adjust(raw_pvalues)
    for parameter, values in output.items():
        for metric, result in values["metrics"].items():
            result["holm_pvalue"] = adjusted[f"{parameter}:{metric}"]
    comparison_adjusted = holm_adjust(comparison_pvalues)
    for parameter, values in output.items():
        for comparison, result in values["comparisons"].items():
            left, right = comparison.split("_vs_")
            result["holm_pvalue"] = comparison_adjusted[f"{parameter}:{left}-vs-{right}"]
    return output


def projected_features(
    starting: np.ndarray,
    fit_bases: np.ndarray,
    evaluate_bases: np.ndarray,
    deltas: np.ndarray,
    components: int = 15,
) -> np.ndarray:
    normalized = unit_rows(starting)
    pca = PCA(
        n_components=min(components, fit_bases.size - 1, normalized.shape[1]),
        whiten=True,
        svd_solver="full",
    )
    pca.fit(normalized[fit_bases])
    state = pca.transform(normalized[evaluate_bases])
    repeated = np.repeat(state, deltas.size, axis=0)
    tiled_deltas = np.tile(deltas, evaluate_bases.size)
    scale = max(float(np.max(np.abs(deltas))), 1e-12)
    signed = tiled_deltas[:, None] / scale
    absolute = np.abs(signed)
    return np.column_stack((
        repeated,
        signed,
        absolute,
        repeated * signed,
        repeated * absolute,
    ))


def choose_separate_alphas(
    starting: np.ndarray,
    vectors: np.ndarray,
    deltas: np.ndarray,
    training_bases: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    inner = KFold(n_splits=min(4, training_bases.size), shuffle=True, random_state=seed)
    direction_scores = {alpha: [] for alpha in ALPHAS}
    magnitude_scores = {alpha: [] for alpha in ALPHAS}
    for inner_train_index, validation_index in inner.split(training_bases):
        inner_train = training_bases[inner_train_index]
        validation = training_bases[validation_index]
        train_x = projected_features(starting, inner_train, inner_train, deltas)
        validation_x = projected_features(starting, inner_train, validation, deltas)
        train_vectors = vectors[inner_train].reshape(-1, vectors.shape[-1])
        validation_vectors = vectors[validation].reshape(-1, vectors.shape[-1])
        train_direction = unit_rows(train_vectors)
        validation_direction = unit_rows(validation_vectors)
        train_log_magnitude = np.log(np.maximum(np.linalg.norm(train_vectors, axis=1), 1e-12))
        validation_log_magnitude = np.log(np.maximum(np.linalg.norm(validation_vectors, axis=1), 1e-12))
        for alpha in ALPHAS:
            direction_model = Ridge(alpha=alpha, solver="svd").fit(train_x, train_direction)
            direction_prediction = unit_rows(direction_model.predict(validation_x))
            direction_scores[alpha].append(mean_direction_cosine(validation_direction, direction_prediction))
            magnitude_model = Ridge(alpha=alpha, solver="svd").fit(train_x, train_log_magnitude)
            magnitude_scores[alpha].append(-float(np.mean(
                (validation_log_magnitude - magnitude_model.predict(validation_x)) ** 2
            )))
    direction_alpha = max(ALPHAS, key=lambda alpha: np.mean(direction_scores[alpha]))
    magnitude_alpha = max(ALPHAS, key=lambda alpha: np.mean(magnitude_scores[alpha]))
    return float(direction_alpha), float(magnitude_alpha)


def mean_direction_cosine(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(row_cosine(actual, predicted).mean())


def separate_direction_magnitude_prediction(
    starting: np.ndarray,
    vectors: np.ndarray,
    deltas: np.ndarray,
    seed: int,
    permutation_draws: int,
    forest_trees: int = 200,
) -> dict:
    sample_count, delta_count, dimension = vectors.shape
    outer = KFold(n_splits=min(5, sample_count), shuffle=True, random_state=seed)
    fixed = np.empty_like(vectors)
    direction_prediction = np.empty_like(vectors)
    linear_magnitude_prediction = np.empty((sample_count, delta_count), dtype=float)
    nonlinear_magnitude_prediction = np.empty((sample_count, delta_count), dtype=float)
    fixed_magnitude_prediction = np.empty((sample_count, delta_count), dtype=float)
    selected_alphas = []
    all_bases = np.arange(sample_count)

    for fold, (train_index, test_index) in enumerate(outer.split(all_bases)):
        train_bases = all_bases[train_index]
        test_bases = all_bases[test_index]
        train_vectors = vectors[train_bases].reshape(-1, dimension)
        train_magnitudes = np.linalg.norm(train_vectors, axis=1)
        train_x = projected_features(starting, train_bases, train_bases, deltas)
        test_x = projected_features(starting, train_bases, test_bases, deltas)
        direction_alpha, magnitude_alpha = choose_separate_alphas(
            starting, vectors, deltas, train_bases, seed + fold + 1
        )
        selected_alphas.append({"direction": direction_alpha, "magnitude": magnitude_alpha})

        direction_model = Ridge(alpha=direction_alpha, solver="svd")
        direction_model.fit(train_x, unit_rows(train_vectors))
        direction_prediction[test_bases] = unit_rows(direction_model.predict(test_x)).reshape(
            test_bases.size, delta_count, dimension
        )
        log_magnitudes = np.log(np.maximum(train_magnitudes, 1e-12))
        magnitude_model = Ridge(alpha=magnitude_alpha, solver="svd").fit(train_x, log_magnitudes)
        linear_magnitude_prediction[test_bases] = np.exp(magnitude_model.predict(test_x)).reshape(
            test_bases.size, delta_count
        )
        nonlinear_model = RandomForestRegressor(
            n_estimators=forest_trees,
            min_samples_leaf=5,
            max_features=0.7,
            random_state=seed + fold,
            n_jobs=1,
        ).fit(train_x, log_magnitudes)
        nonlinear_magnitude_prediction[test_bases] = np.exp(nonlinear_model.predict(test_x)).reshape(
            test_bases.size, delta_count
        )
        for delta_index in range(delta_count):
            fixed[test_bases, delta_index] = vectors[train_bases, delta_index].mean(axis=0)
            fixed_magnitude_prediction[test_bases, delta_index] = np.linalg.norm(
                vectors[train_bases, delta_index], axis=1
            ).mean()

    actual_flat = vectors.reshape(-1, dimension)
    direction_flat = direction_prediction.reshape(-1, dimension)
    fixed_flat = fixed.reshape(-1, dimension)
    actual_magnitude = np.linalg.norm(vectors, axis=-1)
    linear_vector = direction_prediction * linear_magnitude_prediction[:, :, None]
    nonlinear_vector = direction_prediction * nonlinear_magnitude_prediction[:, :, None]
    fixed_squared_error = np.sum((vectors - fixed) ** 2, axis=-1)
    linear_squared_error = np.sum((vectors - linear_vector) ** 2, axis=-1)
    nonlinear_squared_error = np.sum((vectors - nonlinear_vector) ** 2, axis=-1)
    direction_improvement_by_base = (
        row_cosine(actual_flat, direction_flat) - row_cosine(actual_flat, fixed_flat)
    ).reshape(sample_count, delta_count).mean(axis=1)
    return {
        "split": "five-fold held-out starting configurations; all magnitudes from a base remain together",
        "direction": {
            "fixed_vector_mean_cosine": mean_direction_cosine(actual_flat, fixed_flat),
            "linear_unit_target_mean_cosine": mean_direction_cosine(actual_flat, direction_flat),
            "linear_vs_fixed_base_sign_flip_pvalue": paired_sign_flip_pvalue(
                direction_improvement_by_base,
                np.random.default_rng(seed + 500),
                permutation_draws,
            ),
        },
        "magnitude": {
            "fixed_mean_r2": float(r2_score(actual_magnitude.ravel(), fixed_magnitude_prediction.ravel())),
            "fixed_mean_mae": float(mean_absolute_error(actual_magnitude.ravel(), fixed_magnitude_prediction.ravel())),
            "linear_r2": float(r2_score(actual_magnitude.ravel(), linear_magnitude_prediction.ravel())),
            "linear_mae": float(mean_absolute_error(actual_magnitude.ravel(), linear_magnitude_prediction.ravel())),
            "nonlinear_r2": float(r2_score(actual_magnitude.ravel(), nonlinear_magnitude_prediction.ravel())),
            "nonlinear_mae": float(mean_absolute_error(actual_magnitude.ravel(), nonlinear_magnitude_prediction.ravel())),
        },
        "recombined_vector": {
            "linear_mse_improvement_over_fixed": float(1 - linear_squared_error.mean() / fixed_squared_error.mean()),
            "nonlinear_mse_improvement_over_fixed": float(1 - nonlinear_squared_error.mean() / fixed_squared_error.mean()),
        },
        "selected_ridge_alphas": selected_alphas,
    }


def separated_prediction_analysis(
    experiment: dict,
    seed: int,
    permutation_draws: int,
    forest_trees: int,
) -> dict:
    output = {}
    pvalues = {}
    for index, parameter in enumerate(sorted(set(experiment["parameters"].tolist()))):
        keep = experiment["parameters"] == parameter
        result = separate_direction_magnitude_prediction(
            experiment["starting"],
            experiment["displacements"][:, keep, :],
            experiment["deltas"][keep],
            seed + index,
            permutation_draws,
            forest_trees,
        )
        output[parameter] = result
        pvalues[parameter] = result["direction"]["linear_vs_fixed_base_sign_flip_pvalue"]
    adjusted = holm_adjust(pvalues)
    for parameter in output:
        output[parameter]["direction"]["holm_pvalue"] = adjusted[parameter]
    return output


def analyze(
    embedding_path: Path,
    acoustic_path: Path,
    seed: int = 20260818,
    permutation_draws: int = 10_000,
    forest_trees: int = 200,
) -> dict:
    experiment = experiment_arrays(embedding_path)
    acoustics = np.load(acoustic_path, allow_pickle=False)
    if not np.all(acoustics["completed"].astype(bool)):
        raise ValueError("Experiment 4 acoustic checkpoint is incomplete.")
    return {
        "experiment": "why-tongue-position-is-harder-v4",
        "sample_count": int(experiment["starting"].shape[0]),
        "embedding_dimension": int(experiment["starting"].shape[1]),
        "acoustic_feature_dimension": int(acoustics["base_features"].shape[1]),
        "acoustic_variability": acoustic_variability_analysis(
            experiment, acoustics, seed, permutation_draws
        ),
        "separated_prediction": separated_prediction_analysis(
            experiment, seed + 1_000, permutation_draws, forest_trees
        ),
        "neighborhoods": neighborhood_analysis(
            experiment,
            acoustics["base_features"].astype(np.float64),
            seed + 2_000,
            permutation_draws,
        ),
    }


def markdown_report(analysis: dict, embedding_metadata: dict, acoustic_metadata: dict) -> str:
    acoustic_rows = []
    prediction_rows = []
    neighborhood_rows = []
    for parameter in sorted(analysis["acoustic_variability"]["per_control"]):
        acoustic = analysis["acoustic_variability"]["per_control"][parameter]
        prediction = analysis["separated_prediction"][parameter]
        acoustic_rows.append(
            f"| `{parameter}` | {acoustic['acoustic_norm_cv']:.3f} | {acoustic['hubert_norm_cv']:.3f} | "
            f"{acoustic['acoustic_hubert_norm_spearman']:.3f} | "
            f"{acoustic['acoustic_mean_pairwise_direction_cosine']:.3f} | "
            f"{acoustic['hubert_mean_pairwise_direction_cosine']:.3f} |"
        )
        prediction_rows.append(
            f"| `{parameter}` | {prediction['direction']['fixed_vector_mean_cosine']:.3f} | "
            f"{prediction['direction']['linear_unit_target_mean_cosine']:.3f} | "
            f"{prediction['magnitude']['fixed_mean_r2']:.3f} | {prediction['magnitude']['linear_r2']:.3f} | "
            f"{prediction['magnitude']['nonlinear_r2']:.3f} | "
            f"{prediction['recombined_vector']['nonlinear_mse_improvement_over_fixed']:.3f} | "
            f"{prediction['direction']['holm_pvalue']:.4g} |"
        )
        metrics = analysis["neighborhoods"][parameter]["metrics"]
        neighborhood_rows.append(
            f"| `{parameter}` | {metrics['hubert']['nearest_neighbor_mean_direction_cosine']:.3f} | "
            f"{metrics['articulatory']['nearest_neighbor_mean_direction_cosine']:.3f} | "
            f"{metrics['acoustic']['nearest_neighbor_mean_direction_cosine']:.3f} | "
            f"{metrics['hubert']['spearman_rho']:.3f} | {metrics['articulatory']['spearman_rho']:.3f} | "
            f"{metrics['acoustic']['spearman_rho']:.3f} |"
        )
    tongue_acoustic = analysis["acoustic_variability"]["per_control"]["tongueIndex"]
    tongue_prediction = analysis["separated_prediction"]["tongueIndex"]
    tongue_neighborhood = analysis["neighborhoods"]["tongueIndex"]["metrics"]
    tongue_comparisons = analysis["neighborhoods"]["tongueIndex"]["comparisons"]
    most_variable_acoustic = max(
        analysis["acoustic_variability"]["per_control"],
        key=lambda parameter: analysis["acoustic_variability"]["per_control"][parameter]["acoustic_norm_cv"],
    )
    smallest_tongue_neighborhood_p = min(
        comparison["holm_pvalue"] for comparison in tongue_comparisons.values()
    )
    return f"""# Experiment 4: Why Is Tongue Position Harder to Predict?

## Question

Is tongue-position displacement difficult because its acoustic consequences are unusually variable, because direction and magnitude require different models, or because HuBERT similarity is not the best neighborhood geometry?

## Data and safeguards

- Starting configurations: {analysis['sample_count']}
- HuBERT extractor: `{embedding_metadata.get('model_id', 'unknown')}` at `{embedding_metadata.get('resolved_revision', 'unknown')}`
- Acoustic representation: {acoustic_metadata.get('feature', {}).get('name', 'unknown')} ({analysis['acoustic_feature_dimension']} dimensions)
- Acoustic comparisons reuse Experiment 2's synthesis seed, duration, warmup, and paired-noise design.
- Every learned prediction holds out complete starting configurations and keeps every magnitude from a base in the same fold.

![Acoustic and HuBERT displacement variability](./FIGURE_V4_ACOUSTIC.png)

## 1. Is acoustic change unusually variable?

| Transformation | Acoustic CV | HuBERT CV | Norm correlation ρ | Acoustic direction cosine | HuBERT direction cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(acoustic_rows)}

Across controls and after standardizing within each control, acoustic and HuBERT displacement magnitudes have Spearman ρ **{analysis['acoustic_variability']['pooled_within_control_norm_spearman']:.3f}** with a base-block permutation p-value of **{analysis['acoustic_variability']['pooled_base_block_permutation_pvalue']:.4g}**. The correlation between the five acoustic CVs and five HuBERT CVs is exploratory because there are only five controls.

![Separate direction and magnitude prediction](./FIGURE_V4_SEPARATED.png)

## 2. Do direction and magnitude require different models?

| Transformation | Fixed direction | Linear direction | Fixed magnitude R² | Linear magnitude R² | Nonlinear magnitude R² | Recombined MSE gain | Direction Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(prediction_rows)}

Direction models fit unit displacement vectors. Magnitude models predict `log ‖Δz‖`; the nonlinear model is a random forest. Inputs contain a training-fold PCA of the starting HuBERT state plus signed and absolute intervention size. Ridge penalties are selected only inside each training fold.

![Neighborhood geometry comparison](./FIGURE_V4_NEIGHBORHOODS.png)

## 3. Which neighborhood best predicts the local direction?

| Transformation | HuBERT nearest | Articulatory nearest | Acoustic nearest | HuBERT ρ | Articulatory ρ | Acoustic ρ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(neighborhood_rows)}

Articulatory distance is standardized Euclidean distance over the five controlled base parameters. Acoustic distance is standardized Euclidean distance over the time-mean 40-bin log-mel spectrum. HuBERT distance is cosine distance.

## Tongue-position result

- Acoustic magnitude CV: **{tongue_acoustic['acoustic_norm_cv']:.3f}**; HuBERT magnitude CV: **{tongue_acoustic['hubert_norm_cv']:.3f}**.
- Direction prediction improves from **{tongue_prediction['direction']['fixed_vector_mean_cosine']:.3f}** to **{tongue_prediction['direction']['linear_unit_target_mean_cosine']:.3f}**.
- Magnitude R² is **{tongue_prediction['magnitude']['linear_r2']:.3f}** for the linear model and **{tongue_prediction['magnitude']['nonlinear_r2']:.3f}** for the nonlinear model.
- Nearest-neighbor direction cosine is **{tongue_neighborhood['hubert']['nearest_neighbor_mean_direction_cosine']:.3f}** in HuBERT space, **{tongue_neighborhood['articulatory']['nearest_neighbor_mean_direction_cosine']:.3f}** in articulatory space, and **{tongue_neighborhood['acoustic']['nearest_neighbor_mean_direction_cosine']:.3f}** in acoustic space.

## Main finding

The simple acoustic-variability hypothesis is not supported: tongue position is not the most variable control in log-mel displacement magnitude; `{most_variable_acoustic}` is. Acoustic magnitude still matters—within-control acoustic and HuBERT displacement norms correlate strongly—but it does not explain why tongue position is uniquely hard.

Separating direction from magnitude clarifies the failure. Tongue direction prediction improves significantly over the fixed-vector baseline (Holm p **{tongue_prediction['direction']['holm_pvalue']:.4g}**), while nonlinear magnitude prediction improves only slightly over the fixed per-delta mean and the recombined full vector remains worse than baseline. The obstacle is therefore not magnitude alone.

HuBERT and acoustic neighborhoods are numerically similar for tongue position, and none of the three pairwise neighborhood comparisons is significant after Holm correction (smallest adjusted p **{smallest_tongue_neighborhood_p:.3f}**). These data do not show that the wrong neighborhood metric caused the difficulty.

Taken together, tongue position appears difficult because its displacement field contains residual, high-dimensional state dependence that is only partly captured by the current HuBERT-state, delta-interaction, and low-level acoustic models—not because tongue movement simply produces more variable acoustic magnitude.

## Interpretation boundary

These results compare one synthetic vocal-tract generator, one HuBERT checkpoint and layer pooling scheme, five controls, and 50 starting configurations. The acoustic metric is deliberately low-level and does not establish perceptual equivalence or articulatory causality.
"""


def plot_results(analysis: dict, acoustic_path: Path, separated_path: Path, neighborhood_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    color = "#a52562"
    labels = sorted(analysis["acoustic_variability"]["per_control"])
    display = [DISPLAY_NAMES.get(label, label) for label in labels]
    x = np.arange(len(labels))

    figure, axis = plt.subplots(figsize=(11.5, 5.8))
    acoustic_cv = [analysis["acoustic_variability"]["per_control"][label]["acoustic_norm_cv"] for label in labels]
    hubert_cv = [analysis["acoustic_variability"]["per_control"][label]["hubert_norm_cv"] for label in labels]
    axis.bar(x - 0.19, acoustic_cv, 0.38, label="Log-mel displacement", color="#4f7c8d")
    axis.bar(x + 0.19, hubert_cv, 0.38, label="HuBERT displacement", color=color)
    axis.set_xticks(x, display, rotation=15, ha="right")
    axis.set_ylabel("Coefficient of variation")
    axis.set_title("Context variability of a fixed intervention", loc="left", fontsize=17, fontweight="bold")
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(acoustic_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    fixed_direction = [analysis["separated_prediction"][label]["direction"]["fixed_vector_mean_cosine"] for label in labels]
    linear_direction = [analysis["separated_prediction"][label]["direction"]["linear_unit_target_mean_cosine"] for label in labels]
    axes[0].bar(x - 0.19, fixed_direction, 0.38, label="Fixed vector", color="#94a3b8")
    axes[0].bar(x + 0.19, linear_direction, 0.38, label="State + delta", color=color)
    axes[0].set_xticks(x, display, rotation=18, ha="right")
    axes[0].set_ylabel("Held-out direction cosine")
    axes[0].set_title("Direction", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)
    magnitude_methods = (
        ("fixed_mean_r2", "Fixed mean", "#94a3b8"),
        ("linear_r2", "Linear", "#4f7c8d"),
        ("nonlinear_r2", "Nonlinear", color),
    )
    width = 0.24
    for index, (key, name, method_color) in enumerate(magnitude_methods):
        values = [analysis["separated_prediction"][label]["magnitude"][key] for label in labels]
        axes[1].bar(x + (index - 1) * width, values, width, label=name, color=method_color)
    axes[1].axhline(0, color="#68717d", linewidth=0.9)
    axes[1].set_xticks(x, display, rotation=18, ha="right")
    axes[1].set_ylabel("Held-out magnitude R²")
    axes[1].set_title("Magnitude", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, ncol=3)
    figure.suptitle("Separate prediction of HuBERT displacement", fontsize=17, fontweight="bold")
    figure.tight_layout()
    figure.savefig(separated_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11.5, 5.8))
    neighborhood_methods = (
        ("hubert", "HuBERT", "#94a3b8"),
        ("articulatory", "Articulatory", "#4f7c8d"),
        ("acoustic", "Acoustic", color),
    )
    for index, (key, name, method_color) in enumerate(neighborhood_methods):
        values = [analysis["neighborhoods"][label]["metrics"][key]["nearest_neighbor_mean_direction_cosine"] for label in labels]
        axis.bar(x + (index - 1) * width, values, width, label=name, color=method_color)
    axis.axhline(0, color="#68717d", linewidth=0.9)
    axis.set_xticks(x, display, rotation=15, ha="right")
    axis.set_ylabel("Nearest-neighbor direction cosine")
    axis.set_title("Which geometry finds states with similar local directions?", loc="left", fontsize=17, fontweight="bold")
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    figure.savefig(neighborhood_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--acoustics", type=Path, default=DEFAULT_ACOUSTICS)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--acoustic-figure", type=Path)
    parser.add_argument("--separated-figure", type=Path)
    parser.add_argument("--neighborhood-figure", type=Path)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--forest-trees", type=int, default=200)
    args = parser.parse_args()
    if args.permutations < 100 or args.forest_trees < 10:
        parser.error("permutations must be at least 100 and forest trees at least 10")
    return args


def main() -> None:
    args = parse_args()
    embedding_path = args.embeddings.resolve()
    acoustic_path = args.acoustics.resolve()
    output_json = (args.output_json or embedding_path.with_name("analysis_v4.json")).resolve()
    report_path = (args.report or embedding_path.with_name("REPORT_V4.md")).resolve()
    acoustic_figure = (args.acoustic_figure or embedding_path.with_name("FIGURE_V4_ACOUSTIC.png")).resolve()
    separated_figure = (args.separated_figure or embedding_path.with_name("FIGURE_V4_SEPARATED.png")).resolve()
    neighborhood_figure = (args.neighborhood_figure or embedding_path.with_name("FIGURE_V4_NEIGHBORHOODS.png")).resolve()
    embedding_metadata_path = embedding_path.with_suffix(".metadata.json")
    acoustic_metadata_path = acoustic_path.with_suffix(".metadata.json")
    embedding_metadata = json.loads(embedding_metadata_path.read_text(encoding="utf8"))
    acoustic_metadata = json.loads(acoustic_metadata_path.read_text(encoding="utf8"))
    result = analyze(embedding_path, acoustic_path, args.seed, args.permutations, args.forest_trees)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    report_path.write_text(markdown_report(result, embedding_metadata, acoustic_metadata), encoding="utf8")
    plot_results(result, acoustic_figure, separated_figure, neighborhood_figure)
    print(f"Analysis: {output_json}")
    print(f"Report: {report_path}")
    print(f"Figures: {acoustic_figure}, {separated_figure}, {neighborhood_figure}")


if __name__ == "__main__":
    main()
