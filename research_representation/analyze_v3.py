"""Experiment 3: test whether HuBERT displacement fields depend smoothly on state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

try:
    from .analyze_v2 import DISPLAY_NAMES, holm_adjust, row_cosine
except ImportError:  # Direct script execution.
    from analyze_v2 import DISPLAY_NAMES, holm_adjust, row_cosine


DEFAULT_INPUT = Path(__file__).with_name("experiment_v2.npz")
ALPHAS = (1e-6, 1e-4, 1e-2, 1.0, 100.0, 10_000.0)


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def upper_triangle(values: np.ndarray) -> np.ndarray:
    return values[np.triu_indices(values.shape[0], k=1)]


def paired_sign_flip_pvalue(
    differences: np.ndarray,
    rng: np.random.Generator,
    draws: int = 10_000,
) -> float:
    """One-sided paired randomization test for a positive mean improvement."""
    differences = np.asarray(differences, dtype=float)
    observed = float(differences.mean())
    extreme = 0
    remaining = draws
    while remaining:
        count = min(1_000, remaining)
        signs = rng.choice((-1.0, 1.0), size=(count, differences.size))
        null_means = (signs * differences).mean(axis=1)
        extreme += int(np.sum(null_means >= observed))
        remaining -= count
    return float((extreme + 1) / (draws + 1))


def jackknife_spearman_interval(
    left_matrix: np.ndarray,
    right_matrix: np.ndarray,
) -> list[float]:
    sample_count = left_matrix.shape[0]
    estimates = []
    for omitted in range(sample_count):
        keep = np.arange(sample_count) != omitted
        left = upper_triangle(left_matrix[np.ix_(keep, keep)])
        right = upper_triangle(right_matrix[np.ix_(keep, keep)])
        estimates.append(float(spearmanr(left, right).statistic))
    estimates = np.asarray(estimates)
    center = float(estimates.mean())
    standard_error = np.sqrt((sample_count - 1) / sample_count * np.sum((estimates - center) ** 2))
    observed = float(spearmanr(upper_triangle(left_matrix), upper_triangle(right_matrix)).statistic)
    return [observed - 1.96 * standard_error, observed + 1.96 * standard_error]


def mantel_spearman_pvalue(
    left_matrix: np.ndarray,
    right_matrix: np.ndarray,
    rng: np.random.Generator,
    draws: int = 10_000,
) -> float:
    """One-sided label-permutation test that respects pairwise dependence."""
    triangle = np.triu_indices(left_matrix.shape[0], k=1)
    left_ranks = rankdata(left_matrix[triangle]).astype(float)
    right_ranks = rankdata(right_matrix[triangle]).astype(float)
    left_ranks -= left_ranks.mean()
    right_ranks -= right_ranks.mean()
    denominator = np.linalg.norm(left_ranks) * np.linalg.norm(right_ranks)
    observed = float(np.dot(left_ranks, right_ranks) / max(denominator, 1e-12))

    ranked_right_matrix = np.zeros_like(right_matrix, dtype=float)
    ranked_right_matrix[triangle] = right_ranks
    ranked_right_matrix[(triangle[1], triangle[0])] = right_ranks
    extreme = 0
    for _ in range(draws):
        order = rng.permutation(left_matrix.shape[0])
        permuted = ranked_right_matrix[np.ix_(order, order)][triangle]
        statistic = float(np.dot(left_ranks, permuted) / max(denominator, 1e-12))
        extreme += statistic >= observed
    return float((extreme + 1) / (draws + 1))


def quantile_curve(x: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    order = np.argsort(x)
    groups = np.array_split(order, bins)
    output = []
    for group in groups:
        values = y[group]
        output.append({
            "mean_starting_similarity": float(np.mean(x[group])),
            "mean_direction_similarity": float(np.mean(values)),
            "direction_similarity_sem": float(np.std(values, ddof=1) / np.sqrt(values.size)),
            "pair_count": int(values.size),
        })
    return output


def smoothness_metrics(
    starting_embeddings: np.ndarray,
    displacement_vectors: np.ndarray,
    seed: int = 20260817,
    permutation_draws: int = 10_000,
) -> dict:
    state_similarity = unit_rows(starting_embeddings) @ unit_rows(starting_embeddings).T
    direction_similarity = unit_rows(displacement_vectors) @ unit_rows(displacement_vectors).T
    state_pairs = upper_triangle(state_similarity)
    direction_pairs = upper_triangle(direction_similarity)
    rho = float(spearmanr(state_pairs, direction_pairs).statistic)
    nearest = np.array(state_similarity, copy=True)
    np.fill_diagonal(nearest, -np.inf)
    nearest_indices = np.argmax(nearest, axis=1)
    nearest_direction_cosines = row_cosine(
        displacement_vectors,
        displacement_vectors[nearest_indices],
    )
    return {
        "pair_count": int(state_pairs.size),
        "starting_cosine_min": float(state_pairs.min()),
        "starting_cosine_median": float(np.median(state_pairs)),
        "starting_cosine_max": float(state_pairs.max()),
        "spearman_rho": rho,
        "spearman_jackknife_95ci": jackknife_spearman_interval(state_similarity, direction_similarity),
        "mantel_permutation_pvalue": mantel_spearman_pvalue(
            state_similarity,
            direction_similarity,
            np.random.default_rng(seed),
            permutation_draws,
        ),
        "nearest_state_mean_direction_cosine": float(nearest_direction_cosines.mean()),
        "nearest_state_median_direction_cosine": float(np.median(nearest_direction_cosines)),
        "all_pair_mean_direction_cosine": float(direction_pairs.mean()),
        "binned_curve": quantile_curve(state_pairs, direction_pairs),
    }


def mean_row_cosine(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(row_cosine(actual, predicted)))


def choose_ridge_alpha(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> float:
    folds = KFold(n_splits=min(4, x.shape[0]), shuffle=True, random_state=seed)
    best_alpha = ALPHAS[0]
    best_score = -np.inf
    for alpha in ALPHAS:
        predictions = np.empty_like(y)
        for train, validation in folds.split(x):
            model = Ridge(alpha=alpha, solver="svd")
            model.fit(x[train], y[train])
            predictions[validation] = model.predict(x[validation])
        score = mean_row_cosine(y, predictions)
        if score > best_score:
            best_score = score
            best_alpha = alpha
    return float(best_alpha)


def prediction_metrics(actual: np.ndarray, predicted: np.ndarray, fixed_errors: np.ndarray) -> dict:
    cosine = row_cosine(actual, predicted)
    squared_errors = np.sum((actual - predicted) ** 2, axis=1)
    return {
        "mean_direction_cosine": float(cosine.mean()),
        "median_direction_cosine": float(np.median(cosine)),
        "direction_cosine_std": float(cosine.std()),
        "mean_squared_vector_error": float(squared_errors.mean()),
        "mse_ratio_to_fixed_vector": float(squared_errors.mean() / max(fixed_errors.mean(), 1e-12)),
        "mse_improvement_over_fixed_vector": float(1 - squared_errors.mean() / max(fixed_errors.mean(), 1e-12)),
        "mean_norm_absolute_error": float(np.mean(np.abs(
            np.linalg.norm(actual, axis=1) - np.linalg.norm(predicted, axis=1)
        ))),
    }


def cross_validated_state_predictors(
    starting_embeddings: np.ndarray,
    targets: np.ndarray,
    seed: int = 20260817,
    permutation_draws: int = 10_000,
) -> dict:
    """Compare fixed, nearest-state, and linear state-dependent predictors."""
    x = unit_rows(starting_embeddings)
    outer = KFold(n_splits=min(5, x.shape[0]), shuffle=True, random_state=seed)
    fixed_predictions = np.empty_like(targets)
    nearest_predictions = np.empty_like(targets)
    ridge_predictions = np.empty_like(targets)
    chosen_alphas = []

    for fold, (train, test) in enumerate(outer.split(x)):
        fixed_predictions[test] = targets[train].mean(axis=0)
        similarity = x[test] @ x[train].T
        nearest_predictions[test] = targets[train[np.argmax(similarity, axis=1)]]
        alpha = choose_ridge_alpha(x[train], targets[train], seed + fold + 1)
        model = Ridge(alpha=alpha, solver="svd")
        model.fit(x[train], targets[train])
        ridge_predictions[test] = model.predict(x[test])
        chosen_alphas.append(alpha)

    fixed_errors = np.sum((targets - fixed_predictions) ** 2, axis=1)
    fixed_cosine = row_cosine(targets, fixed_predictions)
    nearest_cosine = row_cosine(targets, nearest_predictions)
    ridge_cosine = row_cosine(targets, ridge_predictions)
    rng = np.random.default_rng(seed)
    output = {
        "split": "five-fold held-out starting configurations with nested ridge tuning",
        "fixed_vector": prediction_metrics(targets, fixed_predictions, fixed_errors),
        "nearest_state": prediction_metrics(targets, nearest_predictions, fixed_errors),
        "linear_state": prediction_metrics(targets, ridge_predictions, fixed_errors),
        "selected_ridge_alphas": chosen_alphas,
        "nearest_vs_fixed_direction_pvalue": paired_sign_flip_pvalue(
            nearest_cosine - fixed_cosine, rng, permutation_draws
        ),
        "linear_vs_fixed_direction_pvalue": paired_sign_flip_pvalue(
            ridge_cosine - fixed_cosine, rng, permutation_draws
        ),
    }
    return output


def local_linear_field(displacements: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """Estimate each base's through-origin local field from all signed magnitudes."""
    deltas = np.asarray(deltas, dtype=float)
    denominator = float(np.dot(deltas, deltas))
    if denominator <= 0:
        raise ValueError("At least one non-zero intervention is required.")
    return np.einsum("k,nkd->nd", deltas, displacements) / denominator


def load_experiment(input_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    checkpoint = np.load(input_path, allow_pickle=False)
    if not np.all(checkpoint["completed"].astype(bool)):
        raise ValueError("Experiment 2 checkpoint is incomplete.")
    embeddings = checkpoint["embeddings"].astype(np.float64)
    parameters = checkpoint["variant_parameters"].astype(str)
    deltas = checkpoint["variant_deltas"].astype(float)
    if parameters[0] != "base":
        raise ValueError("The first checkpoint variant must be the base state.")
    starting = embeddings[:, 0, :]
    displacements = embeddings[:, 1:, :] - starting[:, None, :]
    return starting, displacements, parameters[1:], deltas[1:]


def analyze(input_path: Path, seed: int = 20260817, permutation_draws: int = 10_000) -> dict:
    starting, displacements, parameters, deltas = load_experiment(input_path)
    smoothness = {}
    predictors = {}
    for index, parameter in enumerate(sorted(set(parameters.tolist()))):
        keep = parameters == parameter
        parameter_deltas = deltas[keep]
        parameter_displacements = displacements[:, keep, :]
        largest_index = int(np.argmax(parameter_deltas))
        targets = parameter_displacements[:, largest_index, :]
        smoothness[parameter] = smoothness_metrics(
            starting,
            targets,
            seed + index,
            permutation_draws,
        )
        predictors[parameter] = cross_validated_state_predictors(
            starting,
            targets,
            seed + 100 + index,
            permutation_draws,
        )

    smoothness_adjusted = holm_adjust({
        parameter: metrics["mantel_permutation_pvalue"]
        for parameter, metrics in smoothness.items()
    })
    predictor_adjusted = holm_adjust({
        parameter: metrics["linear_vs_fixed_direction_pvalue"]
        for parameter, metrics in predictors.items()
    })

    tongue_keep = parameters == "tongueIndex"
    tongue_field = local_linear_field(displacements[:, tongue_keep, :], deltas[tongue_keep])
    field_prediction = cross_validated_state_predictors(
        starting,
        tongue_field,
        seed + 1_000,
        permutation_draws,
    )
    return {
        "experiment": "state-dependent-displacement-fields-v3",
        "sample_count": int(starting.shape[0]),
        "embedding_dimension": int(starting.shape[1]),
        "pair_count": int(starting.shape[0] * (starting.shape[0] - 1) // 2),
        "smoothness": smoothness,
        "smoothness_holm_pvalues": smoothness_adjusted,
        "largest_positive_displacement_prediction": predictors,
        "linear_predictor_holm_pvalues": predictor_adjusted,
        "tongue_local_field_prediction": field_prediction,
    }


def markdown_report(analysis: dict, metadata: dict) -> str:
    smoothness_rows = []
    prediction_rows = []
    for parameter in sorted(analysis["smoothness"]):
        smooth = analysis["smoothness"][parameter]
        predict = analysis["largest_positive_displacement_prediction"][parameter]
        smoothness_rows.append(
            f"| `{parameter}` | {smooth['spearman_rho']:.3f} | "
            f"{smooth['spearman_jackknife_95ci'][0]:.3f}–{smooth['spearman_jackknife_95ci'][1]:.3f} | "
            f"{smooth['nearest_state_mean_direction_cosine']:.3f} | "
            f"{smooth['all_pair_mean_direction_cosine']:.3f} | "
            f"{analysis['smoothness_holm_pvalues'][parameter]:.4g} |"
        )
        prediction_rows.append(
            f"| `{parameter}` | {predict['fixed_vector']['mean_direction_cosine']:.3f} | "
            f"{predict['nearest_state']['mean_direction_cosine']:.3f} | "
            f"{predict['linear_state']['mean_direction_cosine']:.3f} | "
            f"{predict['linear_state']['mse_improvement_over_fixed_vector']:.3f} | "
            f"{analysis['linear_predictor_holm_pvalues'][parameter]:.4g} |"
        )

    tongue_smooth = analysis["smoothness"]["tongueIndex"]
    tongue_predict = analysis["largest_positive_displacement_prediction"]["tongueIndex"]
    field = analysis["tongue_local_field_prediction"]
    tongue_mse_improvement = tongue_predict["linear_state"]["mse_improvement_over_fixed_vector"]
    tongue_mse_summary = (
        f"improving MSE by **{tongue_mse_improvement:.3f}**"
        if tongue_mse_improvement >= 0
        else f"worsening MSE by **{abs(tongue_mse_improvement):.3f}**"
    )
    return f"""# Experiment 3: State-Dependent Displacement Fields in HuBERT

## Question

Does the starting HuBERT representation predict how a controlled physical intervention will move that representation?

This analysis reuses the paired embeddings from Experiment 2. It introduces no new synthesis or model inference. Every prediction is evaluated on completely held-out starting vocal-tract configurations.

## Data and safeguards

- Extractor: `{metadata.get('model_id', 'unknown')}` at `{metadata.get('resolved_revision', 'unknown')}`
- Starting configurations: {analysis['sample_count']}
- Pairwise comparisons per transformation: {analysis['pair_count']}
- Embedding dimension: {analysis['embedding_dimension']}
- Matched synthesis noise within each intervention set: {metadata.get('matched_noise_within_configuration')}
- Pairwise significance uses a label-permutation test; predictor significance uses paired sign flips. Both families use Holm correction.

![Starting-state proximity versus displacement-direction proximity](./FIGURE_V3_SMOOTHNESS.png)

## Does state proximity predict direction proximity?

| Transformation | Spearman ρ | Jackknife 95% CI | Nearest-state cosine | All-pair cosine | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(smoothness_rows)}

The nearest-state column selects, for each configuration, the other configuration with the highest starting-state cosine and then measures the cosine between their displacement vectors.

![Held-out displacement prediction](./FIGURE_V3_PREDICTION.png)

## Can starting state predict the largest positive displacement?

| Transformation | Fixed vector cosine | Nearest-state cosine | Linear state cosine | Linear MSE gain | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(prediction_rows)}

The fixed-vector baseline learns one mean displacement from each training fold. The nearest-state predictor copies the displacement of the closest training state. The linear predictor fits `Δz = Az + b`, with its ridge penalty selected inside each training fold.

## Tongue field model across all signed magnitudes

For each starting tract, a local tongue field is estimated from all signed intervention sizes using the through-origin slope

`vᵢ = Σδ δΔzᵢ(δ) / Σδ δ²`.

The held-out predictor then fits `v̂ = Az + b` and reconstructs a displacement as `Δẑ = δv̂`.

| Predictor | Mean direction cosine | MSE ratio to fixed | MSE improvement |
| --- | ---: | ---: | ---: |
| Fixed field | {field['fixed_vector']['mean_direction_cosine']:.3f} | 1.000 | 0.000 |
| Nearest state | {field['nearest_state']['mean_direction_cosine']:.3f} | {field['nearest_state']['mse_ratio_to_fixed_vector']:.3f} | {field['nearest_state']['mse_improvement_over_fixed_vector']:.3f} |
| Linear state | {field['linear_state']['mean_direction_cosine']:.3f} | {field['linear_state']['mse_ratio_to_fixed_vector']:.3f} | {field['linear_state']['mse_improvement_over_fixed_vector']:.3f} |

## Main finding

For tongue position, starting-state similarity and local direction similarity have Spearman ρ **{tongue_smooth['spearman_rho']:.3f}** (Holm p **{analysis['smoothness_holm_pvalues']['tongueIndex']:.4g}**). The nearest starting state has mean direction cosine **{tongue_smooth['nearest_state_mean_direction_cosine']:.3f}**, compared with **{tongue_smooth['all_pair_mean_direction_cosine']:.3f}** over every pair.

For the `+0.5` tongue intervention, the fixed-vector, nearest-state, and linear-state predictors reach mean held-out direction cosines **{tongue_predict['fixed_vector']['mean_direction_cosine']:.3f}**, **{tongue_predict['nearest_state']['mean_direction_cosine']:.3f}**, and **{tongue_predict['linear_state']['mean_direction_cosine']:.3f}**, respectively. The linear model's MSE change relative to the fixed vector is **{tongue_predict['linear_state']['mse_improvement_over_fixed_vector']:.3f}**.

The positive proximity result and held-out directional improvement support a smoothly state-dependent direction field rather than either a universal translation vector or unstructured variation. They do **not** yet establish accurate full-vector prediction for tongue position: the linear model improves direction while {tongue_mse_summary} relative to the fixed-vector baseline.

## Interpretation boundary

The predictors operate on synthetic Pink Trombone states and HuBERT's final-layer time-mean representation. Generalization to natural speech, other HuBERT layers, or explicitly articulatory representations remains untested.
"""


def plot_results(analysis: dict, input_path: Path, smoothness_path: Path, prediction_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    starting, displacements, parameters, deltas = load_experiment(input_path)
    state_similarity = upper_triangle(unit_rows(starting) @ unit_rows(starting).T)
    color = "#a52562"
    plt.style.use("seaborn-v0_8-whitegrid")

    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    for axis, parameter in zip(axes.ravel(), sorted(set(parameters.tolist()))):
        keep = parameters == parameter
        parameter_deltas = deltas[keep]
        vectors = displacements[:, keep, :][:, int(np.argmax(parameter_deltas)), :]
        direction_similarity = upper_triangle(unit_rows(vectors) @ unit_rows(vectors).T)
        axis.scatter(state_similarity, direction_similarity, s=9, alpha=0.12, color=color, edgecolors="none")
        curve = analysis["smoothness"][parameter]["binned_curve"]
        x = [point["mean_starting_similarity"] for point in curve]
        y = [point["mean_direction_similarity"] for point in curve]
        axis.plot(x, y, color="#173f5f", marker="o", linewidth=2, markersize=4)
        axis.set_title(DISPLAY_NAMES.get(parameter, parameter), loc="left", fontweight="bold")
        axis.set_xlabel("Starting-state cosine")
        axis.set_ylabel("Displacement-direction cosine")
        metrics = analysis["smoothness"][parameter]
        axis.text(
            0.03, 0.05,
            f"Spearman ρ = {metrics['spearman_rho']:.3f}",
            transform=axis.transAxes,
            fontsize=9,
            color="#4b5563",
        )
    axes.ravel()[-1].axis("off")
    figure.suptitle("Do nearby HuBERT states share similar intervention directions?", fontsize=17, fontweight="bold", y=0.985)
    figure.text(0.5, 0.018, "Points are all state pairs; dark markers show decile means", ha="center", fontsize=10, color="#4b5563")
    figure.tight_layout(rect=(0, 0.055, 1, 0.955))
    figure.savefig(smoothness_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    labels = sorted(analysis["largest_positive_displacement_prediction"])
    methods = (("fixed_vector", "Fixed vector"), ("nearest_state", "Nearest state"), ("linear_state", "Linear state"))
    x = np.arange(len(labels))
    width = 0.24
    figure, axis = plt.subplots(figsize=(11.5, 5.8))
    colors = ("#94a3b8", "#4f7c8d", color)
    for offset, ((key, label), method_color) in enumerate(zip(methods, colors)):
        values = [analysis["largest_positive_displacement_prediction"][parameter][key]["mean_direction_cosine"] for parameter in labels]
        axis.bar(x + (offset - 1) * width, values, width, label=label, color=method_color)
    axis.axhline(0, color="#68717d", linewidth=0.9)
    axis.set_xticks(x, [DISPLAY_NAMES.get(label, label) for label in labels], rotation=15, ha="right")
    axis.set_ylabel("Held-out mean direction cosine")
    axis.set_title("Can starting state predict the intervention displacement?", loc="left", fontsize=17, fontweight="bold")
    axis.legend(frameon=False, ncol=3)
    axis.text(0.01, -0.24, "Largest positive intervention for each control · all starting configurations held out by fold", transform=axis.transAxes, color="#4b5563", fontsize=10)
    figure.tight_layout()
    figure.savefig(prediction_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--smoothness-figure", type=Path)
    parser.add_argument("--prediction-figure", type=Path)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--permutations", type=int, default=10_000)
    args = parser.parse_args()
    if args.permutations < 100:
        parser.error("--permutations must be at least 100")
    return args


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_json = (args.output_json or input_path.with_name("analysis_v3.json")).resolve()
    report_path = (args.report or input_path.with_name("REPORT_V3.md")).resolve()
    smoothness_path = (args.smoothness_figure or input_path.with_name("FIGURE_V3_SMOOTHNESS.png")).resolve()
    prediction_path = (args.prediction_figure or input_path.with_name("FIGURE_V3_PREDICTION.png")).resolve()
    metadata_path = input_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf8")) if metadata_path.exists() else {}
    result = analyze(input_path, args.seed, args.permutations)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    report_path.write_text(markdown_report(result, metadata), encoding="utf8")
    plot_results(result, input_path, smoothness_path, prediction_path)
    print(f"Analysis: {output_json}")
    print(f"Report: {report_path}")
    print(f"Smoothness figure: {smoothness_path}")
    print(f"Prediction figure: {prediction_path}")


if __name__ == "__main__":
    main()
