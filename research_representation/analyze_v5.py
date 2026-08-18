"""Experiment 5: measure the dimensionality of intervention-relevant HuBERT state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection

try:
    from .analyze_v2 import DISPLAY_NAMES, row_cosine
    from .analyze_v3 import paired_sign_flip_pvalue, unit_rows
    from .analyze_v4 import ALPHAS, experiment_arrays, largest_displacements
except ImportError:  # Direct script execution.
    from analyze_v2 import DISPLAY_NAMES, row_cosine
    from analyze_v3 import paired_sign_flip_pvalue, unit_rows
    from analyze_v4 import ALPHAS, experiment_arrays, largest_displacements


DEFAULT_INPUT = Path(__file__).with_name("experiment_v2.npz")
DIMENSIONS = (1, 2, 4, 8, 16, 32)


def bootstrap_mean_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    draws: int = 5_000,
) -> list[float]:
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def bootstrap_ratio_interval(
    numerator: np.ndarray,
    denominator: np.ndarray,
    rng: np.random.Generator,
    draws: int = 5_000,
) -> list[float]:
    indices = rng.integers(0, numerator.size, size=(draws, numerator.size))
    ratios = numerator[indices].mean(axis=1) / np.maximum(denominator[indices].mean(axis=1), 1e-12)
    return [float(value) for value in np.quantile(ratios, (0.025, 0.975))]


def prediction_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    fixed_predictions: np.ndarray,
    seed: int,
) -> dict:
    direction_cosines = row_cosine(actual, predicted)
    squared_errors = np.sum((actual - predicted) ** 2, axis=1)
    fixed_errors = np.sum((actual - fixed_predictions) ** 2, axis=1)
    return {
        "mean_direction_cosine": float(direction_cosines.mean()),
        "median_direction_cosine": float(np.median(direction_cosines)),
        "direction_cosine_95ci": bootstrap_mean_interval(
            direction_cosines, np.random.default_rng(seed)
        ),
        "mean_squared_vector_error": float(squared_errors.mean()),
        "mse_ratio_to_fixed": float(squared_errors.mean() / max(fixed_errors.mean(), 1e-12)),
        "mse_ratio_to_fixed_95ci": bootstrap_ratio_interval(
            squared_errors,
            fixed_errors,
            np.random.default_rng(seed + 1),
        ),
        "direction_cosines": direction_cosines.tolist(),
    }


def ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, float]:
    model = RidgeCV(alphas=np.asarray(ALPHAS, dtype=float), gcv_mode="svd")
    model.fit(train_x, train_y)
    return model.predict(test_x), float(model.alpha_)


def pca_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    dimension: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    projector = PCA(
        n_components=dimension,
        whiten=True,
        svd_solver="randomized",
        random_state=seed,
    )
    train_projection = projector.fit_transform(train_x)
    test_projection = projector.transform(test_x)
    return ridge_predict(train_projection, train_y, test_projection)


def random_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    dimension: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    projector = GaussianRandomProjection(n_components=dimension, random_state=seed)
    train_projection = projector.fit_transform(train_x)
    test_projection = projector.transform(test_x)
    scaler = StandardScaler()
    train_projection = scaler.fit_transform(train_projection)
    test_projection = scaler.transform(test_projection)
    return ridge_predict(train_projection, train_y, test_projection)


def learned_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    dimension: int,
) -> np.ndarray:
    model = PLSRegression(n_components=dimension, scale=False, max_iter=1_000, tol=1e-8)
    model.fit(train_x, train_y)
    return model.predict(test_x)


def dimensionality_sweep(
    starting_embeddings: np.ndarray,
    targets: np.ndarray,
    dimensions: tuple[int, ...] = DIMENSIONS,
    random_repetitions: int = 12,
    seed: int = 20260818,
) -> dict:
    sample_count, embedding_dimension = starting_embeddings.shape
    if max(dimensions) >= sample_count - sample_count // 5:
        raise ValueError("Projection dimension must remain below the outer-fold training size.")
    starting = unit_rows(starting_embeddings)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(starting))
    fixed_predictions = np.empty_like(targets)
    full_predictions = np.empty_like(targets)
    pca_predictions = {dimension: np.empty_like(targets) for dimension in dimensions}
    learned_predictions = {dimension: np.empty_like(targets) for dimension in dimensions}
    random_predictions = {
        dimension: [np.empty_like(targets) for _ in range(random_repetitions)]
        for dimension in dimensions
    }
    selected_alphas = {"full": [], "pca": {str(k): [] for k in dimensions}, "random": {str(k): [] for k in dimensions}}

    for fold_index, (train, test) in enumerate(folds):
        train_x = starting[train]
        test_x = starting[test]
        train_y = targets[train]
        fixed_predictions[test] = train_y.mean(axis=0)
        full_predictions[test], alpha = ridge_predict(train_x, train_y, test_x)
        selected_alphas["full"].append(alpha)
        for dimension in dimensions:
            pca_predictions[dimension][test], alpha = pca_predict(
                train_x, train_y, test_x, dimension, seed + fold_index + dimension
            )
            selected_alphas["pca"][str(dimension)].append(alpha)
            learned_predictions[dimension][test] = learned_predict(
                train_x, train_y, test_x, dimension
            )
            selected_alphas["random"][str(dimension)].append([])
            for repetition in range(random_repetitions):
                prediction, alpha = random_predict(
                    train_x,
                    train_y,
                    test_x,
                    dimension,
                    seed + repetition * 10_000 + dimension,
                )
                random_predictions[dimension][repetition][test] = prediction
                selected_alphas["random"][str(dimension)][-1].append(alpha)

    fixed_metrics = prediction_metrics(targets, fixed_predictions, fixed_predictions, seed)
    full_metrics = prediction_metrics(targets, full_predictions, fixed_predictions, seed + 1)
    curves = {"pca": {}, "random": {}, "learned_pls": {}}
    raw_pvalues = {"full": {}, "pca": {}, "random": {}}
    full_cosines = np.asarray(full_metrics["direction_cosines"])
    for dimension in dimensions:
        pca_metrics = prediction_metrics(
            targets, pca_predictions[dimension], fixed_predictions, seed + 100 + dimension
        )
        curves["pca"][str(dimension)] = pca_metrics
        learned_metrics = prediction_metrics(
            targets, learned_predictions[dimension], fixed_predictions, seed + 200 + dimension
        )
        learned_cosines = np.asarray(learned_metrics["direction_cosines"])
        difference = learned_cosines - full_cosines
        learned_metrics["direction_difference_from_full"] = float(difference.mean())
        learned_metrics["direction_difference_from_full_95ci"] = bootstrap_mean_interval(
            difference, np.random.default_rng(seed + 300 + dimension)
        )
        learned_metrics["direction_vs_full_sign_flip_pvalue"] = paired_sign_flip_pvalue(
            difference,
            np.random.default_rng(seed + 400 + dimension),
            10_000,
        )
        curves["learned_pls"][str(dimension)] = learned_metrics
        random_metrics = [
            prediction_metrics(
                targets,
                predictions,
                fixed_predictions,
                seed + 1_000 + dimension * 100 + repetition,
            )
            for repetition, predictions in enumerate(random_predictions[dimension])
        ]
        direction_values = np.asarray([item["mean_direction_cosine"] for item in random_metrics])
        mse_values = np.asarray([item["mse_ratio_to_fixed"] for item in random_metrics])
        curves["random"][str(dimension)] = {
            "repetitions": random_repetitions,
            "mean_direction_cosine": float(direction_values.mean()),
            "projection_95interval_direction_cosine": [
                float(value) for value in np.quantile(direction_values, (0.025, 0.975))
            ],
            "mse_ratio_to_fixed": float(mse_values.mean()),
            "projection_95interval_mse_ratio": [
                float(value) for value in np.quantile(mse_values, (0.025, 0.975))
            ],
            "repetition_metrics": random_metrics,
        }
        learned_direction = np.asarray(learned_metrics["direction_cosines"])
        pca_direction = np.asarray(pca_metrics["direction_cosines"])
        random_direction = np.mean(
            [np.asarray(item["direction_cosines"]) for item in random_metrics],
            axis=0,
        )
        comparisons = {}
        for comparison, reference in (
            ("pca", pca_direction),
            ("random", random_direction),
        ):
            difference = learned_direction - reference
            pvalue = paired_sign_flip_pvalue(
                difference,
                np.random.default_rng(seed + 2_000 + dimension * 10 + len(comparisons)),
                10_000,
            )
            comparisons[comparison] = {
                "mean_direction_cosine_difference": float(difference.mean()),
                "difference_95ci": bootstrap_mean_interval(
                    difference,
                    np.random.default_rng(seed + 3_000 + dimension * 10 + len(comparisons)),
                ),
                "paired_sign_flip_pvalue": pvalue,
            }
            raw_pvalues[comparison][str(dimension)] = pvalue
        learned_metrics["comparisons"] = comparisons
        raw_pvalues["full"][str(dimension)] = learned_metrics["direction_vs_full_sign_flip_pvalue"]
    for comparison, pvalues in raw_pvalues.items():
        adjusted = holm_adjust_local(pvalues)
        for dimension in dimensions:
            if comparison == "full":
                curves["learned_pls"][str(dimension)]["direction_vs_full_holm_pvalue"] = adjusted[str(dimension)]
            else:
                curves["learned_pls"][str(dimension)]["comparisons"][comparison]["holm_pvalue"] = adjusted[str(dimension)]
    return {
        "sample_count": sample_count,
        "input_embedding_dimension": embedding_dimension,
        "tested_dimensions": list(dimensions),
        "random_projection_repetitions": random_repetitions,
        "fixed_vector": fixed_metrics,
        "full_state_ridge": full_metrics,
        "curves": curves,
        "selected_ridge_alphas": selected_alphas,
    }


def holm_adjust_local(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for rank, label in enumerate(ordered):
        running = max(running, pvalues[label] * (count - rank))
        adjusted[label] = float(min(1.0, running))
    return adjusted


def analyze(
    input_path: Path,
    seed: int = 20260818,
    random_repetitions: int = 12,
) -> dict:
    experiment = experiment_arrays(input_path)
    controls = {}
    for index, parameter in enumerate(sorted(set(experiment["parameters"].tolist()))):
        targets, delta = largest_displacements(experiment, parameter)
        controls[parameter] = {
            "delta": delta,
            **dimensionality_sweep(
                experiment["starting"],
                targets,
                DIMENSIONS,
                random_repetitions,
                seed + index * 1_000,
            ),
        }
    return {
        "experiment": "intervention-relevant-state-dimensionality-v5",
        "method": {
            "target": "largest positive displacement vector for each intervention",
            "outer_split": "five-fold held-out starting configurations",
            "pca": "training-fold PCA followed by ridge",
            "random": "repeated Gaussian projection followed by ridge",
            "learned": "training-fold supervised partial least squares",
            "full_reference": "ridge on the complete normalized starting embedding",
        },
        "controls": controls,
    }


def markdown_report(analysis: dict, metadata: dict) -> str:
    summary_rows = []
    for parameter, result in analysis["controls"].items():
        learned = result["curves"]["learned_pls"]
        best_dimension = max(
            result["tested_dimensions"],
            key=lambda dimension: learned[str(dimension)]["mean_direction_cosine"],
        )
        best = learned[str(best_dimension)]
        summary_rows.append(
            f"| `{parameter}` | {result['fixed_vector']['mean_direction_cosine']:.3f} | "
            f"{result['full_state_ridge']['mean_direction_cosine']:.3f} | {best_dimension} | "
            f"{best['mean_direction_cosine']:.3f} | {best['mse_ratio_to_fixed']:.3f} |"
        )
    tongue = analysis["controls"]["tongueIndex"]
    tongue_rows = []
    for dimension in tongue["tested_dimensions"]:
        key = str(dimension)
        tongue_rows.append(
            f"| {dimension} | {tongue['curves']['pca'][key]['mean_direction_cosine']:.3f} | "
            f"{tongue['curves']['random'][key]['mean_direction_cosine']:.3f} | "
            f"{tongue['curves']['learned_pls'][key]['mean_direction_cosine']:.3f} | "
            f"{tongue['curves']['learned_pls'][key]['mse_ratio_to_fixed']:.3f} |"
        )
    best_tongue_dimension = max(
        tongue["tested_dimensions"],
        key=lambda dimension: tongue["curves"]["learned_pls"][str(dimension)]["mean_direction_cosine"],
    )
    best_tongue = tongue["curves"]["learned_pls"][str(best_tongue_dimension)]
    return f"""# Experiment 5: Dimensionality of Intervention-Relevant HuBERT State

## Question

Can the starting-state information needed to predict a fixed intervention be compressed into a small subspace, or does performance continue improving as more HuBERT dimensions are exposed?

## Method

- Extractor: `{metadata.get('model_id', 'unknown')}` at `{metadata.get('resolved_revision', 'unknown')}`
- Starting configurations: {tongue['sample_count']}
- Input embedding dimension: {tongue['input_embedding_dimension']}
- Tested state dimensions: {', '.join(str(value) for value in tongue['tested_dimensions'])}
- Target: largest positive displacement vector for each intervention.
- Every projection and predictor is fit only inside each outer training fold.
- PCA and Gaussian random projections use ridge prediction; supervised projections use partial least squares (PLS).
- Random projections are repeated {tongue['random_projection_repetitions']} times per dimension.

Because there are only 50 starting configurations and 40 training configurations per outer fold, this experiment cannot identify more than 39 independent training-state directions. The sweep stops at 32 rather than making unsupported claims about 64–768 dimensions.

![Held-out displacement direction versus state dimension](./FIGURE_V5_DIRECTION.png)

![Held-out displacement error versus state dimension](./FIGURE_V5_MSE.png)

PCA and PLS bands bootstrap held-out starting configurations. Random-projection bands show variation over projection draws. The two uncertainty sources should not be interpreted as interchangeable confidence intervals.

## Best supervised projection by control

| Transformation | Fixed cosine | Full-state cosine | Best PLS k | Best PLS cosine | Best PLS MSE ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(summary_rows)}

## Tongue-position curve

| k | PCA cosine | Random cosine | Learned PLS cosine | Learned PLS MSE ratio |
| ---: | ---: | ---: | ---: | ---: |
{chr(10).join(tongue_rows)}

The full-state ridge reference reaches direction cosine **{tongue['full_state_ridge']['mean_direction_cosine']:.3f}** and MSE ratio **{tongue['full_state_ridge']['mse_ratio_to_fixed']:.3f}**. The strongest supervised tongue projection uses **{best_tongue_dimension}** dimensions, reaching cosine **{best_tongue['mean_direction_cosine']:.3f}** and MSE ratio **{best_tongue['mse_ratio_to_fixed']:.3f}**.

## Main finding

The tested high-dimensional explanation is not supported for tongue direction. A one-dimensional supervised PLS score reaches direction cosine **{tongue['curves']['learned_pls']['1']['mean_direction_cosine']:.3f}**, compared with **{tongue['curves']['pca']['1']['mean_direction_cosine']:.3f}** for one-dimensional PCA and **{tongue['curves']['random']['1']['mean_direction_cosine']:.3f}** for one-dimensional random projection. The supervised advantage is significant after correction across the six tested dimensions (Holm p **{tongue['curves']['learned_pls']['1']['comparisons']['pca']['holm_pvalue']:.4g}** versus PCA and **{tongue['curves']['learned_pls']['1']['comparisons']['random']['holm_pvalue']:.4g}** versus random).

The one-dimensional supervised score is statistically compatible with the full-state ridge reference: their direction-cosine difference is **{tongue['curves']['learned_pls']['1']['direction_difference_from_full']:.3f}**, with bootstrap 95% interval **{tongue['curves']['learned_pls']['1']['direction_difference_from_full_95ci'][0]:.3f}–{tongue['curves']['learned_pls']['1']['direction_difference_from_full_95ci'][1]:.3f}**. Exposing more supervised dimensions does not produce a rising tongue curve and progressively worsens vector MSE.

This suggests that the predictable *directional* part of the tongue response can be compressed into one intervention-specific linear score hidden across the original 768 coordinates. It does not solve the prediction problem: direction cosine remains low and vector MSE stays close to the fixed baseline even at `k=1`. Other interventions show broader dimensional requirements, with their descriptive PLS optima between 8 and 32 dimensions.

## Interpretation boundary

PLS is a linear supervised projection. Its one-dimensional score is a learned weighted combination of all 768 coordinates, not one original HuBERT coordinate. Success at small `k` identifies a compact predictive subspace for this synthetic intervention, not a uniquely articulatory or causal variable. The reported best `k` values are descriptive choices from the held-out curves rather than independently nested model selections. The 50-state sample size limits the tested effective dimensionality to 32.
"""


def plot_results(analysis: dict, direction_path: Path, mse_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    methods = (
        ("pca", "PCA", "#4f7c8d"),
        ("random", "Random projection", "#94a3b8"),
        ("learned_pls", "Learned PLS", "#a52562"),
    )

    def make_figure(metric: str, output: Path, title: str, ylabel: str) -> None:
        figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
        for axis, (parameter, result) in zip(axes.ravel(), analysis["controls"].items()):
            dimensions = np.asarray(result["tested_dimensions"])
            positions = np.arange(dimensions.size)
            for method, label, color in methods:
                values = []
                lower = []
                upper = []
                for dimension in dimensions:
                    item = result["curves"][method][str(dimension)]
                    values.append(item[metric])
                    interval_key = (
                        "projection_95interval_direction_cosine"
                        if method == "random" and metric == "mean_direction_cosine"
                        else "projection_95interval_mse_ratio"
                        if method == "random"
                        else "direction_cosine_95ci"
                        if metric == "mean_direction_cosine"
                        else "mse_ratio_to_fixed_95ci"
                    )
                    interval = item[interval_key]
                    lower.append(interval[0])
                    upper.append(interval[1])
                values = np.asarray(values)
                axis.plot(positions, values, marker="o", linewidth=2, label=label, color=color)
                axis.fill_between(positions, lower, upper, color=color, alpha=0.12, linewidth=0)
            reference_key = "mean_direction_cosine" if metric == "mean_direction_cosine" else "mse_ratio_to_fixed"
            axis.axhline(result["fixed_vector"][reference_key], color="#68717d", linestyle=":", linewidth=1.2, label="Fixed")
            axis.axhline(result["full_state_ridge"][reference_key], color="#303846", linestyle="--", linewidth=1.2, label="Full state")
            axis.set_xticks(positions, dimensions)
            axis.set_xlabel("Starting-state dimensions k")
            axis.set_ylabel(ylabel)
            axis.set_title(DISPLAY_NAMES.get(parameter, parameter), loc="left", fontweight="bold")
        axes.ravel()[-1].axis("off")
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.92, 0.12), frameon=False)
        figure.suptitle(title, fontsize=17, fontweight="bold", y=0.985)
        figure.tight_layout(rect=(0, 0.04, 1, 0.955))
        figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(figure)

    make_figure(
        "mean_direction_cosine",
        direction_path,
        "How many HuBERT state dimensions predict intervention direction?",
        "Held-out direction cosine",
    )
    make_figure(
        "mse_ratio_to_fixed",
        mse_path,
        "How many HuBERT state dimensions reduce displacement error?",
        "MSE ratio to fixed vector (lower is better)",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--direction-figure", type=Path)
    parser.add_argument("--mse-figure", type=Path)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--random-repetitions", type=int, default=12)
    args = parser.parse_args()
    if args.random_repetitions < 2:
        parser.error("--random-repetitions must be at least 2")
    return args


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_json = (args.output_json or input_path.with_name("analysis_v5.json")).resolve()
    report_path = (args.report or input_path.with_name("REPORT_V5.md")).resolve()
    direction_path = (args.direction_figure or input_path.with_name("FIGURE_V5_DIRECTION.png")).resolve()
    mse_path = (args.mse_figure or input_path.with_name("FIGURE_V5_MSE.png")).resolve()
    metadata_path = input_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf8"))
    result = analyze(input_path, args.seed, args.random_repetitions)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    report_path.write_text(markdown_report(result, metadata), encoding="utf8")
    plot_results(result, direction_path, mse_path)
    print(f"Analysis: {output_json}")
    print(f"Report: {report_path}")
    print(f"Figures: {direction_path}, {mse_path}")


if __name__ == "__main__":
    main()
