"""Analyze Experiment 2 displacement directions and magnitudes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT = Path(__file__).with_name("experiment_v2.npz")

DISPLAY_NAMES = {
    "constrictionDiameter": "Constriction diameter",
    "constrictionIndex": "Constriction location",
    "pitchHz": "Pitch",
    "tongueDiameter": "Tongue diameter",
    "tongueIndex": "Tongue position",
}

DELTA_UNITS = {
    "constrictionDiameter": "tract units",
    "constrictionIndex": "tract sections",
    "pitchHz": "Hz",
    "tongueDiameter": "tract units",
    "tongueIndex": "tract sections",
}


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)


def confidence_interval(values: np.ndarray, rng: np.random.Generator, draws: int = 5_000) -> list[float]:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return [float(values.mean()), float(values.mean())]
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def sign_flip_pvalue(vectors: np.ndarray, rng: np.random.Generator, draws: int = 5_000) -> float:
    observed = float(np.linalg.norm(vectors.mean(axis=0)))
    extreme = 0
    remaining = draws
    while remaining:
        count = min(250, remaining)
        signs = rng.choice((-1.0, 1.0), size=(count, vectors.shape[0]))
        null_norms = np.linalg.norm(signs @ vectors / vectors.shape[0], axis=1)
        extreme += int(np.sum(null_norms >= observed))
        remaining -= count
    return float((extreme + 1) / (draws + 1))


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for rank, label in enumerate(ordered):
        running = max(running, pvalues[label] * (count - rank))
        adjusted[label] = float(min(1.0, running))
    return adjusted


def direction_metrics(vectors: np.ndarray, seed: int = 20260817) -> dict:
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] < 2:
        raise ValueError("Direction analysis requires at least two displacement vectors.")
    rng = np.random.default_rng(seed)
    mean_vector = vectors.mean(axis=0)
    individual_norms = np.linalg.norm(vectors, axis=1)
    alignments = row_cosine(vectors, np.broadcast_to(mean_vector, vectors.shape))
    leave_one_out_means = (vectors.sum(axis=0) - vectors) / (vectors.shape[0] - 1)
    leave_one_out = row_cosine(vectors, leave_one_out_means)
    normalized = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    pairwise_matrix = normalized @ normalized.T
    pairwise = pairwise_matrix[np.triu_indices(vectors.shape[0], k=1)]
    mean_norm = float(np.linalg.norm(mean_vector))
    mean_individual_norm = float(np.mean(individual_norms))
    return {
        "sample_count": int(vectors.shape[0]),
        "embedding_dimension": int(vectors.shape[1]),
        "nonzero_displacement_fraction": float(np.mean(individual_norms > 1e-8)),
        "mean_pairwise_cosine": float(pairwise.mean()),
        "median_pairwise_cosine": float(np.median(pairwise)),
        "pairwise_cosine_std": float(pairwise.std()),
        "pairwise_cosine_95ci": confidence_interval(pairwise, rng),
        "mean_alignment_to_mean": float(alignments.mean()),
        "alignment_to_mean_std": float(alignments.std()),
        "alignment_to_mean_95ci": confidence_interval(alignments, rng),
        "mean_leave_one_out_alignment": float(leave_one_out.mean()),
        "leave_one_out_alignment_std": float(leave_one_out.std()),
        "leave_one_out_alignment_95ci": confidence_interval(leave_one_out, rng),
        "resultant_strength": mean_norm / max(mean_individual_norm, 1e-12),
        "mean_displacement_norm": mean_individual_norm,
        "sign_flip_pvalue": sign_flip_pvalue(vectors, rng),
        "mean_vector": mean_vector.tolist(),
    }


def regression_r2(x: np.ndarray, y: np.ndarray, through_origin: bool = False) -> tuple[float, float, float]:
    if through_origin:
        slope = float(np.dot(x, y) / max(np.dot(x, x), 1e-12))
        intercept = 0.0
    else:
        slope, intercept = (float(value) for value in np.polyfit(x, y, 1))
    predicted = slope * x + intercept
    denominator = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(np.sum((y - predicted) ** 2)) / max(denominator, 1e-12)
    return slope, intercept, r2


def linearity_metrics(displacements: np.ndarray, deltas: np.ndarray) -> dict:
    output = {}
    norms = np.linalg.norm(displacements, axis=-1)
    for side, keep in (("positive", deltas > 0), ("negative", deltas < 0)):
        if not np.any(keep):
            continue
        magnitudes = np.abs(deltas[keep])
        selected = norms[:, keep]
        order = np.argsort(magnitudes)
        magnitudes = magnitudes[order]
        selected = selected[:, order]
        mean_norms = selected.mean(axis=0)
        slope, intercept, r2 = regression_r2(magnitudes, mean_norms)
        origin_slope, _, origin_r2 = regression_r2(magnitudes, mean_norms, through_origin=True)
        per_sample_r2 = [regression_r2(magnitudes, row)[2] for row in selected]
        largest_vectors = displacements[:, np.flatnonzero(keep)[order[-1]], :]
        direction_to_largest = []
        for original_index in np.flatnonzero(keep)[order]:
            direction_to_largest.append(float(np.mean(row_cosine(
                displacements[:, original_index, :],
                largest_vectors,
            ))))
        output[side] = {
            "magnitudes": magnitudes.tolist(),
            "mean_displacement_norms": mean_norms.tolist(),
            "std_displacement_norms": selected.std(axis=0).tolist(),
            "linear_slope": slope,
            "linear_intercept": intercept,
            "linear_r2": r2,
            "origin_slope": origin_slope,
            "origin_r2": origin_r2,
            "median_per_sample_r2": float(np.median(per_sample_r2)),
            "mean_cosine_to_largest_delta": direction_to_largest,
        }
    return output


def grouped_predictions(estimator, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("Held-out evaluation requires at least two base configurations.")
    folds = GroupKFold(n_splits=min(5, unique_groups.size))
    return cross_val_predict(estimator, x, y, groups=groups, cv=folds)


def nested_grouped_ridge_predictions(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    alphas = (0.1, 1.0, 10.0, 100.0, 1_000.0)
    unique_groups = np.unique(groups)
    outer = GroupKFold(n_splits=min(5, unique_groups.size))
    predicted = np.empty_like(y, dtype=float)
    chosen_alphas = []
    for train, test in outer.split(x, y, groups):
        train_groups = groups[train]
        inner = GroupKFold(n_splits=min(4, np.unique(train_groups).size))
        best_alpha = None
        best_score = -np.inf
        for alpha in alphas:
            estimator = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))
            inner_prediction = cross_val_predict(
                estimator,
                x[train],
                y[train],
                groups=train_groups,
                cv=inner,
            )
            score = r2_score(y[train], inner_prediction)
            if score > best_score:
                best_score = score
                best_alpha = alpha
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha, solver="lsqr"))
        estimator.fit(x[train], y[train])
        predicted[test] = estimator.predict(x[test])
        chosen_alphas.append(float(best_alpha))
    return predicted, chosen_alphas


def classify_transformations(
    displacements: np.ndarray,
    variant_parameters: np.ndarray,
    variant_deltas: np.ndarray,
) -> dict:
    positive = variant_deltas > 0
    x = displacements[:, positive, :].reshape(-1, displacements.shape[-1])
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    labels_per_sample = variant_parameters[positive]
    y = np.tile(labels_per_sample, displacements.shape[0])
    groups = np.repeat(np.arange(displacements.shape[0]), labels_per_sample.size)
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3_000, class_weight="balanced", C=0.5),
    )
    predicted = grouped_predictions(estimator, x, y, groups)
    labels = sorted(np.unique(y).tolist())
    group_accuracies = np.asarray([
        np.mean(predicted[groups == group] == y[groups == group])
        for group in np.unique(groups)
    ])
    matrix = confusion_matrix(y, predicted, labels=labels)
    return {
        "features": "unit-normalized positive displacement vectors",
        "split": "grouped by held-out base configuration",
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro")),
        "base_group_accuracy_95ci": confidence_interval(
            group_accuracies,
            np.random.default_rng(20260817),
        ),
        "labels": labels,
        "per_class_recall": {
            label: float(matrix[index, index] / max(matrix[index].sum(), 1))
            for index, label in enumerate(labels)
        },
        "confusion_matrix": matrix.tolist(),
    }


def regress_magnitudes(
    displacements: np.ndarray,
    variant_parameters: np.ndarray,
    variant_deltas: np.ndarray,
) -> dict:
    output = {}
    for parameter in sorted(set(variant_parameters.tolist())):
        keep = variant_parameters == parameter
        x = displacements[:, keep, :].reshape(-1, displacements.shape[-1])
        deltas = variant_deltas[keep]
        y = np.tile(deltas, displacements.shape[0])
        groups = np.repeat(np.arange(displacements.shape[0]), deltas.size)
        predicted, chosen_alphas = nested_grouped_ridge_predictions(x, y, groups)
        output[parameter] = {
            "r2": float(r2_score(y, predicted)),
            "mae": float(mean_absolute_error(y, predicted)),
            "sign_accuracy": float(np.mean(np.sign(y) == np.sign(predicted))),
            "outer_fold_alphas": chosen_alphas,
            "target_min": float(y.min()),
            "target_max": float(y.max()),
        }
    return output


def analyze(input_path: Path, seed: int = 20260817) -> dict:
    checkpoint = np.load(input_path, allow_pickle=False)
    completed = checkpoint["completed"].astype(bool)
    if not np.all(completed):
        raise ValueError(f"Experiment is incomplete: {int(completed.sum())}/{completed.size} samples saved.")
    embeddings = checkpoint["embeddings"].astype(np.float64)
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("Embeddings contain non-finite values.")
    variant_parameters = checkpoint["variant_parameters"].astype(str)
    variant_deltas = checkpoint["variant_deltas"].astype(float)
    if variant_parameters[0] != "base":
        raise ValueError("The first embedding variant must be the base configuration.")
    displacement_parameters = variant_parameters[1:]
    displacement_deltas = variant_deltas[1:]
    displacements = embeddings[:, 1:, :] - embeddings[:, :1, :]
    direction = {}
    linearity = {}
    max_positive_means = {}
    for parameter in sorted(set(displacement_parameters.tolist())):
        keep = displacement_parameters == parameter
        parameter_deltas = displacement_deltas[keep]
        parameter_vectors = displacements[:, keep, :]
        direction[parameter] = {}
        for index, delta in enumerate(parameter_deltas):
            direction[parameter][str(float(delta))] = direction_metrics(
                parameter_vectors[:, index, :],
                seed=seed + index,
            )
        linearity[parameter] = linearity_metrics(parameter_vectors, parameter_deltas)
        max_index = int(np.argmax(parameter_deltas))
        max_positive_means[parameter] = parameter_vectors[:, max_index, :].mean(axis=0)

    direction_labels = sorted(max_positive_means)
    direction_matrix = np.asarray([max_positive_means[label] for label in direction_labels])
    mean_direction_cosines = row_cosine(
        direction_matrix[:, None, :],
        direction_matrix[None, :, :],
    )
    max_positive_pvalues = {}
    for parameter in direction_labels:
        positive = [float(value) for value in direction[parameter] if float(value) > 0]
        max_positive_pvalues[parameter] = direction[parameter][str(max(positive))]["sign_flip_pvalue"]
    return {
        "sample_count": int(embeddings.shape[0]),
        "embedding_dimension": int(embeddings.shape[-1]),
        "direction": direction,
        "linearity": linearity,
        "mean_direction_labels": direction_labels,
        "mean_direction_cosine_matrix": mean_direction_cosines.tolist(),
        "max_positive_direction_holm_pvalues": holm_adjust(max_positive_pvalues),
        "transformation_classifier": classify_transformations(
            displacements,
            displacement_parameters,
            displacement_deltas,
        ),
        "magnitude_regression": regress_magnitudes(
            displacements,
            displacement_parameters,
            displacement_deltas,
        ),
    }


def markdown_report(analysis: dict, metadata: dict) -> str:
    extractor = metadata.get("extractor", "unknown")
    warning = (
        "> **Smoke-test result only:** this run used the mock spectral extractor, not HuBERT.\n\n"
        if extractor == "mock"
        else ""
    )
    direction_rows = []
    for parameter, by_delta in analysis["direction"].items():
        positive = [float(value) for value in by_delta if float(value) > 0]
        delta = max(positive)
        metrics = by_delta[str(delta)]
        adjusted_p = analysis["max_positive_direction_holm_pvalues"][parameter]
        direction_rows.append(
            f"| `{parameter}` | {delta:g} | {metrics['mean_pairwise_cosine']:.4f} | "
            f"{metrics['mean_leave_one_out_alignment']:.4f} | {metrics['resultant_strength']:.4f} | "
            f"{metrics['nonzero_displacement_fraction']:.3f} | "
            f"{metrics['sign_flip_pvalue']:.4g} | {adjusted_p:.4g} |"
        )
    linearity_rows = []
    for parameter, metrics in analysis["linearity"].items():
        positive = metrics.get("positive", {})
        linearity_rows.append(
            f"| `{parameter}` | {positive.get('linear_r2', float('nan')):.4f} | "
            f"{positive.get('origin_r2', float('nan')):.4f} | "
            f"{positive.get('median_per_sample_r2', float('nan')):.4f} | "
            f"{positive.get('mean_cosine_to_largest_delta', [float('nan')])[0]:.4f} |"
        )
    regression_rows = [
        f"| `{parameter}` | {metrics['r2']:.4f} | {metrics['mae']:.4f} | {metrics['sign_accuracy']:.3f} |"
        for parameter, metrics in analysis["magnitude_regression"].items()
    ]
    classifier = analysis["transformation_classifier"]
    tongue_direction = analysis["direction"]["tongueIndex"]["0.5"]
    tongue_linearity = analysis["linearity"]["tongueIndex"]["positive"]
    tongue_regression = analysis["magnitude_regression"]["tongueIndex"]
    tongue_recall = classifier["per_class_recall"]["tongueIndex"]
    return f"""# Experiment 2: Directions of Physical Displacement in HuBERT Space

{warning}## Method

- Extractor: `{metadata.get('model_id', extractor)}` at `{metadata.get('resolved_revision', 'unknown')}`
- Base configurations: {analysis['sample_count']}
- Embedding dimension: {analysis['embedding_dimension']}
- Matched synthesis noise within each base/intervention set: {metadata.get('matched_noise_within_configuration')}
- Evaluation splits keep every intervention from a base configuration in the same fold.

![Mean HuBERT displacement norm across each physical sweep](./FIGURE_V2.png)

Lines show the mean embedding displacement norm over base configurations; shaded regions are 95% confidence intervals for the mean. Panel scales differ because the controls use different physical units.

## Direction consistency at the largest positive intervention

| Transformation | Delta | Pairwise cosine | Leave-one-out alignment | Resultant strength | Nonzero | Raw p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(direction_rows)}

Leave-one-out alignment compares each displacement with the mean of all *other* displacements, avoiding the optimistic self-inclusion of the ordinary mean vector.

## Positive-delta magnitude linearity

| Transformation | Mean-curve R² | Through-origin R² | Median within-base R² | Small→largest cosine |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(linearity_rows)}

## Held-out transformation classification

- Direction-only balanced accuracy: **{classifier['balanced_accuracy']:.3f}**
- Base-group bootstrap 95% CI: **{classifier['base_group_accuracy_95ci'][0]:.3f}–{classifier['base_group_accuracy_95ci'][1]:.3f}**
- Macro F1: **{classifier['macro_f1']:.3f}**
- Chance balanced accuracy: **{1 / len(classifier['labels']):.3f}**

## Held-out signed-magnitude regression

| Transformation | R² | MAE | Direction accuracy |
| --- | ---: | ---: | ---: |
{chr(10).join(regression_rows)}

## Main finding

Tongue-position change is highly linear *within* a starting tract: its positive mean displacement curve has R² **{tongue_linearity['linear_r2']:.4f}**, the median within-base R² is **{tongue_linearity['median_per_sample_r2']:.4f}**, and the smallest positive displacement aligns with the largest at cosine **{tongue_linearity['mean_cosine_to_largest_delta'][0]:.4f}**. This is strong evidence for a stable local trajectory.

The trajectory is not a strong global tongue-forward direction across starting tracts. At `+0.5`, mean pairwise cosine is only **{tongue_direction['mean_pairwise_cosine']:.4f}**, leave-one-out alignment is **{tongue_direction['mean_leave_one_out_alignment']:.4f}**, and the Holm-adjusted sign-flip p-value is **{analysis['max_positive_direction_holm_pvalues']['tongueIndex']:.4f}**. On unseen starting tracts, signed tongue magnitude has R² **{tongue_regression['r2']:.4f}**, and transformation-class recall for tongue position is **{tongue_recall:.3f}**.

Across all five controls, direction-only classification reaches balanced accuracy **{classifier['balanced_accuracy']:.3f}** versus **{1 / len(classifier['labels']):.3f}** chance. HuBERT therefore preserves useful information about the kind of physical transformation, but that information is context-dependent rather than one universal vector per control.

## Interpretation boundary

Consistent displacement directions support a stable transformation in this representation for this synthesizer and intervention range. They do not by themselves establish articulatory causality, phoneme identity, or generalization to natural speech.
"""


def plot_displacement_curves(analysis: dict, output_path: Path) -> None:
    """Plot signed intervention sweeps with uncertainty across base configurations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    axes = axes.ravel()
    sample_count = max(int(analysis["sample_count"]), 1)
    color = "#a52562"

    for axis, parameter in zip(axes, sorted(analysis["linearity"])):
        points = [(0.0, 0.0, 0.0)]
        by_side = analysis["linearity"][parameter]
        for side in ("negative", "positive"):
            metrics = by_side.get(side)
            if not metrics:
                continue
            sign = -1.0 if side == "negative" else 1.0
            for magnitude, mean, std in zip(
                metrics["magnitudes"],
                metrics["mean_displacement_norms"],
                metrics["std_displacement_norms"],
            ):
                ci = 1.96 * float(std) / np.sqrt(sample_count)
                points.append((sign * float(magnitude), float(mean), ci))
        points.sort(key=lambda item: item[0])
        x, mean, ci = (np.asarray(values) for values in zip(*points))
        axis.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.16, linewidth=0)
        axis.plot(x, mean, color=color, marker="o", markersize=4.5, linewidth=2)
        axis.axvline(0, color="#68717d", linewidth=0.9, alpha=0.6)
        axis.set_title(DISPLAY_NAMES.get(parameter, parameter), loc="left", fontweight="bold")
        axis.set_xlabel(f"Signed physical change ({DELTA_UNITS.get(parameter, 'units')})")
        axis.set_ylabel("Mean embedding displacement ‖Δz‖")
        positive = by_side.get("positive", {})
        if positive:
            axis.text(
                0.98,
                0.04,
                f"positive sweep R² = {positive['linear_r2']:.3f}",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color="#4b5563",
            )
        axis.set_ylim(bottom=0)

    axes[-1].axis("off")
    figure.suptitle(
        "HuBERT embedding displacement across controlled physical sweeps",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.018,
        "50 held-out starting vocal-tract configurations · bands show 95% confidence intervals for the mean",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.955))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_json = (args.output_json or input_path.with_name("analysis_v2.json")).resolve()
    report_path = (args.report or input_path.with_name("REPORT_V2.md")).resolve()
    figure_path = (args.figure or input_path.with_name("FIGURE_V2.png")).resolve()
    metadata_path = input_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf8")) if metadata_path.exists() else {}
    result = analyze(input_path, args.seed)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    report_path.write_text(markdown_report(result, metadata), encoding="utf8")
    plot_displacement_curves(result, figure_path)
    print(f"Analysis: {output_json}")
    print(f"Report: {report_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
