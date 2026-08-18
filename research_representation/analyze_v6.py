"""Experiment 6: interpret the one-dimensional tongue-response PLS state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import linalg, signal
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

try:
    from . import engine
    from .analyze_v3 import unit_rows
    from .analyze_v4 import ALPHAS, experiment_arrays, largest_displacements
    from .analyze_v5 import holm_adjust_local
    from .experiment import Variant, render_variants
    from .experiment_v4 import base_configurations, file_sha256, log_mel_feature
except ImportError:  # Direct script execution.
    import engine
    from analyze_v3 import unit_rows
    from analyze_v4 import ALPHAS, experiment_arrays, largest_displacements
    from analyze_v5 import holm_adjust_local
    from experiment import Variant, render_variants
    from experiment_v4 import base_configurations, file_sha256, log_mel_feature


DEFAULT_EMBEDDINGS = Path(__file__).with_name("experiment_v2.npz")
DEFAULT_ACOUSTICS = Path(__file__).with_name("acoustic_v4.npz")
DEFAULT_INTERPRETABLE_CACHE = Path(__file__).with_name("acoustic_v6.npz")
PHYSICAL_LABELS = {
    "tongueIndex": "Initial tongue position",
    "tongueDiameter": "Tongue diameter",
    "constrictionIndex": "Constriction location",
    "constrictionDiameter": "Constriction diameter",
    "pitchHz": "Pitch",
}
ACOUSTIC_LABELS = {
    "formantF1Hz": "LPC F1",
    "formantF2Hz": "LPC F2",
    "spectralCentroidHz": "Spectral centroid",
    "spectralBandwidthHz": "Spectral bandwidth",
    "spectralFlatness": "Spectral flatness",
}


def fit_pls(starting: np.ndarray, targets: np.ndarray) -> PLSRegression:
    model = PLSRegression(n_components=1, scale=False, max_iter=1_000, tol=1e-8)
    model.fit(starting, targets)
    return model


def standardized_score(model: PLSRegression, values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference_scores = model.transform(reference)[:, 0]
    scale = max(float(reference_scores.std(ddof=1)), 1e-12)
    return (model.transform(values)[:, 0] - reference_scores.mean()) / scale


def cross_fitted_pls_score(
    starting_embeddings: np.ndarray,
    targets: np.ndarray,
    seed: int,
) -> dict:
    starting = unit_rows(starting_embeddings)
    full_model = fit_pls(starting, targets)
    full_rotation = full_model.x_rotations_[:, 0]
    full_rotation /= max(np.linalg.norm(full_rotation), 1e-12)
    full_score = standardized_score(full_model, starting, starting)

    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    cross_fitted = np.empty(starting.shape[0], dtype=float)
    fold_stability = []
    fold_sizes = []
    for train, test in folds.split(starting):
        model = fit_pls(starting[train], targets[train])
        rotation = model.x_rotations_[:, 0]
        rotation /= max(np.linalg.norm(rotation), 1e-12)
        alignment = float(np.dot(rotation, full_rotation))
        orientation = 1.0 if alignment >= 0 else -1.0
        cross_fitted[test] = orientation * standardized_score(
            model, starting[test], starting[train]
        )
        fold_stability.append(abs(alignment))
        fold_sizes.append({"train": int(train.size), "test": int(test.size)})

    correlation = spearmanr(cross_fitted, full_score)
    return {
        "score": cross_fitted,
        "full_sample_score": full_score,
        "fold_axis_absolute_cosines_to_full": fold_stability,
        "mean_fold_axis_absolute_cosine_to_full": float(np.mean(fold_stability)),
        "minimum_fold_axis_absolute_cosine_to_full": float(np.min(fold_stability)),
        "cross_fitted_vs_full_score_spearman": float(correlation.statistic),
        "fold_sizes": fold_sizes,
        "orientation_note": "Fold axes are sign-aligned to the full-data axis; reported associations are unchanged in magnitude by a global sign flip.",
    }


def frame_lpc_formants(frame: np.ndarray, sample_rate: int, order: int = 18) -> tuple[float, float] | None:
    frame = np.asarray(frame, dtype=float)
    frame = frame - frame.mean()
    frame *= np.hamming(frame.size)
    autocorrelation = signal.correlate(frame, frame, mode="full", method="fft")
    center = frame.size - 1
    values = autocorrelation[center:center + order + 1]
    if values[0] <= 1e-12:
        return None
    values[0] *= 1.0001
    try:
        coefficients = linalg.solve_toeplitz(values[:-1], -values[1:])
    except linalg.LinAlgError:
        return None
    roots = np.roots(np.r_[1.0, coefficients])
    roots = roots[np.imag(roots) > 0]
    frequencies = np.angle(roots) * sample_rate / (2 * np.pi)
    bandwidths = -np.log(np.maximum(np.abs(roots), 1e-12)) * sample_rate / np.pi
    keep = (
        (frequencies >= 90)
        & (frequencies <= 5_000)
        & (bandwidths > 0)
        & (bandwidths < 1_000)
    )
    valid = np.sort(frequencies[keep])
    if valid.size < 2:
        return None
    return float(valid[0]), float(valid[1])


def acoustic_measurements(audio: np.ndarray) -> dict[str, float]:
    audio = np.asarray(audio, dtype=float)
    downsampled = signal.resample_poly(audio, 160, 441)
    emphasized = np.r_[downsampled[0], downsampled[1:] - 0.97 * downsampled[:-1]]
    frame_size = round(0.025 * 16_000)
    hop_size = round(0.010 * 16_000)
    formants = []
    for start in range(0, emphasized.size - frame_size + 1, hop_size):
        estimate = frame_lpc_formants(emphasized[start:start + frame_size], 16_000)
        if estimate is not None:
            formants.append(estimate)
    if not formants:
        first_formant = second_formant = float("nan")
    else:
        first_formant, second_formant = np.median(np.asarray(formants), axis=0)

    frequencies, _, spectrum = signal.stft(
        audio,
        fs=engine.SAMPLE_RATE,
        window="hann",
        nperseg=round(0.025 * engine.SAMPLE_RATE),
        noverlap=round(0.015 * engine.SAMPLE_RATE),
        nfft=2_048,
        boundary=None,
        padded=False,
    )
    power = np.mean(np.abs(spectrum) ** 2, axis=1)
    keep = (frequencies >= 80) & (frequencies <= 8_000)
    frequencies = frequencies[keep]
    power = power[keep]
    total = max(float(power.sum()), 1e-12)
    centroid = float(np.sum(frequencies * power) / total)
    bandwidth = float(np.sqrt(np.sum(((frequencies - centroid) ** 2) * power) / total))
    flatness = float(np.exp(np.mean(np.log(power + 1e-12))) / max(np.mean(power), 1e-12))
    return {
        "formantF1Hz": float(first_formant),
        "formantF2Hz": float(second_formant),
        "spectralCentroidHz": centroid,
        "spectralBandwidthHz": bandwidth,
        "spectralFlatness": flatness,
    }


def load_or_render_interpretable_acoustics(
    embedding_path: Path,
    acoustic_path: Path,
    cache_path: Path,
    acoustic_metadata: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    embedding_digest = file_sha256(embedding_path)
    acoustic_digest = file_sha256(acoustic_path)
    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=False)
        if str(cache["embedding_sha256"]) != embedding_digest:
            raise ValueError("Experiment 6 acoustic cache does not match the embedding checkpoint.")
        if str(cache["acoustic_sha256"]) != acoustic_digest:
            raise ValueError("Experiment 6 acoustic cache does not match the log-mel checkpoint.")
        names = cache["feature_names"].astype(str).tolist()
        if names != list(ACOUSTIC_LABELS):
            raise ValueError("Experiment 6 acoustic cache has an unexpected feature order.")
        values = cache["feature_values"].astype(float)
        return (
            {name: values[:, index] for index, name in enumerate(names)},
            json.loads(str(cache["validation_json"])),
        )

    source = np.load(embedding_path, allow_pickle=False)
    acoustic_cache = np.load(acoustic_path, allow_pickle=False)
    configurations = base_configurations(source)
    seed = int(acoustic_metadata["seed"])
    duration = float(acoustic_metadata["duration_seconds"])
    warmup = float(acoustic_metadata["warmup_seconds"])
    output = {name: np.empty(len(configurations), dtype=float) for name in ACOUSTIC_LABELS}
    maximum_log_mel_difference = 0.0
    for sample_index, configuration in enumerate(configurations):
        audio = render_variants(
            configuration,
            (Variant("base", 0.0),),
            duration,
            warmup,
            seed * 1_000_003 + sample_index,
            1,
        )[0]
        rerendered = log_mel_feature(audio).astype(float)
        cached = acoustic_cache["base_features"][sample_index].astype(float)
        maximum_log_mel_difference = max(
            maximum_log_mel_difference,
            float(np.max(np.abs(rerendered - cached))),
        )
        for name, value in acoustic_measurements(audio).items():
            output[name][sample_index] = value
    validation = {
        "sample_count": len(configurations),
        "maximum_absolute_log_mel_rerender_difference": maximum_log_mel_difference,
        "formant_method": "median framewise LPC after 16 kHz resampling and pre-emphasis",
        "spectral_method": "mean 25 ms STFT power from 80 to 8000 Hz",
        "missing_values": {
            name: int(np.count_nonzero(~np.isfinite(values))) for name, values in output.items()
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        feature_names=np.asarray(list(output)),
        feature_values=np.column_stack(list(output.values())),
        embedding_sha256=np.asarray(embedding_digest),
        acoustic_sha256=np.asarray(acoustic_digest),
        validation_json=np.asarray(json.dumps(validation, sort_keys=True)),
    )
    temporary.replace(cache_path)
    return output, validation


def bootstrap_spearman_interval(
    values: np.ndarray,
    scores: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> list[float]:
    estimates = []
    for _ in range(draws):
        indices = rng.integers(0, values.size, size=values.size)
        estimate = spearmanr(values[indices], scores[indices]).statistic
        if np.isfinite(estimate):
            estimates.append(estimate)
    return [float(value) for value in np.quantile(estimates, (0.025, 0.975))]


def spearman_permutation_pvalue(
    values: np.ndarray,
    scores: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> float:
    observed = abs(float(spearmanr(values, scores).statistic))
    extreme = 0
    for _ in range(draws):
        permuted = rng.permutation(scores)
        extreme += abs(float(spearmanr(values, permuted).statistic)) >= observed
    return float((extreme + 1) / (draws + 1))


def univariate_associations(
    feature_groups: dict[str, dict[str, np.ndarray]],
    scores: np.ndarray,
    seed: int,
    permutation_draws: int,
    bootstrap_draws: int,
) -> dict:
    results = {}
    raw_pvalues = {}
    index = 0
    for group, features in feature_groups.items():
        for name, raw_values in features.items():
            values = np.asarray(raw_values, dtype=float)
            keep = np.isfinite(values) & np.isfinite(scores)
            values = values[keep]
            valid_scores = scores[keep]
            spearman = spearmanr(values, valid_scores)
            pearson = pearsonr(values, valid_scores)
            pvalue = spearman_permutation_pvalue(
                values,
                valid_scores,
                np.random.default_rng(seed + index * 100),
                permutation_draws,
            )
            results[name] = {
                "label": (PHYSICAL_LABELS | ACOUSTIC_LABELS)[name],
                "group": group,
                "valid_samples": int(values.size),
                "spearman_rho": float(spearman.statistic),
                "spearman_95ci": bootstrap_spearman_interval(
                    values,
                    valid_scores,
                    np.random.default_rng(seed + index * 100 + 1),
                    bootstrap_draws,
                ),
                "spearman_permutation_pvalue": pvalue,
                "pearson_r": float(pearson.statistic),
                "univariate_linear_r2": float(pearson.statistic ** 2),
            }
            raw_pvalues[name] = pvalue
            index += 1
    adjusted = holm_adjust_local(raw_pvalues)
    for name, value in adjusted.items():
        results[name]["holm_pvalue"] = value
    return results


def impute_training_medians(
    train_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    medians = np.nanmedian(train_x, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    return (
        np.where(np.isfinite(train_x), train_x, medians),
        np.where(np.isfinite(test_x), test_x, medians),
    )


def ridge_oof_predictions(
    features: np.ndarray,
    scores: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    interactions: bool,
) -> tuple[np.ndarray, list[float]]:
    predictions = np.empty_like(scores)
    selected_alphas = []
    for train, test in folds:
        train_x, test_x = impute_training_medians(features[train], features[test])
        if interactions:
            expansion = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            train_x = expansion.fit_transform(train_x)
            test_x = expansion.transform(test_x)
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
        model = RidgeCV(alphas=np.asarray(ALPHAS, dtype=float), gcv_mode="svd")
        model.fit(train_x, scores[train])
        predictions[test] = model.predict(test_x)
        selected_alphas.append(float(model.alpha_))
    return predictions, selected_alphas


def prediction_metrics(scores: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(scores, predictions)),
        "spearman_rho": float(spearmanr(scores, predictions).statistic),
        "pearson_r": float(pearsonr(scores, predictions).statistic),
        "rmse": float(np.sqrt(np.mean((scores - predictions) ** 2))),
    }


def multivariable_models(
    groups: dict[str, tuple[np.ndarray, bool]],
    scores: np.ndarray,
    seed: int,
    permutation_draws: int,
    repeated_splits: int,
) -> dict:
    base_folds = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(scores))
    repeated = RepeatedKFold(
        n_splits=5,
        n_repeats=repeated_splits,
        random_state=seed,
    )
    all_repeated_splits = list(repeated.split(scores))
    output = {}
    raw_pvalues = {}
    for group_index, (name, (features, interactions)) in enumerate(groups.items()):
        predictions, selected_alphas = ridge_oof_predictions(
            features, scores, base_folds, interactions
        )
        metrics = prediction_metrics(scores, predictions)
        rng = np.random.default_rng(seed + group_index * 10_000)
        extreme = 0
        for _ in range(permutation_draws):
            permuted = rng.permutation(scores)
            null_predictions, _ = ridge_oof_predictions(
                features, permuted, base_folds, interactions
            )
            null_rho = spearmanr(permuted, null_predictions).statistic
            extreme += null_rho >= metrics["spearman_rho"]
        pvalue = float((extreme + 1) / (permutation_draws + 1))

        by_repeat = []
        for repeat in range(repeated_splits):
            repeat_folds = all_repeated_splits[repeat * 5:(repeat + 1) * 5]
            repeat_predictions, _ = ridge_oof_predictions(
                features, scores, repeat_folds, interactions
            )
            by_repeat.append(prediction_metrics(scores, repeat_predictions))
        r2_values = np.asarray([item["r2"] for item in by_repeat])
        rho_values = np.asarray([item["spearman_rho"] for item in by_repeat])
        output[name] = {
            **metrics,
            "feature_count": int(features.shape[1]),
            "pairwise_interactions": interactions,
            "selected_alphas": selected_alphas,
            "spearman_permutation_pvalue": pvalue,
            "split_distribution": {
                "repeats": repeated_splits,
                "r2_median": float(np.median(r2_values)),
                "r2_95interval": [float(value) for value in np.quantile(r2_values, (0.025, 0.975))],
                "spearman_median": float(np.median(rho_values)),
                "spearman_95interval": [float(value) for value in np.quantile(rho_values, (0.025, 0.975))],
                "r2_values": r2_values.tolist(),
                "spearman_values": rho_values.tolist(),
            },
            "oof_predictions": predictions.tolist(),
        }
        raw_pvalues[name] = pvalue
    adjusted = holm_adjust_local(raw_pvalues)
    for name, value in adjusted.items():
        output[name]["holm_pvalue"] = value
    return output


def analyze(
    embedding_path: Path,
    acoustic_path: Path,
    interpretable_cache_path: Path,
    seed: int = 20260818,
    association_permutations: int = 10_000,
    regression_permutations: int = 1_000,
    bootstrap_draws: int = 5_000,
    repeated_splits: int = 25,
) -> dict:
    experiment = experiment_arrays(embedding_path)
    targets, delta = largest_displacements(experiment, "tongueIndex")
    score_result = cross_fitted_pls_score(experiment["starting"], targets, seed)
    scores = score_result.pop("score")
    full_scores = score_result.pop("full_sample_score")

    acoustic_metadata = json.loads(acoustic_path.with_suffix(".metadata.json").read_text(encoding="utf8"))
    acoustic_features, acoustic_validation = load_or_render_interpretable_acoustics(
        embedding_path, acoustic_path, interpretable_cache_path, acoustic_metadata
    )
    physical_features = {
        str(name): experiment["base_parameters"][:, index]
        for index, name in enumerate(experiment["parameter_names"])
    }
    feature_groups = {"physical": physical_features, "acoustic": acoustic_features}
    associations = univariate_associations(
        feature_groups,
        scores,
        seed + 1_000,
        association_permutations,
        bootstrap_draws,
    )

    physical_matrix = np.column_stack(list(physical_features.values()))
    acoustic_matrix = np.column_stack(list(acoustic_features.values()))
    tongue_position = physical_features["tongueIndex"][:, None]
    second_formant = acoustic_features["formantF2Hz"][:, None]
    model_groups = {
        "tongue_position_only": (tongue_position, False),
        "f2_only": (second_formant, False),
        "tongue_position_plus_f2": (np.column_stack((tongue_position, second_formant)), False),
        "physical_additive": (physical_matrix, False),
        "acoustic_additive": (acoustic_matrix, False),
        "combined_additive": (np.column_stack((physical_matrix, acoustic_matrix)), False),
    }
    models = multivariable_models(
        model_groups,
        scores,
        seed + 2_000,
        regression_permutations,
        repeated_splits,
    )
    f2_r2 = np.asarray(models["f2_only"]["split_distribution"]["r2_values"])
    tongue_r2 = np.asarray(models["tongue_position_only"]["split_distribution"]["r2_values"])
    combined_r2 = np.asarray(models["tongue_position_plus_f2"]["split_distribution"]["r2_values"])

    def split_difference(left: np.ndarray, right: np.ndarray) -> dict:
        differences = left - right
        return {
            "median_r2_difference": float(np.median(differences)),
            "r2_difference_95interval": [
                float(value) for value in np.quantile(differences, (0.025, 0.975))
            ],
        }

    return {
        "experiment": "interpret-tongue-response-pls-scalar-v6",
        "sample_count": int(scores.size),
        "intervention": {"parameter": "tongueIndex", "delta": delta},
        "score_definition": {
            "input": "unit-normalized starting HuBERT embedding",
            "target": "HuBERT displacement under tongueIndex +0.5",
            "estimator": "one-component PLS fitted in five outer folds",
            "test_score_is_cross_fitted": True,
            **score_result,
        },
        "acoustic_validation": acoustic_validation,
        "features": {
            "physical": {name: values.tolist() for name, values in physical_features.items()},
            "acoustic": {name: values.tolist() for name, values in acoustic_features.items()},
        },
        "cross_fitted_scores": scores.tolist(),
        "full_sample_scores": full_scores.tolist(),
        "univariate_associations": associations,
        "candidate_relationships": {
            "tongue_position_vs_f2_spearman": float(
                spearmanr(tongue_position[:, 0], second_formant[:, 0]).statistic
            )
        },
        "multivariable_models": models,
        "model_comparisons": {
            "f2_minus_tongue_position": split_difference(f2_r2, tongue_r2),
            "tongue_position_plus_f2_minus_f2": split_difference(combined_r2, f2_r2),
            "note": "Descriptive differences over the same 25 repeated five-fold partitions; splits are not independent inferential replicates.",
        },
        "inference": {
            "association_permutations": association_permutations,
            "regression_permutations": regression_permutations,
            "bootstrap_draws": bootstrap_draws,
            "repeated_five_fold_splits": repeated_splits,
            "multiple_testing": "Holm correction within the 10 univariate tests and within the six held-out regression models",
        },
    }


def markdown_report(analysis: dict, embedding_metadata: dict) -> str:
    associations = analysis["univariate_associations"]
    ranked = sorted(associations.items(), key=lambda item: abs(item[1]["spearman_rho"]), reverse=True)
    association_rows = []
    for _, result in ranked:
        association_rows.append(
            f"| {result['label']} | {result['group']} | {result['spearman_rho']:.3f} | "
            f"{result['spearman_95ci'][0]:.3f}–{result['spearman_95ci'][1]:.3f} | "
            f"{result['univariate_linear_r2']:.3f} | {result['holm_pvalue']:.4g} |"
        )
    model_labels = {
        "tongue_position_only": "Initial tongue position only",
        "f2_only": "LPC F2 only",
        "tongue_position_plus_f2": "Initial tongue position + LPC F2",
        "physical_additive": "Physical parameters (additive)",
        "acoustic_additive": "Acoustic measurements (additive)",
        "combined_additive": "Physical + acoustic (additive)",
    }
    model_rows = []
    for name, result in analysis["multivariable_models"].items():
        interval = result["split_distribution"]["r2_95interval"]
        model_rows.append(
            f"| {model_labels[name]} | {result['feature_count']} | {result['r2']:.3f} | "
            f"{interval[0]:.3f}–{interval[1]:.3f} | {result['spearman_rho']:.3f} | "
            f"{result['holm_pvalue']:.4g} |"
        )
    strongest_name, strongest = ranked[0]
    best_model_name, best_model = max(
        analysis["multivariable_models"].items(), key=lambda item: item[1]["r2"]
    )
    significant = [item for item in ranked if item[1]["holm_pvalue"] < 0.05]
    if significant:
        univariate_conclusion = (
            f"The strongest individual correlate is **{strongest['label']}** "
            f"(Spearman ρ **{strongest['spearman_rho']:.3f}**, Holm p **{strongest['holm_pvalue']:.4g}**)."
        )
    else:
        univariate_conclusion = (
            f"No individual measurement survives multiplicity correction. The largest observed association is "
            f"**{strongest['label']}** (Spearman ρ **{strongest['spearman_rho']:.3f}**, "
            f"Holm p **{strongest['holm_pvalue']:.4g}**)."
        )
    return f"""# Experiment 6: What Does the Tongue-Response PLS Scalar Represent?

## Question

Does the compact HuBERT state identified in Experiment 5 correspond to a known starting vocal-tract parameter or acoustic property?

## Method

- Extractor: `{embedding_metadata.get('model_id', 'unknown')}` at `{embedding_metadata.get('resolved_revision', 'unknown')}`
- Starting configurations: {analysis['sample_count']}
- Intervention: `tongueIndex +{analysis['intervention']['delta']}`
- Score: one-component PLS state, trained in five folds so every reported scalar is produced by a model that did not see that starting configuration's intervention target.
- Physical candidates: the five randomized Pink Trombone starting controls.
- Acoustic candidates: framewise LPC F1/F2 plus spectral centroid, bandwidth, and flatness from the exact matched-noise starting audio.
- Univariate inference: paired bootstrap intervals, permutation tests, and Holm correction across 10 candidates.
- Multivariable inference: training-fold imputation/scaling and ridge selection, outer held-out predictions, permutation tests, and 25 repeated five-fold split checks.

The five fold-specific PLS axes have mean absolute cosine **{analysis['score_definition']['mean_fold_axis_absolute_cosine_to_full']:.3f}** with the full-data axis (minimum **{analysis['score_definition']['minimum_fold_axis_absolute_cosine_to_full']:.3f}**). Cross-fitted and full-data scores correlate at Spearman ρ **{analysis['score_definition']['cross_fitted_vs_full_score_spearman']:.3f}**. The re-rendered audio reproduces the saved log-mel cache with maximum absolute difference **{analysis['acoustic_validation']['maximum_absolute_log_mel_rerender_difference']:.3g}**.

![PLS scalar associations](./FIGURE_V6_ASSOCIATIONS.png)

## Individual physical and acoustic associations

| Candidate | Type | Spearman ρ | Bootstrap 95% interval | Linear R² | Holm p |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(association_rows)}

{univariate_conclusion}

## Can known state variables explain the scalar jointly?

![Held-out scalar prediction](./FIGURE_V6_MODELS.png)

| State description | Raw features | Held-out R² | Repeated-split 95% range | Prediction ρ | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(model_rows)}

The strongest tested state description is **{model_labels[best_model_name]}**, with held-out R² **{best_model['r2']:.3f}** and prediction ρ **{best_model['spearman_rho']:.3f}**. This is an explanation of the cross-fitted scalar, not another HuBERT displacement predictor.

Initial tongue position and LPC F2 are themselves strongly associated (Spearman ρ **{analysis['candidate_relationships']['tongue_position_vs_f2_spearman']:.3f}**). Comparing their held-out models therefore matters more than comparing their raw correlations alone. F2 by itself reaches R² **{analysis['multivariable_models']['f2_only']['r2']:.3f}**, versus **{analysis['multivariable_models']['tongue_position_only']['r2']:.3f}** for initial tongue position; using both reaches **{analysis['multivariable_models']['tongue_position_plus_f2']['r2']:.3f}**.

Across the same 25 repeated partitions, F2 improves R² over tongue position by a median **{analysis['model_comparisons']['f2_minus_tongue_position']['median_r2_difference']:.3f}** (split range **{analysis['model_comparisons']['f2_minus_tongue_position']['r2_difference_95interval'][0]:.3f}–{analysis['model_comparisons']['f2_minus_tongue_position']['r2_difference_95interval'][1]:.3f}**). Adding tongue position to F2 changes R² by median **{analysis['model_comparisons']['tongue_position_plus_f2_minus_f2']['median_r2_difference']:.3f}** (split range **{analysis['model_comparisons']['tongue_position_plus_f2_minus_f2']['r2_difference_95interval'][0]:.3f}–{analysis['model_comparisons']['tongue_position_plus_f2_minus_f2']['r2_difference_95interval'][1]:.3f}**). These split ranges describe robustness to partition choice, not independent confidence intervals.

## Interpretation

The compact state is best described by the starting second-formant configuration among the variables tested. The raw tongue coordinate is also strongly related, but it does not improve held-out reconstruction when added to F2. This is consistent with the scalar tracking the acoustic consequence of the starting tongue configuration more directly than the simulator coordinate itself. It does not establish that F2 causally mediates HuBERT's response.

The scalar's sign is conventional: reversing the PLS axis reverses every signed correlation but leaves association magnitude, p-values, and regression performance unchanged.

## Limitations

There are only 50 synthetic starting states. LPC formants are estimates from short synthesized clips, not measured human vocal-tract resonances. The PLS axis is defined using HuBERT responses from this same intervention dataset, although each analyzed score is cross-fitted. Correlation does not establish that HuBERT explicitly encodes an anatomical variable or that the relationship transfers to natural speech.
"""


def plot_results(analysis: dict, association_path: Path, model_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    associations = sorted(
        analysis["univariate_associations"].values(),
        key=lambda item: abs(item["spearman_rho"]),
    )
    labels = [item["label"] for item in associations]
    values = np.asarray([item["spearman_rho"] for item in associations])
    intervals = np.asarray([item["spearman_95ci"] for item in associations])
    colors = ["#4f7c8d" if item["group"] == "physical" else "#a52562" for item in associations]
    figure, axis = plt.subplots(figsize=(9.5, 6.2))
    positions = np.arange(len(labels))
    axis.barh(positions, values, color=colors, alpha=0.9)
    axis.errorbar(
        values,
        positions,
        xerr=np.vstack((values - intervals[:, 0], intervals[:, 1] - values)),
        fmt="none",
        ecolor="#303846",
        elinewidth=1.1,
        capsize=2,
    )
    axis.axvline(0, color="#303846", linewidth=1)
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Spearman correlation with cross-fitted PLS score")
    axis.set_title("What tracks the compact tongue-response state?", loc="left", fontweight="bold")
    figure.tight_layout()
    figure.savefig(association_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    model_labels = {
        "tongue_position_only": "Tongue\nposition",
        "f2_only": "LPC F2",
        "tongue_position_plus_f2": "Tongue +\nF2",
        "physical_additive": "Physical\nadditive",
        "acoustic_additive": "Acoustic\nadditive",
        "combined_additive": "Combined\nadditive",
    }
    models = analysis["multivariable_models"]
    names = list(models)
    values = np.asarray([models[name]["r2"] for name in names])
    intervals = np.asarray([models[name]["split_distribution"]["r2_95interval"] for name in names])
    interval_lower = np.minimum(intervals[:, 0], values)
    interval_upper = np.maximum(intervals[:, 1], values)
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    positions = np.arange(len(names))
    axis.bar(
        positions,
        values,
        color=["#4f7c8d", "#a52562", "#7755a6", "#6e93a2", "#bd5a88", "#9175b6"],
    )
    axis.errorbar(
        positions,
        values,
        yerr=np.vstack((values - interval_lower, interval_upper - values)),
        fmt="none",
        ecolor="#303846",
        elinewidth=1.1,
        capsize=3,
    )
    axis.axhline(0, color="#303846", linewidth=1)
    axis.set_xticks(positions, [model_labels[name] for name in names])
    axis.set_ylabel("Held-out R² for the PLS scalar")
    axis.set_title("Can known starting state reconstruct the scalar?", loc="left", fontweight="bold")
    figure.tight_layout()
    figure.savefig(model_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--acoustics", type=Path, default=DEFAULT_ACOUSTICS)
    parser.add_argument("--interpretable-cache", type=Path, default=DEFAULT_INTERPRETABLE_CACHE)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--association-figure", type=Path)
    parser.add_argument("--model-figure", type=Path)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--association-permutations", type=int, default=10_000)
    parser.add_argument("--regression-permutations", type=int, default=1_000)
    parser.add_argument("--bootstrap-draws", type=int, default=5_000)
    parser.add_argument("--repeated-splits", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedding_path = args.embeddings.resolve()
    acoustic_path = args.acoustics.resolve()
    interpretable_cache_path = args.interpretable_cache.resolve()
    output_json = (args.output_json or embedding_path.with_name("analysis_v6.json")).resolve()
    report_path = (args.report or embedding_path.with_name("REPORT_V6.md")).resolve()
    association_path = (
        args.association_figure or embedding_path.with_name("FIGURE_V6_ASSOCIATIONS.png")
    ).resolve()
    model_path = (args.model_figure or embedding_path.with_name("FIGURE_V6_MODELS.png")).resolve()
    embedding_metadata = json.loads(
        embedding_path.with_suffix(".metadata.json").read_text(encoding="utf8")
    )
    result = analyze(
        embedding_path,
        acoustic_path,
        interpretable_cache_path,
        args.seed,
        args.association_permutations,
        args.regression_permutations,
        args.bootstrap_draws,
        args.repeated_splits,
    )
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    report_path.write_text(markdown_report(result, embedding_metadata), encoding="utf8")
    plot_results(result, association_path, model_path)
    print(f"Analysis: {output_json}")
    print(f"Report: {report_path}")
    print(f"Figures: {association_path}, {model_path}")


if __name__ == "__main__":
    main()
