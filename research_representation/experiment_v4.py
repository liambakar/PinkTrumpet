"""Render matched low-level acoustic features for Experiment 4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import signal

try:
    from . import engine
    from .experiment import (
        INTERVENTIONS,
        PARAMETER_NAMES,
        STATIC_PARAMETERS,
        Variant,
        render_variants,
    )
except ImportError:  # Direct script execution.
    import engine
    from experiment import INTERVENTIONS, PARAMETER_NAMES, STATIC_PARAMETERS, Variant, render_variants


DEFAULT_SOURCE = Path(__file__).with_name("experiment_v2.npz")
DEFAULT_OUTPUT = Path(__file__).with_name("acoustic_v4.npz")
MEL_BINS = 40
FFT_SIZE = 2_048
WINDOW_SECONDS = 0.025
HOP_SECONDS = 0.010
MIN_FREQUENCY = 80.0
MAX_FREQUENCY = 8_000.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hz_to_mel(frequency: np.ndarray | float) -> np.ndarray:
    return 2_595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10 ** (np.asarray(mel) / 2_595.0) - 1.0)


def mel_filterbank(
    sample_rate: int = engine.SAMPLE_RATE,
    fft_size: int = FFT_SIZE,
    mel_bins: int = MEL_BINS,
    minimum_frequency: float = MIN_FREQUENCY,
    maximum_frequency: float = MAX_FREQUENCY,
) -> np.ndarray:
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    mel_edges = np.linspace(hz_to_mel(minimum_frequency), hz_to_mel(maximum_frequency), mel_bins + 2)
    hz_edges = mel_to_hz(mel_edges)
    filters = np.zeros((mel_bins, frequencies.size), dtype=np.float64)
    for index in range(mel_bins):
        left, center, right = hz_edges[index:index + 3]
        filters[index] = np.maximum(
            0.0,
            np.minimum(
                (frequencies - left) / max(center - left, 1e-12),
                (right - frequencies) / max(right - center, 1e-12),
            ),
        )
        normalization = filters[index].sum()
        if normalization > 0:
            filters[index] /= normalization
    return filters


def log_mel_feature(audio: np.ndarray, sample_rate: int = engine.SAMPLE_RATE) -> np.ndarray:
    """Return a time-aligned flattened log-mel power representation."""
    window = round(WINDOW_SECONDS * sample_rate)
    hop = round(HOP_SECONDS * sample_rate)
    _, _, spectrum = signal.stft(
        np.asarray(audio, dtype=np.float64),
        fs=sample_rate,
        window="hann",
        nperseg=window,
        noverlap=window - hop,
        nfft=FFT_SIZE,
        boundary=None,
        padded=False,
    )
    power = np.abs(spectrum) ** 2
    mel_power = mel_filterbank(sample_rate) @ power
    log_mel = np.log(mel_power + 1e-10)
    return log_mel.T.reshape(-1).astype(np.float32)


def largest_positive_variants() -> tuple[Variant, ...]:
    return (
        Variant("base", 0.0),
        *(Variant(item.name, max(item.deltas)) for item in INTERVENTIONS),
    )


def base_configurations(source: np.lib.npyio.NpzFile) -> list[dict[str, float]]:
    names = source["parameter_names"].astype(str).tolist()
    if tuple(names) != PARAMETER_NAMES:
        raise ValueError("Experiment 2 parameter order does not match the current experiment code.")
    return [
        {**STATIC_PARAMETERS, **dict(zip(names, row.astype(float), strict=True))}
        for row in source["base_parameters"]
    ]


def save_checkpoint(
    path: Path,
    base_features: np.ndarray,
    intervention_features: np.ndarray,
    completed: np.ndarray,
    variants: tuple[Variant, ...],
) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        base_features=base_features,
        intervention_features=intervention_features,
        completed=completed,
        intervention_parameters=np.asarray([variant.parameter for variant in variants[1:]]),
        intervention_deltas=np.asarray([variant.delta for variant in variants[1:]], dtype=np.float64),
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> Path:
    source_path = args.source.resolve()
    source = np.load(source_path, allow_pickle=False)
    if not np.all(source["completed"].astype(bool)):
        raise ValueError("Experiment 2 checkpoint is incomplete.")
    metadata_path = source_path.with_suffix(".metadata.json")
    source_metadata = json.loads(metadata_path.read_text(encoding="utf8"))
    configurations = base_configurations(source)
    variants = largest_positive_variants()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    probe = render_variants(
        configurations[0], variants, args.duration, args.warmup, args.seed * 1_000_003, 1
    )
    feature_dimension = log_mel_feature(probe[0]).size
    expected_base_shape = (len(configurations), feature_dimension)
    expected_intervention_shape = (len(configurations), len(variants) - 1, feature_dimension)
    if output.exists() and not args.overwrite:
        checkpoint = np.load(output, allow_pickle=False)
        base_features = checkpoint["base_features"]
        intervention_features = checkpoint["intervention_features"]
        completed = checkpoint["completed"].astype(bool)
        if base_features.shape != expected_base_shape or intervention_features.shape != expected_intervention_shape:
            raise ValueError("Existing acoustic checkpoint shape does not match the requested configuration.")
    else:
        base_features = np.full(expected_base_shape, np.nan, dtype=np.float32)
        intervention_features = np.full(expected_intervention_shape, np.nan, dtype=np.float32)
        completed = np.zeros(len(configurations), dtype=bool)

    started = time.monotonic()
    pending = np.flatnonzero(~completed)
    for position, sample_index in enumerate(pending, 1):
        clips = render_variants(
            configurations[sample_index],
            variants,
            args.duration,
            args.warmup,
            args.seed * 1_000_003 + int(sample_index),
            args.workers,
        )
        features = np.asarray([log_mel_feature(clip) for clip in clips], dtype=np.float32)
        base_features[sample_index] = features[0]
        intervention_features[sample_index] = features[1:]
        completed[sample_index] = True
        save_checkpoint(output, base_features, intervention_features, completed, variants)
        elapsed = time.monotonic() - started
        remaining = elapsed / position * (len(pending) - position)
        print(
            f"[{int(completed.sum()):03d}/{len(configurations):03d}] saved {output.name} · "
            f"ETA {remaining / 60:.1f} min",
            flush=True,
        )

    metadata = {
        "experiment": "low-level-acoustic-features-v4",
        "source_checkpoint": source_path.name,
        "source_checkpoint_sha256": file_sha256(source_path),
        "source_engine_sha256": source_metadata.get("engine_sha256"),
        "current_engine_sha256": file_sha256(Path(engine.__file__)),
        "sample_count": len(configurations),
        "seed": args.seed,
        "duration_seconds": args.duration,
        "warmup_seconds": args.warmup,
        "matched_noise_with_source_experiment": True,
        "feature": {
            "name": "flattened-log-mel-power",
            "dimension": feature_dimension,
            "mel_bins": MEL_BINS,
            "fft_size": FFT_SIZE,
            "window_seconds": WINDOW_SECONDS,
            "hop_seconds": HOP_SECONDS,
            "minimum_frequency_hz": MIN_FREQUENCY,
            "maximum_frequency_hz": MAX_FREQUENCY,
        },
        "interventions": [
            {"parameter": variant.parameter, "delta": variant.delta}
            for variant in variants[1:]
        ],
    }
    temporary = output.with_suffix(".metadata.tmp.json")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf8")
    temporary.replace(output.with_suffix(".metadata.json"))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--duration", type=float, default=0.2)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0 or args.warmup < 0 or args.workers < 1:
        parser.error("duration and workers must be positive; warmup cannot be negative")
    return args


def main() -> None:
    output = run(parse_args())
    print(f"Acoustic checkpoint complete: {output}")


if __name__ == "__main__":
    main()
