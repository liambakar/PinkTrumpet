"""Experiment 2: measure physical-control directions in HuBERT space.

The experiment keeps every base configuration paired with all interventions,
uses the same synthesis noise seed inside each pair, and checkpoints raw
embeddings so every downstream statistic can be recomputed independently.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from scipy import signal

try:
    from . import engine
except ImportError:  # Direct script execution.
    import engine


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(__file__).with_name("experiment_v2.npz")
MODEL_ID = "facebook/hubert-base-ls960"
MODEL_REVISION = "dba3bb02fda4248b6e082697eee756de8fe8aa8a"
PARAMETER_NAMES = (
    "tongueIndex",
    "tongueDiameter",
    "constrictionIndex",
    "constrictionDiameter",
    "pitchHz",
)


@dataclass(frozen=True)
class Intervention:
    name: str
    deltas: tuple[float, ...]
    base_range: tuple[float, float]
    unit: str


INTERVENTIONS = (
    Intervention("tongueIndex", (-0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5), (12.5, 28.5), "tract sections"),
    Intervention("tongueDiameter", (-0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5), (2.55, 3.0), "diameter units"),
    Intervention("constrictionIndex", (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5), (20.0, 37.0), "tract sections"),
    Intervention("constrictionDiameter", (-0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5), (0.55, 0.9), "diameter units"),
    Intervention("pitchHz", (-25, -20, -15, -10, -5, 5, 10, 15, 20, 25), (100.0, 200.0), "Hz"),
)

STATIC_PARAMETERS = {
    "tenseness": 0.6,
    "intensity": 1.0,
    "loudness": 1.0,
    "voicing": 1.0,
    "aspiration": 0.0,
    "vibrato": 0.0,
    "wobble": 0.0,
    "fricativeIntensity": 0.0,
    "velum": 0.01,
}

_PARALLEL_RENDERING_AVAILABLE = True


@dataclass(frozen=True)
class Variant:
    parameter: str
    delta: float


def experiment_variants(positive_only: bool = False) -> tuple[Variant, ...]:
    variants = [Variant("base", 0.0)]
    for intervention in INTERVENTIONS:
        for delta in intervention.deltas:
            if positive_only and delta < 0:
                continue
            variants.append(Variant(intervention.name, delta))
    return tuple(variants)


def make_base_configurations(sample_count: int, seed: int) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    configurations = []
    for _ in range(sample_count):
        parameters = dict(STATIC_PARAMETERS)
        for intervention in INTERVENTIONS:
            low, high = intervention.base_range
            parameters[intervention.name] = float(rng.uniform(low, high))
        configurations.append(parameters)
    return configurations


def parameters_for_variant(base: dict[str, float], variant: Variant) -> dict[str, float]:
    parameters = dict(base)
    if variant.parameter != "base":
        parameters[variant.parameter] += variant.delta
    return parameters


def render_audio_task(task: tuple[dict[str, float], float, float, int]) -> np.ndarray:
    parameters, duration, warmup, noise_seed = task
    start = round(warmup * engine.SAMPLE_RATE)
    audio = engine.generate_audio(parameters, duration + warmup, seed=noise_seed)
    return np.asarray(audio[start:], dtype=np.float32)


def render_variants(
    base: dict[str, float],
    variants: tuple[Variant, ...],
    duration: float,
    warmup: float,
    noise_seed: int,
    workers: int,
) -> list[np.ndarray]:
    global _PARALLEL_RENDERING_AVAILABLE
    tasks = [
        (parameters_for_variant(base, variant), duration, warmup, noise_seed)
        for variant in variants
    ]
    if workers <= 1 or not _PARALLEL_RENDERING_AVAILABLE:
        return [render_audio_task(task) for task in tasks]
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(render_audio_task, tasks))
    except OSError as error:
        _PARALLEL_RENDERING_AVAILABLE = False
        print(
            f"Parallel rendering unavailable ({error}); falling back to one process.",
            file=sys.stderr,
            flush=True,
        )
        return [render_audio_task(task) for task in tasks]


class MockExtractor:
    """Fast deterministic spectral extractor for pipeline tests only."""

    model_id = "mock-log-spectrum"
    revision = "local"
    resolved_revision = "local"
    dimension = 128
    device = "cpu"

    def embed(self, clips: list[np.ndarray], batch_size: int) -> np.ndarray:
        del batch_size
        embeddings = []
        for audio in clips:
            audio_16k = signal.resample_poly(audio, 160, 441)
            spectrum = np.abs(np.fft.rfft(audio_16k, n=4096))[: self.dimension]
            embeddings.append(np.log1p(spectrum))
        return np.asarray(embeddings, dtype=np.float32)

    def versions(self) -> dict[str, str]:
        return {}


class HubertExtractor:
    def __init__(self, model_id: str, revision: str | None, device: str, seed: int):
        try:
            import torch
            import transformers
            from transformers import HubertModel, Wav2Vec2FeatureExtractor
        except ImportError as error:
            raise RuntimeError(
                "HuBERT dependencies are missing. Install requirements-research.txt "
                "or run with --extractor mock for a pipeline smoke test."
            ) from error

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        torch.manual_seed(seed)
        self._torch = torch
        self._transformers_version = transformers.__version__
        self.model_id = model_id
        self.revision = revision or MODEL_REVISION
        self.device = device
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, revision=self.revision)
        self.model = HubertModel.from_pretrained(model_id, revision=self.revision).to(device)
        self.model.eval()
        self.dimension = int(self.model.config.hidden_size)
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None) or self.revision

    def embed(self, clips: list[np.ndarray], batch_size: int) -> np.ndarray:
        embeddings = []
        torch = self._torch
        for start in range(0, len(clips), batch_size):
            batch = [signal.resample_poly(audio, 160, 441).astype(np.float32) for audio in clips[start:start + batch_size]]
            inputs = self.processor(batch, sampling_rate=16_000, return_tensors="pt", padding=True)
            inputs = inputs.to(self.device)
            with torch.no_grad():
                hidden = self.model(**inputs).last_hidden_state
                pooled = hidden.mean(dim=1).cpu().numpy().astype(np.float32)
            embeddings.append(pooled)
        return np.concatenate(embeddings, axis=0)

    def versions(self) -> dict[str, str]:
        return {
            "torch": self._torch.__version__,
            "transformers": self._transformers_version,
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_checkpoint(
    path: Path,
    embeddings: np.ndarray,
    base_parameters: np.ndarray,
    completed: np.ndarray,
    variants: tuple[Variant, ...],
) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        embeddings=embeddings,
        base_parameters=base_parameters,
        parameter_names=np.asarray(PARAMETER_NAMES),
        variant_parameters=np.asarray([variant.parameter for variant in variants]),
        variant_deltas=np.asarray([variant.delta for variant in variants], dtype=np.float64),
        completed=completed,
    )
    temporary.replace(path)


def write_metadata(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf8")
    temporary.replace(path)


def load_or_initialize(
    output: Path,
    sample_count: int,
    variants: tuple[Variant, ...],
    dimension: int,
    base_parameters: np.ndarray,
    overwrite: bool,
) -> tuple[np.ndarray, np.ndarray]:
    expected_shape = (sample_count, len(variants), dimension)
    if output.exists() and not overwrite:
        checkpoint = np.load(output, allow_pickle=False)
        embeddings = checkpoint["embeddings"]
        completed = checkpoint["completed"].astype(bool)
        if embeddings.shape != expected_shape:
            raise ValueError(
                f"Checkpoint shape {embeddings.shape} does not match requested {expected_shape}. "
                "Use --overwrite or matching arguments."
            )
        if not np.allclose(checkpoint["base_parameters"], base_parameters):
            raise ValueError("Checkpoint base configurations do not match the requested seed.")
        return embeddings, completed
    return np.full(expected_shape, np.nan, dtype=np.float32), np.zeros(sample_count, dtype=bool)


def run(args: argparse.Namespace) -> Path:
    variants = experiment_variants(args.positive_only)
    bases = make_base_configurations(args.samples, args.seed)
    base_matrix = np.asarray(
        [[parameters[name] for name in PARAMETER_NAMES] for parameters in bases],
        dtype=np.float64,
    )
    extractor = (
        MockExtractor()
        if args.extractor == "mock"
        else HubertExtractor(args.model, args.revision, args.device, args.seed)
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    embeddings, completed = load_or_initialize(
        output,
        args.samples,
        variants,
        extractor.dimension,
        base_matrix,
        args.overwrite,
    )
    metadata_path = output.with_suffix(".metadata.json")
    metadata = {
        "experiment": "physical-displacement-directions-v2",
        "created_at_unix": time.time(),
        "sample_count": args.samples,
        "seed": args.seed,
        "duration_seconds": args.duration,
        "warmup_seconds": args.warmup,
        "synthesis_sample_rate_hz": engine.SAMPLE_RATE,
        "embedding_sample_rate_hz": 16_000,
        "matched_noise_within_configuration": True,
        "vibrato_wobble_aspiration_and_frication_disabled": True,
        "smooth_source_modulation": "deterministic and matched within each configuration",
        "extractor": args.extractor,
        "model_id": extractor.model_id,
        "requested_revision": extractor.revision,
        "resolved_revision": extractor.resolved_revision,
        "embedding_dimension": extractor.dimension,
        "device": extractor.device,
        "positive_only": args.positive_only,
        "interventions": [
            {
                "name": item.name,
                "deltas": [value for value in item.deltas if not args.positive_only or value > 0],
                "base_range": list(item.base_range),
                "unit": item.unit,
            }
            for item in INTERVENTIONS
        ],
        "static_parameters": STATIC_PARAMETERS,
        "engine_sha256": file_sha256(Path(engine.__file__)),
        "git_revision": git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            **extractor.versions(),
        },
    }
    started = time.monotonic()
    pending = np.flatnonzero(~completed)
    for position, sample_index in enumerate(pending, 1):
        noise_seed = args.seed * 1_000_003 + int(sample_index)
        clips = render_variants(
            bases[sample_index],
            variants,
            args.duration,
            args.warmup,
            noise_seed,
            args.workers,
        )
        sample_embeddings = extractor.embed(clips, args.batch_size)
        if sample_embeddings.shape != (len(variants), extractor.dimension):
            raise RuntimeError(f"Unexpected embedding shape {sample_embeddings.shape}.")
        embeddings[sample_index] = sample_embeddings
        completed[sample_index] = True
        save_checkpoint(output, embeddings, base_matrix, completed, variants)
        metadata["completed_samples"] = int(completed.sum())
        write_metadata(metadata_path, metadata)
        elapsed = time.monotonic() - started
        average = elapsed / position
        remaining = average * (len(pending) - position)
        print(
            f"[{int(completed.sum()):03d}/{args.samples:03d}] "
            f"saved {output.name} · ETA {remaining / 60:.1f} min",
            flush=True,
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--duration", type=float, default=0.2)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--extractor", choices=("hubert", "mock"), default="hubert")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--positive-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if args.duration <= 0 or args.warmup < 0:
        parser.error("duration must be positive and warmup cannot be negative")
    if args.workers < 1 or args.batch_size < 1:
        parser.error("workers and batch size must be positive")
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(f"Experiment complete: {result}")
