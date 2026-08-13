"""Train and run a phoneme-spectrum discriminator using scikit-fda."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.signal import resample_poly
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PHONEMES = ("aa", "ao", "dcl", "iy", "sh")
FRAME_SIZE = 512
SAMPLE_RATE = 16_000
SPECTRUM_SIZE = 256


def normalize_curve(curve: np.ndarray) -> np.ndarray:
    """Remove absolute level so the model compares spectral shape."""
    curve = np.asarray(curve, dtype=np.float64).reshape(-1)
    deviation = float(curve.std())
    if deviation < 1e-8:
        return np.zeros_like(curve)
    return (curve - float(curve.mean())) / deviation


def audio_to_log_periodogram(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert arbitrary mono PCM into the dataset's 512-sample spectral shape."""
    signal = np.asarray(samples, dtype=np.float64).reshape(-1)
    if signal.size == 0 or not np.all(np.isfinite(signal)):
        raise ValueError("Audio samples must be a non-empty finite array.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    signal -= float(signal.mean())
    if sample_rate != SAMPLE_RATE:
        divisor = int(np.gcd(sample_rate, SAMPLE_RATE))
        signal = resample_poly(signal, SAMPLE_RATE // divisor, sample_rate // divisor)
    if signal.size < FRAME_SIZE:
        signal = np.pad(signal, (0, FRAME_SIZE - signal.size))
    elif signal.size > FRAME_SIZE:
        start = (signal.size - FRAME_SIZE) // 2
        signal = signal[start:start + FRAME_SIZE]
    power = np.abs(np.fft.fft(signal, n=FRAME_SIZE)[:SPECTRUM_SIZE]) ** 2 / FRAME_SIZE
    return normalize_curve(np.log(np.maximum(power, 1e-12)))


def _extract_dataset() -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    from skfda.datasets import fetch_phoneme

    dataset = fetch_phoneme()
    curves = np.asarray(dataset.data.data_matrix, dtype=np.float64).reshape(-1, SPECTRUM_SIZE)
    labels = np.asarray(dataset.target)
    categories = getattr(dataset, "categories", {})
    target_names = tuple(str(name) for name in categories.get("phoneme", PHONEMES))
    if labels.dtype.kind in "OUS":
        name_to_index = {name: index for index, name in enumerate(target_names)}
        labels = np.asarray([name_to_index[str(label)] for label in labels], dtype=np.int64)
    else:
        labels = labels.astype(np.int64)
    curves = np.stack([normalize_curve(curve) for curve in curves])
    utterances = np.asarray(getattr(dataset, "meta", np.arange(curves.shape[0]))).reshape(-1).astype(str)
    return curves, labels, target_names, utterances


@dataclass
class TrainingReport:
    accuracy: float
    samples: int
    phonemes: tuple[str, ...]


class PhonemeDiscriminator:
    """MLP phoneme discriminator plus target-centroid spectral similarity."""

    def __init__(self, model: Any, centroids: np.ndarray, target_names: tuple[str, ...], report: TrainingReport):
        self.model = model
        self.centroids = centroids
        self.target_names = target_names
        self.report = report

    @classmethod
    def train(cls, random_state: int = 17) -> "PhonemeDiscriminator":
        curves, labels, target_names, utterances = _extract_dataset()
        predefined_test = np.char.startswith(utterances, "test.")
        if predefined_test.any() and (~predefined_test).any():
            train_x, train_y = curves[~predefined_test], labels[~predefined_test]
            validation_x, validation_y = curves[predefined_test], labels[predefined_test]
        else:
            train_x, validation_x, train_y, validation_y = train_test_split(
                curves,
                labels,
                test_size=0.2,
                random_state=random_state,
                stratify=labels,
            )
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 48),
                activation="relu",
                alpha=1e-3,
                batch_size=128,
                learning_rate_init=8e-4,
                max_iter=220,
                early_stopping=True,
                validation_fraction=0.12,
                n_iter_no_change=14,
                random_state=random_state,
            ),
        )
        model.fit(train_x, train_y)
        accuracy = accuracy_score(validation_y, model.predict(validation_x))
        centroids = np.stack([train_x[train_y == index].mean(axis=0) for index in range(len(target_names))])
        centroids = np.stack([normalize_curve(curve) for curve in centroids])
        report = TrainingReport(float(accuracy), int(curves.shape[0]), target_names)
        return cls(model, centroids, target_names, report)

    def score(self, samples: np.ndarray, sample_rate: int, phoneme: str) -> dict[str, Any]:
        if phoneme not in self.target_names:
            raise ValueError(f"Unsupported phoneme '{phoneme}'. Choose one of: {', '.join(self.target_names)}")
        curve = audio_to_log_periodogram(samples, sample_rate)
        probabilities = self.model.predict_proba(curve[None, :])[0]
        index = self.target_names.index(phoneme)
        class_index = list(self.model.classes_).index(index)
        probability = float(probabilities[class_index])
        cosine = float(np.dot(curve, self.centroids[index]) / (
            np.linalg.norm(curve) * np.linalg.norm(self.centroids[index]) + 1e-12
        ))
        centroid_similarity = float(np.clip((cosine + 1) / 2, 0, 1))
        combined = 0.7 * probability + 0.3 * centroid_similarity
        return {
            "phoneme": phoneme,
            "score": float(np.clip(combined, 0, 1)),
            "discriminatorProbability": probability,
            "centroidSimilarity": centroid_similarity,
            "predictedPhoneme": self.target_names[int(self.model.predict(curve[None, :])[0])],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load_or_train(cls, path: Path) -> "PhonemeDiscriminator":
        if path.exists():
            cached = joblib.load(path)
            if tuple(cached.target_names) == PHONEMES:
                return cached
        discriminator = cls.train()
        discriminator.save(path)
        return discriminator
