"""Render matched Pink Trombone states and emit acoustic parity results as JSON."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from scipy import linalg, signal


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 48_000
WARMUP_SECONDS = 0.45
DURATION_SECONDS = 0.6

BASE = {
    "pitchHz": 140.0,
    "tenseness": 0.62,
    "intensity": 1.0,
    "loudness": 1.0,
    "voicing": 1.0,
    "tongueIndex": 17.5,
    "tongueDiameter": 2.75,
    "constrictionIndex": 32.0,
    "constrictionDiameter": 3.5,
    "velum": 0.01,
    "warmupSeconds": WARMUP_SECONDS,
    "durationSeconds": DURATION_SECONDS,
}


def build_cases() -> list[dict]:
    named = [
        ("neutral", {}),
        ("open_a_low", {"pitchHz": 105, "tongueIndex": 13, "tongueDiameter": 2.55}),
        ("open_a_high", {"pitchHz": 220, "tongueIndex": 13, "tongueDiameter": 2.55}),
        ("bright_i", {"pitchHz": 155, "tongueIndex": 27.4, "tongueDiameter": 3.15,
                      "constrictionIndex": 40, "constrictionDiameter": 2.4}),
        ("rounded_u", {"pitchHz": 125, "tongueIndex": 23, "tongueDiameter": 3.3,
                       "constrictionIndex": 41, "constrictionDiameter": 0.9}),
        ("mid_e", {"pitchHz": 175, "tongueIndex": 20, "tongueDiameter": 3.15}),
        ("low_back", {"pitchHz": 115, "tongueIndex": 12.3, "tongueDiameter": 2.15}),
        ("nasal_m", {"pitchHz": 135, "tongueIndex": 17, "tongueDiameter": 2.7,
                     "constrictionIndex": 41, "constrictionDiameter": 0.05, "velum": 0.4}),
    ]
    rng = np.random.default_rng(20260813)
    for index in range(8):
        named.append((f"seeded_{index + 1:02d}", {
            "pitchHz": float(rng.uniform(90, 240)),
            "tenseness": float(rng.uniform(0.45, 0.85)),
            "tongueIndex": float(rng.uniform(12, 29)),
            "tongueDiameter": float(rng.uniform(2.05, 3.5)),
            "constrictionIndex": float(rng.uniform(28, 42)),
            "constrictionDiameter": float(rng.uniform(0.75, 3.5)),
            "velum": 0.01,
        }))
    return [{"name": name, **BASE, **updates} for name, updates in named]


def encode_parameters(parameters: dict) -> str:
    payload = json.dumps(parameters, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def render(command: list[str]) -> np.ndarray:
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
    if completed.stderr:
        sys.stderr.buffer.write(completed.stderr)
    return np.frombuffer(completed.stdout, dtype=np.float32).astype(np.float64)


def write_wav(path: Path, values: np.ndarray) -> None:
    peak = max(float(np.max(np.abs(values))), 1e-12)
    pcm = np.round(np.clip(values / peak * 0.9, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm.tobytes())


def align(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    reference = reference - np.mean(reference)
    candidate = candidate - np.mean(candidate)
    max_lag = round(SAMPLE_RATE * 0.02)
    correlation = signal.correlate(candidate, reference, mode="full", method="fft")
    lags = signal.correlation_lags(candidate.size, reference.size, mode="full")
    keep = np.abs(lags) <= max_lag
    lag = int(lags[keep][np.argmax(np.abs(correlation[keep]))])
    if lag > 0:
        candidate = candidate[lag:]
        reference = reference[:candidate.size]
    elif lag < 0:
        reference = reference[-lag:]
        candidate = candidate[:reference.size]
    length = min(reference.size, candidate.size)
    return reference[:length], candidate[:length], lag


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values) + 1e-18))


def mean_spectrum(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frequencies, _, stft = signal.stft(
        values,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=2048,
        noverlap=1536,
        boundary=None,
        padded=False,
    )
    return frequencies, np.mean(np.abs(stft), axis=1) + 1e-12


def spectral_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    frequencies, ref_magnitude = mean_spectrum(reference)
    _, cur_magnitude = mean_spectrum(candidate)
    keep = (frequencies >= 80) & (frequencies <= 8_000)
    ref_db = 20 * np.log10(ref_magnitude[keep] / np.max(ref_magnitude[keep]))
    cur_db = 20 * np.log10(cur_magnitude[keep] / np.max(cur_magnitude[keep]))
    active = np.maximum(ref_db, cur_db) >= -60
    difference = ref_db[active] - cur_db[active]
    cosine = float(np.dot(ref_magnitude[keep], cur_magnitude[keep]) /
                   (np.linalg.norm(ref_magnitude[keep]) * np.linalg.norm(cur_magnitude[keep]) + 1e-18))
    return {
        "log_spectral_mae_db": float(np.mean(np.abs(difference))),
        "log_spectral_rmse_db": float(np.sqrt(np.mean(difference * difference))),
        "magnitude_cosine": cosine,
    }


def estimate_formants(values: np.ndarray) -> list[float | None]:
    frame_length = round(0.04 * SAMPLE_RATE)
    hop = round(0.02 * SAMPLE_RATE)
    order = 18
    frame_formants: list[list[float]] = []
    emphasized = signal.lfilter([1, -0.97], [1], values)
    for start in range(0, max(1, emphasized.size - frame_length), hop):
        frame = emphasized[start:start + frame_length]
        if frame.size < frame_length or rms(frame) < 1e-6:
            continue
        frame = frame * signal.windows.hamming(frame_length, sym=False)
        autocorrelation = signal.correlate(frame, frame, mode="full", method="fft")[frame_length - 1:]
        try:
            coefficients = linalg.solve_toeplitz(
                (autocorrelation[:order], autocorrelation[:order]),
                -autocorrelation[1:order + 1],
                check_finite=False,
            )
        except linalg.LinAlgError:
            continue
        roots = np.roots(np.concatenate(([1.0], coefficients)))
        roots = roots[np.imag(roots) >= 0]
        angles = np.arctan2(np.imag(roots), np.real(roots))
        frequencies = angles * SAMPLE_RATE / (2 * np.pi)
        bandwidths = -0.5 * SAMPLE_RATE / np.pi * np.log(np.maximum(np.abs(roots), 1e-12))
        valid = sorted(float(freq) for freq, bandwidth in zip(frequencies, bandwidths)
                       if 90 < freq < 5_000 and 0 < bandwidth < 900)
        if len(valid) >= 2:
            frame_formants.append(valid[:2])
    if not frame_formants:
        return [None, None, None]
    first_two = [float(value) for value in np.median(np.asarray(frame_formants), axis=0)]
    return [*first_two, None]


def compare_audio(reference: np.ndarray, candidate: np.ndarray) -> dict:
    reference_rms = rms(reference)
    candidate_rms = rms(candidate)
    gain_difference = 20 * np.log10((candidate_rms + 1e-12) / (reference_rms + 1e-12))
    reference, candidate, lag = align(reference / reference_rms, candidate / candidate_rms)
    waveform_correlation = float(abs(np.corrcoef(reference, candidate)[0, 1]))
    spectrum = spectral_metrics(reference, candidate)

    envelope_ref = signal.savgol_filter(np.abs(signal.hilbert(reference)), 1001, 2)
    envelope_cur = signal.savgol_filter(np.abs(signal.hilbert(candidate)), 1001, 2)
    envelope_error = rms(envelope_ref - envelope_cur) / max(rms(envelope_ref), 1e-12)
    envelope_similarity = float(np.exp(-envelope_error))

    original_formants = estimate_formants(reference)
    current_formants = estimate_formants(candidate)
    pairs = [(a, b) for a, b in zip(original_formants, current_formants) if a and b]
    relative_formant_error = float(np.mean([abs(a - b) / a for a, b in pairs])) if pairs else None
    spectral_similarity = float(np.exp(-spectrum["log_spectral_mae_db"] / 12))
    score = 100 * (
        0.50 * spectral_similarity
        + 0.20 * waveform_correlation
        + 0.15 * envelope_similarity
        + 0.15 * max(0.0, spectrum["magnitude_cosine"])
    )
    return {
        "similarity_score": float(np.clip(score, 0, 100)),
        "waveform_correlation": waveform_correlation,
        "alignment_lag_samples": lag,
        "gain_difference_db": float(gain_difference),
        "envelope_similarity": envelope_similarity,
        "original_formants_hz": original_formants,
        "current_formants_hz": current_formants,
        "mean_relative_formant_error": relative_formant_error,
        **spectrum,
    }


def round_numbers(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, list):
        return [round_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: round_numbers(item) for key, item in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True, help="Downloaded original Pink Trombone HTML")
    parser.add_argument("--artifacts", type=Path, help="Write JSON results and representative WAV pairs")
    args = parser.parse_args()
    if not args.original.is_file():
        parser.error(f"Original HTML not found: {args.original}")

    results = []
    cases = build_cases()
    for index, parameters in enumerate(cases, 1):
        print(f"[{index:02d}/{len(cases)}] {parameters['name']}", file=sys.stderr, flush=True)
        encoded = encode_parameters(parameters)
        original = render(["node", "comparison/render-original.mjs", str(args.original), encoded])
        current = render(["node", "comparison/render-current.mjs", encoded])
        results.append({
            "name": parameters["name"],
            "parameters": {key: value for key, value in parameters.items()
                           if key not in {"name", "warmupSeconds", "durationSeconds"}},
            "metrics": compare_audio(original, current),
        })
        if args.artifacts and parameters["name"] in {"neutral", "bright_i", "seeded_05", "seeded_06"}:
            audio_directory = args.artifacts / "audio"
            audio_directory.mkdir(parents=True, exist_ok=True)
            write_wav(audio_directory / f"{parameters['name']}-original.wav", original)
            write_wav(audio_directory / f"{parameters['name']}-current.wav", current)

    scores = np.asarray([case["metrics"]["similarity_score"] for case in results])
    spectral_errors = np.asarray([case["metrics"]["log_spectral_mae_db"] for case in results])
    formant_errors = np.asarray([
        case["metrics"]["mean_relative_formant_error"]
        for case in results
        if case["metrics"]["mean_relative_formant_error"] is not None
    ])
    waveform_correlations = np.asarray([case["metrics"]["waveform_correlation"] for case in results])
    envelope_similarities = np.asarray([case["metrics"]["envelope_similarity"] for case in results])
    magnitude_cosines = np.asarray([case["metrics"]["magnitude_cosine"] for case in results])
    gain_differences = np.asarray([case["metrics"]["gain_difference_db"] for case in results])
    payload = {
        "method": {
            "sample_rate_hz": SAMPLE_RATE,
            "warmup_seconds": WARMUP_SECONDS,
            "analyzed_seconds": DURATION_SECONDS,
            "case_count": len(results),
            "noise_and_modulation_disabled": True,
            "seed": 20260813,
        },
        "summary": {
            "mean_similarity_score": float(np.mean(scores)),
            "median_similarity_score": float(np.median(scores)),
            "minimum_similarity_score": float(np.min(scores)),
            "maximum_similarity_score": float(np.max(scores)),
            "mean_log_spectral_mae_db": float(np.mean(spectral_errors)),
            "mean_waveform_correlation": float(np.mean(waveform_correlations)),
            "mean_envelope_similarity": float(np.mean(envelope_similarities)),
            "mean_magnitude_cosine": float(np.mean(magnitude_cosines)),
            "median_absolute_gain_difference_db": float(np.median(np.abs(gain_differences))),
            "mean_relative_formant_error_when_available": (
                float(np.mean(formant_errors)) if formant_errors.size else None
            ),
            "formant_estimate_coverage": float(formant_errors.size / len(results)),
            "best_case": results[int(np.argmax(scores))]["name"],
            "worst_case": results[int(np.argmin(scores))]["name"],
        },
        "cases": results,
    }
    serialized = json.dumps(round_numbers(payload), indent=2) + "\n"
    if args.artifacts:
        args.artifacts.mkdir(parents=True, exist_ok=True)
        (args.artifacts / "results.json").write_text(serialized, encoding="utf8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
