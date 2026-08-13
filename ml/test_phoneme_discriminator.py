import unittest

import numpy as np

from ml.phoneme_discriminator import (
    FRAME_SIZE,
    SPECTRUM_SIZE,
    audio_to_log_periodogram,
    normalize_curve,
)


class AudioFeatureTests(unittest.TestCase):
    def test_log_periodogram_has_dataset_shape(self) -> None:
        time = np.arange(FRAME_SIZE) / 16_000
        signal = np.sin(2 * np.pi * 180 * time)
        curve = audio_to_log_periodogram(signal, 16_000)
        self.assertEqual(curve.shape, (SPECTRUM_SIZE,))
        self.assertAlmostEqual(float(curve.mean()), 0.0, places=7)
        self.assertAlmostEqual(float(curve.std()), 1.0, places=7)

    def test_resamples_arbitrary_browser_rate(self) -> None:
        signal = np.sin(2 * np.pi * 220 * np.arange(1_536) / 48_000)
        curve = audio_to_log_periodogram(signal, 48_000)
        self.assertEqual(curve.shape, (SPECTRUM_SIZE,))
        self.assertTrue(np.all(np.isfinite(curve)))

    def test_constant_curve_is_safe(self) -> None:
        self.assertTrue(np.all(normalize_curve(np.ones(256)) == 0))

    def test_empty_audio_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            audio_to_log_periodogram(np.array([]), 16_000)


if __name__ == "__main__":
    unittest.main()
