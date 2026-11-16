"""Independent PSD oracle and fitted-PCA checkpoint regression tests."""
import io
import unittest
import numpy as np
import torch
from quantopo.ablations.summaries import PowerSpectrumSummary

class SummaryTests(unittest.TestCase):
    def test_fixed_grid_against_numpy_oracle(self):
        rng = np.random.default_rng(71)
        for n in [1, 2, 7, 16, 31, 32]:
            with self.subTest(n=n):
                x = rng.normal(size=(n, 5))
                phase = np.exp(-2j * np.pi * np.arange(1, 17)[:, None] * np.arange(n)[None, :] / 32)
                expected = np.log1p(np.abs(phase @ x)**2 / n).ravel()
                states = torch.tensor(x[None], dtype=torch.float64)
                actual = PowerSpectrumSummary(4).features(states, torch.ones(1, n, dtype=torch.bool))
                np.testing.assert_allclose(actual.numpy()[0], expected, rtol=1e-12, atol=1e-12)

    def test_full_length_matches_orthonormal_rfft(self):
        g = torch.Generator().manual_seed(17)
        x = torch.randn(3, 32, 5, generator=g, dtype=torch.float64)
        expected = torch.log1p(torch.fft.rfft(x, dim=1, norm="ortho")[:, 1:].abs().square()).flatten(1)
        actual = PowerSpectrumSummary(4).features(x, torch.ones(3, 32, dtype=torch.bool))
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_padding_does_not_change_short_sequence_spectrum(self):
        g = torch.Generator().manual_seed(19)
        x = torch.randn(1, 7, 5, generator=g, dtype=torch.float64)
        padded = torch.cat([x, torch.full((1, 25, 5), 1e9, dtype=x.dtype)], 1)
        s = PowerSpectrumSummary(4)
        actual = s.features(padded, torch.arange(32)[None] < 7)
        expected = s.features(x, torch.ones(1, 7, dtype=torch.bool))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_fitted_pca_checkpoint_restores_without_refitting(self):
        g = torch.Generator().manual_seed(23)
        train = torch.randn(20, 32, 5, generator=g)
        mask = torch.ones(20, 32, dtype=torch.bool)
        s = PowerSpectrumSummary(8)
        s.fit(train, mask, torch.randn(20, 8, generator=g))
        stream = io.BytesIO()
        torch.save(s.state_dict(), stream)
        stream.seek(0)
        restored = PowerSpectrumSummary(8)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        self.assertTrue(restored.pca.fitted)
        torch.testing.assert_close(restored(train, mask), s(train, mask), rtol=0, atol=0)
        torch.testing.assert_close(restored.pca.mean, s.pca.mean, rtol=0, atol=0)

    def test_unfitted_checkpoint_restores_and_still_requires_training_fit(self):
        s = PowerSpectrumSummary(8)
        s.load_state_dict(PowerSpectrumSummary(8).state_dict())
        with self.assertRaisesRegex(RuntimeError, "fit PCA"):
            s(torch.ones(1, 7, 5), torch.ones(1, 7, dtype=torch.bool))

    def test_incompatible_output_width_is_rejected(self):
        s = PowerSpectrumSummary(8)
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            s.load_state_dict(PowerSpectrumSummary(4).state_dict())

if __name__ == "__main__":
    unittest.main()
