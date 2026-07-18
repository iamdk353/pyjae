"""Tests for pyjae.baselines: matched-capacity denoising baselines."""

import numpy as np
import torch
from sklearn.decomposition import PCA

from pyjae.baselines import (
    DenoisingAutoencoder,
    cv_select_k,
    factor_analysis_denoise,
    mean_predictor,
    pca_denoise,
    wiener_oracle,
)


def _lowrank_generator(seed: int, c: int = 10, t: int = 20, rank: int = 3, noise_std: float = 0.4):
    """Build a shared-subspace low-rank + noise data generator for tests.

    Returns a ``gen(b)`` closure producing (clean, noisy) torch tensors of
    shape (b, c, t) that all share one fixed mixing matrix and channel
    mean (only the latents and noise differ draw to draw), plus that
    mixing matrix and channel mean so tests can compute analytic
    covariances (e.g. for the Wiener oracle).

    Args:
        seed: Random seed.
        c: Number of channels.
        t: Number of timepoints.
        rank: True latent rank of the clean signal.
        noise_std: Per-channel iid Gaussian noise std.

    Returns:
        Tuple of (gen, w, channel_mean).
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((c, rank))
    channel_mean = rng.uniform(-1.0, 1.0, size=c)

    def gen(b: int):
        latents = rng.standard_normal((b, rank, t))
        clean = np.einsum("cr,brt->bct", w, latents) + channel_mean[None, :, None]
        noise = rng.standard_normal((b, c, t)) * noise_std
        noisy = clean + noise
        return (
            torch.tensor(clean, dtype=torch.float32),
            torch.tensor(noisy, dtype=torch.float32),
        )

    return gen, w, channel_mean


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean squared error between two same-shape tensors."""
    return float(torch.mean((a - b) ** 2))


class TestPCADenoise:
    """Tests for pca_denoise."""

    def test_reduces_noise_and_shape(self):
        gen, _, _ = _lowrank_generator(seed=1)
        clean_train, noisy_train = gen(60)
        clean_test, noisy_test = gen(30)

        denoised = pca_denoise(noisy_train, noisy_test, k=3)

        assert isinstance(denoised, torch.Tensor)
        assert denoised.dtype == torch.float32
        assert denoised.shape == noisy_test.shape

        mse_noisy = _mse(noisy_test, clean_test)
        mse_denoised = _mse(denoised, clean_test)
        assert mse_denoised < mse_noisy

    def test_stats_fit_on_train_only(self):
        """Train-derived mean/std/PCA basis must not depend on which test set is passed."""
        gen, _, _ = _lowrank_generator(seed=5)
        _, noisy_train = gen(60)
        _, noisy_test_a = gen(25)
        _, noisy_test_b = gen(25)
        k = 3

        # Independent reference implementation using ONLY train statistics.
        train_np = noisy_train.numpy()
        b_tr, c, t_tr = train_np.shape
        train_flat = train_np.transpose(0, 2, 1).reshape(b_tr * t_tr, c)
        mean = train_flat.mean(axis=0, keepdims=True)
        std = train_flat.std(axis=0, keepdims=True)
        pca_ref = PCA(n_components=k).fit((train_flat - mean) / std)

        for noisy_test in (noisy_test_a, noisy_test_b):
            denoised = pca_denoise(noisy_train, noisy_test, k)

            test_np = noisy_test.numpy()
            b_te, _, t_te = test_np.shape
            test_flat = test_np.transpose(0, 2, 1).reshape(b_te * t_te, c)
            test_z = (test_flat - mean) / std
            expected_z = pca_ref.inverse_transform(pca_ref.transform(test_z))
            expected = (expected_z * std + mean).reshape(b_te, t_te, c).transpose(0, 2, 1)

            assert np.allclose(denoised.numpy(), expected, atol=1e-4)


class TestFactorAnalysisDenoise:
    """Tests for factor_analysis_denoise."""

    def test_runs_and_reduces_noise(self):
        gen, _, _ = _lowrank_generator(seed=6)
        clean_train, noisy_train = gen(80)
        clean_test, noisy_test = gen(30)

        denoised = factor_analysis_denoise(noisy_train, noisy_test, k=3)

        assert isinstance(denoised, torch.Tensor)
        assert denoised.dtype == torch.float32
        assert denoised.shape == noisy_test.shape

        mse_noisy = _mse(noisy_test, clean_test)
        mse_denoised = _mse(denoised, clean_test)
        assert mse_denoised < mse_noisy


class TestMeanPredictor:
    """Tests for mean_predictor."""

    def test_shape_and_constant_over_time(self):
        gen, _, _ = _lowrank_generator(seed=7)
        _, noisy_test = gen(10)

        out = mean_predictor(noisy_test)

        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.float32
        assert out.shape == noisy_test.shape

        std_over_time = out.std(dim=2)
        assert torch.allclose(std_over_time, torch.zeros_like(std_over_time), atol=1e-5)

        expected_mean = noisy_test.mean(dim=2, keepdim=True).expand_as(noisy_test)
        assert torch.allclose(out, expected_mean, atol=1e-5)


class TestWienerOracle:
    """Tests for wiener_oracle."""

    def test_reduces_mse_with_known_covariances(self):
        gen, w, channel_mean = _lowrank_generator(seed=8, c=10, t=25, rank=3, noise_std=0.6)
        clean_test, noisy_test = gen(50)

        # True covariances by construction: clean = w @ latents (latents ~
        # N(0, I)) + channel_mean, so Cov(clean) = w @ w.T; noise is iid
        # per-channel Gaussian with std=noise_std, so Cov(noise) = noise_std^2 * I.
        signal_cov = w @ w.T
        noise_cov = np.eye(w.shape[0]) * 0.6**2

        denoised = wiener_oracle(noisy_test, signal_cov, noise_cov, channel_mean)

        assert isinstance(denoised, torch.Tensor)
        assert denoised.dtype == torch.float32
        assert denoised.shape == noisy_test.shape

        mse_noisy = _mse(noisy_test, clean_test)
        mse_denoised = _mse(denoised, clean_test)
        assert mse_denoised < mse_noisy

    def test_zero_noise_cov_is_identity(self):
        c = 6
        rng = np.random.default_rng(9)
        a = rng.standard_normal((c, c))
        signal_cov = a @ a.T + np.eye(c) * 0.1  # full-rank positive definite
        noise_cov = np.zeros((c, c))
        mean = rng.standard_normal(c)

        noisy_test = torch.tensor(rng.standard_normal((5, c, 8)), dtype=torch.float32)
        denoised = wiener_oracle(noisy_test, signal_cov, noise_cov, mean)

        assert torch.allclose(denoised, noisy_test, atol=1e-4)


class TestDenoisingAutoencoder:
    """Tests for DenoisingAutoencoder."""

    def test_fits_without_error_and_denoise_shape(self):
        gen, _, _ = _lowrank_generator(seed=10, c=8, t=12, rank=2, noise_std=0.3)
        clean_train, noisy_train = gen(40)
        _, noisy_test = gen(15)

        dae = DenoisingAutoencoder(n_channels=8, latent_dim=2, hidden=(16,))
        dae.fit(noisy_train, clean_train, epochs=5, batch_size=64)

        assert len(dae.loss_history_) == 5

        denoised = dae.denoise(noisy_test)

        assert isinstance(denoised, torch.Tensor)
        assert denoised.dtype == torch.float32
        assert denoised.shape == noisy_test.shape
        assert torch.isfinite(denoised).all()


class TestCVSelectK:
    """Tests for cv_select_k."""

    def test_returns_k_from_list_without_crashing(self):
        gen, _, _ = _lowrank_generator(seed=11, c=10, t=20, rank=3, noise_std=0.2)
        _, noisy_train = gen(150)
        ks = [1, 2, 3, 4, 6, 8]

        k = cv_select_k(noisy_train, ks, seed=0)

        assert k in ks
