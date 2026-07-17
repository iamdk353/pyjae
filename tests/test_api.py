"""Unit tests for the high-level JAE facade."""

import os
import tempfile

import numpy as np
import pytest

from jae import JAE
from jae.data import simulate_neural_data


@pytest.fixture
def small_data():
    """Small synthetic dataset with real low-rank structure."""
    _, noisy, _ = simulate_neural_data(
        n_samples=48, n_channels=32, n_timepoints=64, latent_dim=4, snr_db=5.0, seed=0
    )
    return noisy


class TestJAEInitialization:
    def test_default_backend(self):
        model = JAE(verbose=False)
        assert model.backend == "jae1"
        assert not model.is_fitted

    def test_jepa_backend(self):
        model = JAE(backend="jepa", verbose=False)
        assert model.backend == "jepa"

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="backend"):
            JAE(backend="nonsense", verbose=False)

    def test_device_specification(self):
        model = JAE(device="cpu", verbose=False)
        assert model.device.type == "cpu"


class TestJAE1Backend:
    def test_fit_denoise(self, small_data):
        model = JAE(latent_dim=4, backend="jae1", device="cpu", verbose=False)
        model.fit(small_data, epochs=3, batch_size=8)
        assert model.is_fitted
        assert model.input_channels_ == 32
        denoised = model.denoise(small_data)
        assert denoised.shape == tuple(small_data.shape)
        assert isinstance(denoised, np.ndarray)

    def test_auto_latent_dim(self, small_data):
        model = JAE(backend="jae1", device="cpu", verbose=False)
        model.fit(small_data, epochs=2, batch_size=8)
        assert model.latent_dim == max(32 // 8, 4)

    def test_odd_channels_ok(self):
        data = np.random.randn(40, 31, 64).astype(np.float32)
        model = JAE(latent_dim=4, backend="jae1", device="cpu", verbose=False)
        model.fit(data, epochs=2, batch_size=8)
        assert model.is_fitted

    def test_training_history(self, small_data):
        model = JAE(latent_dim=4, backend="jae1", device="cpu", verbose=False)
        model.fit(small_data, epochs=4, batch_size=8)
        assert len(model.get_training_history()["train_loss"]) == 4

    def test_loss_decreases(self, small_data):
        model = JAE(latent_dim=4, backend="jae1", device="cpu", verbose=False)
        model.fit(small_data, epochs=40, batch_size=8)
        hist = model.get_training_history()["train_loss"]
        assert hist[-1] < hist[0]


class TestJEPABackend:
    def test_fit_denoise(self, small_data):
        model = JAE(
            latent_dim=16, backend="jepa", device="cpu", verbose=False, patch_len=8, d_model=32
        )
        model.fit(small_data, epochs=3, batch_size=8)
        assert model.is_fitted
        denoised = model.denoise(small_data)
        assert denoised.shape == tuple(small_data.shape)


class TestErrors:
    def test_denoise_before_fit(self, small_data):
        model = JAE(latent_dim=4, device="cpu", verbose=False)
        with pytest.raises(RuntimeError, match="fitted"):
            model.denoise(small_data)

    def test_denoise_wrong_channels(self, small_data):
        model = JAE(latent_dim=4, device="cpu", verbose=False)
        model.fit(small_data, epochs=2, batch_size=8)
        with pytest.raises(ValueError, match="channels"):
            model.denoise(np.random.randn(4, 16, 64).astype(np.float32))

    def test_fit_invalid_shape(self):
        model = JAE(latent_dim=4, device="cpu", verbose=False)
        with pytest.raises(ValueError, match="3D"):
            model.fit(np.random.randn(10, 32).astype(np.float32), epochs=1)

    def test_fit_with_nan(self):
        data = np.random.randn(20, 32, 64).astype(np.float32)
        data[0, 0, 0] = np.nan
        model = JAE(latent_dim=4, device="cpu", verbose=False)
        with pytest.raises(ValueError, match="NaN"):
            model.fit(data, epochs=1)


class TestScore:
    def test_score_perfect(self):
        model = JAE(verbose=False)
        signal = np.random.randn(8, 32, 64)
        assert abs(model.score(signal, signal) - 1.0) < 1e-6

    def test_score_per_channel(self):
        model = JAE(verbose=False)
        signal = np.random.randn(8, 32, 64)
        report = model.score(signal, signal, per_channel=True)
        assert "per_channel" in report
        assert len(report["per_channel"]) == 32


class TestSaveLoad:
    def test_save_and_load(self, small_data):
        model = JAE(latent_dim=4, backend="jae1", device="cpu", verbose=False)
        model.fit(small_data, epochs=3, batch_size=8)
        before = model.denoise(small_data)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pth")
            model.save(path)
            loaded = JAE(device="cpu", verbose=False)
            loaded.load(path)
            after = loaded.denoise(small_data)
            np.testing.assert_allclose(before, after, rtol=1e-4, atol=1e-5)

    def test_save_unfitted(self):
        model = JAE(verbose=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="unfitted"):
                model.save(os.path.join(tmpdir, "m.pth"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
