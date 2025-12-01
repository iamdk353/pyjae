"""
Unit tests for JAE API.
"""

import pytest
import numpy as np
import torch
import tempfile
import os

from jae import JAE


class TestJAEInitialization:
    """Tests for JAE initialization."""

    def test_default_initialization(self):
        """Test default initialization uses original paper settings."""
        model = JAE(verbose=False)
        assert model.num_networks == 2
        assert model.subsample_fraction == 0.5
        assert not model.use_unet
        assert not model.use_vicreg
        assert not model.is_fitted

    def test_custom_latent_dim(self):
        """Test custom latent dimension."""
        model = JAE(latent_dim=12, verbose=False)
        assert model.latent_dim == 12

    def test_enhanced_mode(self):
        """Test enhanced mode activation."""
        # U-Net enables enhanced mode
        model = JAE(use_unet=True, verbose=False)
        assert model._use_enhanced_model()

        # VICReg enables enhanced mode
        model = JAE(use_vicreg=True, verbose=False)
        assert model._use_enhanced_model()

        # More than 2 networks enables enhanced mode
        model = JAE(num_networks=5, verbose=False)
        assert model._use_enhanced_model()

    def test_device_specification(self):
        """Test device specification."""
        model = JAE(device='cpu', verbose=False)
        assert model.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_detection(self):
        """Test GPU detection."""
        model = JAE(use_gpu=True, verbose=False)
        assert model.device.type == 'cuda'


class TestJAEFit:
    """Tests for JAE fit method."""

    @pytest.fixture
    def small_data(self):
        """Small synthetic dataset."""
        return np.random.randn(50, 96, 64).astype(np.float32)

    def test_fit_basic(self, small_data):
        """Test basic fit."""
        model = JAE(latent_dim=8, verbose=False)
        model.fit(small_data, epochs=2, batch_size=8)
        assert model.is_fitted
        assert model.input_channels_ == 96

    def test_fit_with_numpy(self, small_data):
        """Test fit with numpy array."""
        model = JAE(latent_dim=8, verbose=False)
        model.fit(small_data, epochs=2, batch_size=8)
        assert model.is_fitted

    def test_fit_with_tensor(self):
        """Test fit with torch tensor."""
        data = torch.randn(50, 96, 64)
        model = JAE(latent_dim=8, verbose=False)
        model.fit(data, epochs=2, batch_size=8)
        assert model.is_fitted

    def test_fit_auto_latent_dim(self, small_data):
        """Test auto latent dimension detection."""
        model = JAE(verbose=False)
        model.fit(small_data, epochs=2, batch_size=8)
        assert model.latent_dim == 12  # 96 // 8

    def test_fit_odd_channels_original_mode(self):
        """Test that odd channels raise error in original mode."""
        data = np.random.randn(50, 97, 64).astype(np.float32)
        model = JAE(latent_dim=8, verbose=False)
        with pytest.raises(ValueError, match="even number"):
            model.fit(data, epochs=2)

    def test_fit_odd_channels_enhanced_mode(self):
        """Test that odd channels work in enhanced mode."""
        data = np.random.randn(50, 97, 64).astype(np.float32)
        model = JAE(latent_dim=8, num_networks=3, verbose=False)
        model.fit(data, epochs=2, batch_size=8)
        assert model.is_fitted

    def test_fit_with_validation(self, small_data):
        """Test fit with validation split."""
        model = JAE(latent_dim=8, verbose=False)
        model.fit(small_data, epochs=2, batch_size=8, validation_split=0.2)
        assert model.is_fitted
        assert 'val_loss' in model.training_history_

    def test_fit_returns_self(self, small_data):
        """Test that fit returns self."""
        model = JAE(latent_dim=8, verbose=False)
        result = model.fit(small_data, epochs=2, batch_size=8)
        assert result is model

    def test_training_history(self, small_data):
        """Test training history recording."""
        model = JAE(latent_dim=8, verbose=False)
        model.fit(small_data, epochs=5, batch_size=8)
        history = model.get_training_history()
        assert len(history['train_loss']) == 5

    def test_fit_invalid_shape(self):
        """Test invalid data shape."""
        data = np.random.randn(50, 96)  # 2D
        model = JAE(latent_dim=8, verbose=False)
        with pytest.raises(ValueError, match="3D"):
            model.fit(data, epochs=2)

    def test_fit_with_nan(self):
        """Test NaN detection."""
        data = np.random.randn(50, 96, 64).astype(np.float32)
        data[0, 0, 0] = np.nan
        model = JAE(latent_dim=8, verbose=False)
        with pytest.raises(ValueError, match="NaN"):
            model.fit(data, epochs=2)


class TestJAEDenoise:
    """Tests for denoise method."""

    @pytest.fixture
    def fitted_model(self):
        """Fitted model fixture."""
        data = np.random.randn(50, 96, 64).astype(np.float32)
        model = JAE(latent_dim=8, verbose=False)
        model.fit(data, epochs=2, batch_size=8)
        return model

    def test_denoise_basic(self, fitted_model):
        """Test basic denoising."""
        data = np.random.randn(20, 96, 64).astype(np.float32)
        denoised = fitted_model.denoise(data)
        assert isinstance(denoised, np.ndarray)
        assert denoised.shape == (20, 96, 64)

    def test_denoise_before_fit(self):
        """Test denoise before fit raises error."""
        model = JAE(latent_dim=8, verbose=False)
        data = np.random.randn(20, 96, 64).astype(np.float32)
        with pytest.raises(RuntimeError, match="fitted"):
            model.denoise(data)

    def test_denoise_wrong_channels(self, fitted_model):
        """Test wrong channel count."""
        data = np.random.randn(20, 48, 64).astype(np.float32)
        with pytest.raises(ValueError, match="channels"):
            fitted_model.denoise(data)

    def test_denoise_single_sample(self, fitted_model):
        """Test single sample denoising."""
        data = np.random.randn(1, 96, 64).astype(np.float32)
        denoised = fitted_model.denoise(data)
        assert denoised.shape == (1, 96, 64)


class TestJAEScore:
    """Tests for score method (VAF)."""

    def test_score_perfect(self):
        """Test VAF = 1 for identical signals."""
        model = JAE(verbose=False)
        signal = np.random.randn(10, 96, 64)
        vaf = model.score(signal, signal)
        assert abs(vaf - 1.0) < 1e-6

    def test_score_range(self):
        """Test VAF is in valid range."""
        model = JAE(verbose=False)
        signal = np.random.randn(10, 96, 64)
        noisy = signal + 0.5 * np.random.randn(*signal.shape)
        vaf = model.score(signal, noisy)
        assert 0 < vaf < 1


class TestJAEFitDenoise:
    """Tests for fit_denoise method."""

    def test_fit_denoise(self):
        """Test fit_denoise convenience method."""
        data = np.random.randn(50, 96, 64).astype(np.float32)
        model = JAE(latent_dim=8, verbose=False)
        denoised = model.fit_denoise(data, epochs=2, batch_size=8)
        assert model.is_fitted
        assert denoised.shape == data.shape


class TestJAESaveLoad:
    """Tests for save/load."""

    @pytest.fixture
    def fitted_model(self):
        """Fitted model fixture."""
        data = np.random.randn(50, 96, 64).astype(np.float32)
        model = JAE(latent_dim=8, verbose=False)
        model.fit(data, epochs=2, batch_size=8)
        return model

    def test_save_and_load(self, fitted_model):
        """Test save and load."""
        test_data = np.random.randn(10, 96, 64).astype(np.float32)
        denoised_before = fitted_model.denoise(test_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pth')
            fitted_model.save(path)

            new_model = JAE(verbose=False)
            new_model.load(path)
            denoised_after = new_model.denoise(test_data)

            np.testing.assert_allclose(denoised_before, denoised_after, rtol=1e-5)

    def test_save_unfitted(self):
        """Test saving unfitted model."""
        model = JAE(verbose=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pth')
            with pytest.raises(RuntimeError, match="unfitted"):
                model.save(path)


class TestJAEEnhancedMode:
    """Tests for enhanced mode (U-Net, VICReg)."""

    def test_unet_mode(self):
        """Test U-Net architecture."""
        data = np.random.randn(50, 96, 64).astype(np.float32)
        model = JAE(latent_dim=8, use_unet=True, num_networks=3, verbose=False)
        model.fit(data, epochs=2, batch_size=8)
        denoised = model.denoise(data)
        assert denoised.shape == data.shape

    def test_vicreg_mode(self):
        """Test VICReg loss."""
        data = np.random.randn(50, 96, 64).astype(np.float32)
        model = JAE(latent_dim=8, use_vicreg=True, num_networks=3, verbose=False)
        model.fit(data, epochs=2, batch_size=8)
        assert model.is_fitted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
