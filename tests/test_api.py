"""
Unit tests for high-level JAE API.
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
        """Test default initialization."""
        model = JAE(verbose=False)

        assert model.model_type == 'jae2'
        assert not model.is_fitted
        assert model.latent_dim is None  # Not set until fit

    def test_jae1_initialization(self):
        """Test JAE1 initialization."""
        model = JAE(model_type='jae1', latent_dim=12, verbose=False)

        assert model.model_type == 'jae1'
        assert model.latent_dim == 12

    def test_jae2_initialization(self):
        """Test JAE2 initialization with custom parameters."""
        model = JAE(
            model_type='jae2',
            latent_dim=16,
            num_networks=8,
            subsample_fraction=0.7,
            verbose=False
        )

        assert model.model_type == 'jae2'
        assert model.latent_dim == 16
        assert model.config.num_networks == 8
        assert model.config.subsample_fraction == 0.7

    def test_invalid_model_type(self):
        """Test that invalid model type raises error."""
        with pytest.raises(ValueError, match="model_type"):
            JAE(model_type='invalid', verbose=False)

    def test_device_specification(self):
        """Test device specification."""
        model = JAE(device='cpu', verbose=False)
        assert model.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_detection(self):
        """Test GPU detection when available."""
        model = JAE(use_gpu=True, verbose=False)
        assert model.device.type == 'cuda'


class TestJAEFit:
    """Tests for JAE fit method."""

    @pytest.fixture
    def small_data(self):
        """Generate small synthetic dataset for testing."""
        return np.random.randn(50, 96, 64).astype(np.float32)

    def test_fit_basic(self, small_data):
        """Test basic fit functionality."""
        model = JAE(latent_dim=8, verbose=False)
        model.fit(small_data, epochs=2, batch_size=8)

        assert model.is_fitted
        assert model.input_channels_ == 96
        assert model.model is not None

    def test_fit_with_numpy_array(self, small_data):
        """Test fit with numpy array input."""
        model = JAE(latent_dim=8, verbose=False)
        model.fit(small_data, epochs=2, batch_size=8)

        assert model.is_fitted

    def test_fit_with_torch_tensor(self):
        """Test fit with torch tensor input."""
        data = torch.randn(50, 96, 64)
        model = JAE(latent_dim=8, verbose=False)
        model.fit(data, epochs=2, batch_size=8)

        assert model.is_fitted

    def test_fit_auto_detect_latent_dim(self, small_data):
        """Test automatic latent dimension detection."""
        model = JAE(verbose=False)  # No latent_dim specified
        model.fit(small_data, epochs=2, batch_size=8)

        # Should auto-detect as 96 // 8 = 12
        assert model.latent_dim == 12

    def test_fit_jae1_odd_channels_raises_error(self):
        """Test that JAE1 with odd channels raises error."""
        data = np.random.randn(50, 97, 64).astype(np.float32)
        model = JAE(model_type='jae1', latent_dim=8, verbose=False)

        with pytest.raises(ValueError, match="even number"):
            model.fit(data, epochs=2)

    def test_fit_with_validation_split(self, small_data):
        """Test fit with validation split."""
        model = JAE(latent_dim=8, verbose=False)
        model.fit(small_data, epochs=2, batch_size=8, validation_split=0.2)

        assert model.is_fitted
        assert 'val_loss' in model.training_history_

    def test_fit_returns_self(self, small_data):
        """Test that fit returns self for method chaining."""
        model = JAE(latent_dim=8, verbose=False)
        result = model.fit(small_data, epochs=2, batch_size=8)

        assert result is model

    def test_training_history_recorded(self, small_data):
        """Test that training history is recorded."""
        model = JAE(latent_dim=8, verbose=False)
        model.fit(small_data, epochs=5, batch_size=8)

        history = model.get_training_history()
        assert 'train_loss' in history
        assert len(history['train_loss']) == 5

    def test_fit_invalid_data_shape(self):
        """Test that invalid data shape raises error."""
        data = np.random.randn(50, 96)  # Only 2D
        model = JAE(latent_dim=8, verbose=False)

        with pytest.raises(ValueError, match="3-dimensional"):
            model.fit(data, epochs=2)

    def test_fit_with_nan_raises_error(self):
        """Test that NaN in data raises error."""
        data = np.random.randn(50, 96, 64)
        data[0, 0, 0] = np.nan
        model = JAE(latent_dim=8, verbose=False)

        with pytest.raises(ValueError, match="NaN"):
            model.fit(data, epochs=2)


class TestJAEDenoise:
    """Tests for JAE denoise method."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model for testing."""
        data = np.random.randn(50, 96, 64).astype(np.float32)
        model = JAE(latent_dim=8, verbose=False)
        model.fit(data, epochs=2, batch_size=8)
        return model

    def test_denoise_basic(self, fitted_model):
        """Test basic denoise functionality."""
        data = np.random.randn(20, 96, 64).astype(np.float32)
        denoised = fitted_model.denoise(data)

        assert isinstance(denoised, np.ndarray)
        assert denoised.shape == (20, 96, 64)

    def test_denoise_before_fit_raises_error(self):
        """Test that denoise before fit raises error."""
        model = JAE(latent_dim=8, verbose=False)
        data = np.random.randn(20, 96, 64).astype(np.float32)

        with pytest.raises(RuntimeError, match="fitted"):
            model.denoise(data)

    def test_denoise_wrong_channels_raises_error(self, fitted_model):
        """Test that wrong number of channels raises error."""
        data = np.random.randn(20, 48, 64).astype(np.float32)  # Wrong channels

        with pytest.raises(ValueError, match="channels"):
            fitted_model.denoise(data)

    def test_denoise_single_sample(self, fitted_model):
        """Test denoise with single sample."""
        data = np.random.randn(1, 96, 64).astype(np.float32)
        denoised = fitted_model.denoise(data)

        assert denoised.shape == (1, 96, 64)

    def test_denoise_different_batch_sizes(self, fitted_model):
        """Test denoise with different batch sizes."""
        for n_samples in [1, 10, 50]:
            data = np.random.randn(n_samples, 96, 64).astype(np.float32)
            denoised = fitted_model.denoise(data)
            assert denoised.shape == (n_samples, 96, 64)


class TestJAEFitDenoise:
    """Tests for JAE fit_denoise convenience method."""

    def test_fit_denoise(self):
        """Test fit_denoise convenience method."""
        data = np.random.randn(50, 96, 64).astype(np.float32)
        model = JAE(latent_dim=8, verbose=False)

        denoised = model.fit_denoise(data, epochs=2, batch_size=8)

        assert model.is_fitted
        assert isinstance(denoised, np.ndarray)
        assert denoised.shape == data.shape


class TestJAESaveLoad:
    """Tests for JAE save/load functionality."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model for testing."""
        data = np.random.randn(50, 96, 64).astype(np.float32)
        model = JAE(latent_dim=8, verbose=False)
        model.fit(data, epochs=2, batch_size=8)
        return model

    def test_save_and_load(self, fitted_model):
        """Test saving and loading model."""
        test_data = np.random.randn(10, 96, 64).astype(np.float32)
        denoised_before = fitted_model.denoise(test_data)

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'model.pth')
            fitted_model.save(save_path)

            # Load model
            new_model = JAE(verbose=False)
            new_model.load(save_path)

            # Test that loaded model produces same output
            denoised_after = new_model.denoise(test_data)

            np.testing.assert_allclose(denoised_before, denoised_after, rtol=1e-5)

    def test_save_unfitted_raises_error(self):
        """Test that saving unfitted model raises error."""
        model = JAE(latent_dim=8, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'model.pth')
            with pytest.raises(RuntimeError, match="unfitted"):
                model.save(save_path)


class TestJAEModelTypes:
    """Tests for different JAE model types."""

    def test_jae1_full_pipeline(self):
        """Test full pipeline with JAE1."""
        data = np.random.randn(50, 96, 64).astype(np.float32)

        model = JAE(model_type='jae1', latent_dim=8, verbose=False)
        model.fit(data, epochs=2, batch_size=8)
        denoised = model.denoise(data)

        assert denoised.shape == data.shape

    def test_jae2_full_pipeline(self):
        """Test full pipeline with JAE2."""
        data = np.random.randn(50, 96, 64).astype(np.float32)

        model = JAE(
            model_type='jae2',
            latent_dim=8,
            num_networks=3,
            verbose=False
        )
        model.fit(data, epochs=2, batch_size=8)
        denoised = model.denoise(data)

        assert denoised.shape == data.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

