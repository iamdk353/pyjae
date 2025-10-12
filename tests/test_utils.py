"""
Unit tests for JAE utilities.
"""

import pytest
import numpy as np
import torch

from jae.utils import (
    get_device,
    get_gpu_memory_info,
    generate_simulated_data,
    calculate_snr,
    calculate_r2_score,
    calculate_correlation,
    run_pca_baseline,
    validate_input_data,
    set_seed
)


class TestGetDevice:
    """Tests for device detection."""

    def test_cpu_device(self):
        """Test CPU device selection."""
        device = get_device(use_gpu=False, verbose=False)
        assert device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test CUDA device selection when available."""
        device = get_device(use_gpu=True, verbose=False)
        assert device.type == 'cuda'

    def test_specific_device_id(self):
        """Test specific device ID."""
        if torch.cuda.is_available():
            device = get_device(use_gpu=True, device_id=0, verbose=False)
            assert 'cuda:0' in str(device)

    def test_verbose_output(self, capsys):
        """Test verbose output."""
        device = get_device(use_gpu=False, verbose=True)
        captured = capsys.readouterr()
        assert 'device' in captured.out.lower()


class TestGetGPUMemoryInfo:
    """Tests for GPU memory information."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_info(self):
        """Test GPU memory info retrieval."""
        info = get_gpu_memory_info()

        assert info is not None
        assert 'allocated' in info
        assert 'reserved' in info
        assert 'total' in info
        assert info['total'] > 0

    @pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available")
    def test_no_gpu_returns_none(self):
        """Test that None is returned when no GPU available."""
        info = get_gpu_memory_info()
        assert info is None


class TestGenerateSimulatedData:
    """Tests for simulated data generation."""

    def test_default_generation(self):
        """Test default data generation."""
        clean, noisy = generate_simulated_data(n_samples=100, seed=42)

        assert clean.shape == (100, 96, 128)
        assert noisy.shape == (100, 96, 128)
        assert isinstance(clean, torch.Tensor)
        assert isinstance(noisy, torch.Tensor)

    def test_custom_dimensions(self):
        """Test generation with custom dimensions."""
        clean, noisy = generate_simulated_data(
            n_samples=50,
            seq_len=256,
            n_channels=64,
            latent_dim=4,
            seed=42
        )

        assert clean.shape == (50, 64, 256)
        assert noisy.shape == (50, 64, 256)

    def test_non_negativity(self):
        """Test that noisy data is non-negative."""
        clean, noisy = generate_simulated_data(n_samples=50, seed=42)

        # Noisy data should be non-negative (clipped)
        assert (noisy >= 0).all()

    def test_noise_level_effect(self):
        """Test that noise level affects SNR."""
        clean, noisy_low = generate_simulated_data(n_samples=50, noise_level=0.1, seed=42)
        clean, noisy_high = generate_simulated_data(n_samples=50, noise_level=0.9, seed=42)

        snr_low = calculate_snr(clean, noisy_low)
        snr_high = calculate_snr(clean, noisy_high)

        # Lower noise should give higher SNR
        assert snr_low > snr_high

    def test_reproducibility(self):
        """Test that same seed produces same data."""
        clean1, noisy1 = generate_simulated_data(n_samples=50, seed=42)
        clean2, noisy2 = generate_simulated_data(n_samples=50, seed=42)

        torch.testing.assert_close(clean1, clean2)
        torch.testing.assert_close(noisy1, noisy2)


class TestCalculateSNR:
    """Tests for SNR calculation."""

    def test_identical_signals(self):
        """Test SNR for identical signals."""
        signal = torch.randn(10, 96, 128)
        snr = calculate_snr(signal, signal)

        assert snr == float('inf')

    def test_noisy_signal(self):
        """Test SNR for noisy signal."""
        signal = torch.randn(10, 96, 128)
        noisy = signal + 0.1 * torch.randn_like(signal)

        snr = calculate_snr(signal, noisy)

        assert snr > 0
        assert snr < float('inf')

    def test_high_noise(self):
        """Test SNR decreases with more noise."""
        signal = torch.randn(10, 96, 128)
        noisy_low = signal + 0.1 * torch.randn_like(signal)
        noisy_high = signal + 1.0 * torch.randn_like(signal)

        snr_low = calculate_snr(signal, noisy_low)
        snr_high = calculate_snr(signal, noisy_high)

        assert snr_low > snr_high


class TestCalculateR2Score:
    """Tests for R² score calculation."""

    def test_perfect_prediction(self):
        """Test R² for perfect prediction."""
        y_true = torch.randn(10, 96, 128)
        r2 = calculate_r2_score(y_true, y_true)

        assert abs(r2 - 1.0) < 1e-6

    def test_mean_prediction(self):
        """Test R² for mean prediction."""
        y_true = torch.randn(100, 96, 128)
        y_pred = torch.full_like(y_true, y_true.mean())

        r2 = calculate_r2_score(y_true, y_pred)

        # Should be close to 0 for mean prediction
        assert abs(r2) < 0.1

    def test_numpy_array_input(self):
        """Test R² with numpy arrays."""
        y_true = np.random.randn(10, 96, 128)
        y_pred = y_true + 0.1 * np.random.randn(10, 96, 128)

        r2 = calculate_r2_score(y_true, y_pred)

        assert 0 < r2 < 1

    def test_poor_prediction(self):
        """Test R² for poor prediction."""
        y_true = torch.randn(10, 96, 128)
        y_pred = torch.randn(10, 96, 128)  # Random prediction

        r2 = calculate_r2_score(y_true, y_pred)

        # Random prediction should have R² around 0 or negative
        assert r2 < 0.5


class TestCalculateCorrelation:
    """Tests for correlation calculation."""

    def test_perfect_correlation(self):
        """Test correlation for identical signals."""
        signal = torch.randn(10, 96, 128)
        corr = calculate_correlation(signal, signal)

        assert abs(corr - 1.0) < 1e-6

    def test_positive_correlation(self):
        """Test positive correlation."""
        signal = torch.randn(10, 96, 128)
        similar = signal + 0.1 * torch.randn_like(signal)

        corr = calculate_correlation(signal, similar)

        assert 0.9 < corr < 1.0

    def test_numpy_array_input(self):
        """Test correlation with numpy arrays."""
        signal = np.random.randn(10, 96, 128)
        similar = signal + 0.1 * np.random.randn(10, 96, 128)

        corr = calculate_correlation(signal, similar)

        assert 0 < corr < 1


class TestRunPCABaseline:
    """Tests for PCA baseline."""

    def test_pca_denoising(self):
        """Test PCA denoising basic functionality."""
        noisy_data = torch.randn(50, 96, 128)
        denoised = run_pca_baseline(noisy_data, latent_dim=8)

        assert denoised.shape == noisy_data.shape
        assert isinstance(denoised, torch.Tensor)

    def test_pca_non_negativity(self):
        """Test that PCA output is non-negative."""
        noisy_data = torch.abs(torch.randn(50, 96, 128))
        denoised = run_pca_baseline(noisy_data, latent_dim=8)

        assert (denoised >= 0).all()

    def test_pca_different_latent_dims(self):
        """Test PCA with different latent dimensions."""
        noisy_data = torch.randn(50, 96, 128)

        for latent_dim in [4, 8, 16]:
            denoised = run_pca_baseline(noisy_data, latent_dim=latent_dim)
            assert denoised.shape == noisy_data.shape


class TestValidateInputData:
    """Tests for input data validation."""

    def test_valid_numpy_array(self):
        """Test validation of valid numpy array."""
        data = np.random.randn(10, 96, 128)
        validated = validate_input_data(data)

        assert isinstance(validated, torch.Tensor)
        assert validated.shape == (10, 96, 128)

    def test_valid_torch_tensor(self):
        """Test validation of valid torch tensor."""
        data = torch.randn(10, 96, 128)
        validated = validate_input_data(data)

        assert isinstance(validated, torch.Tensor)
        assert validated.shape == (10, 96, 128)

    def test_wrong_dimensions(self):
        """Test that wrong dimensions raise error."""
        data = np.random.randn(10, 96)  # 2D instead of 3D

        with pytest.raises(ValueError, match="3-dimensional"):
            validate_input_data(data)

    def test_nan_values(self):
        """Test that NaN values raise error."""
        data = np.random.randn(10, 96, 128)
        data[0, 0, 0] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            validate_input_data(data)

    def test_inf_values(self):
        """Test that infinite values raise error."""
        data = np.random.randn(10, 96, 128)
        data[0, 0, 0] = np.inf

        with pytest.raises(ValueError, match="infinite"):
            validate_input_data(data)

    def test_wrong_type(self):
        """Test that wrong type raises error."""
        data = [[1, 2, 3]]  # List instead of array/tensor

        with pytest.raises(TypeError):
            validate_input_data(data)


class TestSetSeed:
    """Tests for seed setting."""

    def test_numpy_reproducibility(self):
        """Test numpy reproducibility."""
        set_seed(42)
        a = np.random.randn(10)

        set_seed(42)
        b = np.random.randn(10)

        np.testing.assert_array_equal(a, b)

    def test_torch_reproducibility(self):
        """Test torch reproducibility."""
        set_seed(42)
        a = torch.randn(10)

        set_seed(42)
        b = torch.randn(10)

        torch.testing.assert_close(a, b)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_reproducibility(self):
        """Test CUDA reproducibility."""
        set_seed(42)
        a = torch.randn(10, device='cuda')

        set_seed(42)
        b = torch.randn(10, device='cuda')

        torch.testing.assert_close(a, b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

