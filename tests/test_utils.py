"""
Unit tests for JAE utilities.
"""

import pytest
import numpy as np
import torch

from jae import (
    simulate_neural_data,
    calculate_vaf,
    calculate_snr,
    run_pca_baseline,
    set_seed,
    get_device,
)
from jae.utils import validate_input_data


class TestSimulateNeuralData:
    """Tests for data simulation."""

    def test_basic_generation(self):
        """Test basic data generation."""
        clean, noisy, info = simulate_neural_data(
            n_samples=100,
            n_channels=96,
            n_timepoints=128,
            latent_dim=6,
            snr_db=5.0,
            seed=42,
        )

        assert clean.shape == (100, 96, 128)
        assert noisy.shape == (100, 96, 128)
        assert isinstance(clean, torch.Tensor)
        assert isinstance(noisy, torch.Tensor)

    def test_info_dict(self):
        """Test info dictionary contents."""
        _, _, info = simulate_neural_data(
            n_samples=50, latent_dim=6, snr_db=5.0, seed=42
        )

        assert 'snr_db' in info
        assert 'latent_dim' in info
        assert info['latent_dim'] == 6

    def test_nonlinear_mode(self):
        """Test nonlinear embedding."""
        clean_lin, _, _ = simulate_neural_data(
            n_samples=50, nonlinear=False, seed=42
        )
        clean_nonlin, _, _ = simulate_neural_data(
            n_samples=50, nonlinear=True, alpha=2.0, seed=42
        )

        # Nonlinear should produce different results
        assert not torch.allclose(clean_lin, clean_nonlin)

    def test_reproducibility(self):
        """Test seed reproducibility."""
        c1, n1, _ = simulate_neural_data(n_samples=50, seed=42)
        c2, n2, _ = simulate_neural_data(n_samples=50, seed=42)

        assert torch.allclose(c1, c2)
        assert torch.allclose(n1, n2)

    def test_non_negativity(self):
        """Test signals are non-negative after clipping."""
        _, noisy, _ = simulate_neural_data(n_samples=100, snr_db=0.0, seed=42)
        assert (noisy >= 0).all()

    def test_snr_range(self):
        """Test different SNR levels."""
        for target_snr in [0, 5, 10, 20]:
            _, _, info = simulate_neural_data(
                n_samples=100, snr_db=target_snr, seed=42
            )
            # Allow some tolerance
            assert abs(info['snr_db'] - target_snr) < 3.0


class TestCalculateVAF:
    """Tests for VAF calculation."""

    def test_perfect_match(self):
        """Test VAF = 1 for identical signals."""
        signal = np.random.randn(10, 96, 64)
        vaf = calculate_vaf(signal, signal)
        assert abs(vaf - 1.0) < 1e-6

    def test_with_noise(self):
        """Test VAF decreases with noise."""
        signal = np.random.randn(10, 96, 64)
        noisy = signal + 0.1 * np.random.randn(*signal.shape)
        vaf = calculate_vaf(signal, noisy)
        assert 0 < vaf < 1

    def test_tensor_input(self):
        """Test with torch tensors."""
        signal = torch.randn(10, 96, 64)
        vaf = calculate_vaf(signal, signal)
        assert abs(vaf - 1.0) < 1e-6


class TestCalculateSNR:
    """Tests for SNR calculation."""

    def test_high_snr(self):
        """Test high SNR for similar signals."""
        signal = np.random.randn(10, 96, 64)
        noisy = signal + 0.01 * np.random.randn(*signal.shape)
        snr = calculate_snr(signal, noisy)
        assert snr > 30  # Should be high

    def test_low_snr(self):
        """Test low SNR for noisy signals."""
        signal = np.random.randn(10, 96, 64)
        noisy = signal + signal  # 0 dB SNR
        snr = calculate_snr(signal, noisy)
        assert -5 < snr < 5


class TestRunPCABaseline:
    """Tests for PCA baseline."""

    def test_basic_pca(self):
        """Test basic PCA denoising."""
        noisy = torch.randn(50, 96, 128)
        denoised = run_pca_baseline(noisy, latent_dim=6)

        assert denoised.shape == noisy.shape
        assert isinstance(denoised, torch.Tensor)

    def test_non_negativity(self):
        """Test PCA output is non-negative."""
        noisy = torch.randn(50, 96, 128)
        denoised = run_pca_baseline(noisy, latent_dim=6)
        assert (denoised >= 0).all()


class TestValidateInputData:
    """Tests for input validation."""

    def test_valid_numpy(self):
        """Test valid numpy input."""
        data = np.random.randn(10, 96, 64).astype(np.float32)
        result = validate_input_data(data)
        assert isinstance(result, torch.Tensor)

    def test_valid_tensor(self):
        """Test valid tensor input."""
        data = torch.randn(10, 96, 64)
        result = validate_input_data(data)
        assert isinstance(result, torch.Tensor)

    def test_wrong_dims(self):
        """Test wrong dimensions."""
        data = np.random.randn(10, 96)
        with pytest.raises(ValueError, match="3D"):
            validate_input_data(data)

    def test_nan_detection(self):
        """Test NaN detection."""
        data = np.random.randn(10, 96, 64)
        data[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            validate_input_data(data)

    def test_inf_detection(self):
        """Test infinity detection."""
        data = np.random.randn(10, 96, 64)
        data[0, 0, 0] = np.inf
        with pytest.raises(ValueError, match="infinite"):
            validate_input_data(data)


class TestSetSeed:
    """Tests for seed setting."""

    def test_reproducibility(self):
        """Test reproducibility."""
        set_seed(42)
        a = np.random.randn(10)
        b = torch.randn(10)

        set_seed(42)
        a2 = np.random.randn(10)
        b2 = torch.randn(10)

        np.testing.assert_array_equal(a, a2)
        assert torch.allclose(b, b2)


class TestGetDevice:
    """Tests for device detection."""

    def test_cpu_device(self):
        """Test CPU device."""
        device = get_device(use_gpu=False, verbose=False)
        assert device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_device(self):
        """Test GPU device."""
        device = get_device(use_gpu=True, verbose=False)
        assert device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
