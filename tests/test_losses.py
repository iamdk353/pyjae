"""
Unit tests for JAE loss functions.
"""

import pytest
import torch

from jae.losses import (
    jae1_loss_fn,
    jae2_loss_fn,
    huber_reconstruction_loss,
    vicreg_loss,
    mse_latent_alignment,
)


class TestJAE1Loss:
    """Tests for JAE1 loss function."""

    def test_basic(self):
        """Test basic loss computation."""
        x1_hat = torch.randn(16, 48, 128)
        x2_hat = torch.randn(16, 48, 128)
        z1 = torch.randn(2048, 12)
        z2 = torch.randn(2048, 12)

        loss = jae1_loss_fn(x1_hat, x2_hat, z1, z2, x1_hat, x2_hat)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0

    def test_perfect_reconstruction(self):
        """Test loss is zero for perfect reconstruction and alignment."""
        x1 = torch.randn(16, 48, 128)
        x2 = torch.randn(16, 48, 128)
        z = torch.randn(2048, 12)

        loss = jae1_loss_fn(x1, x2, z, z, x1, x2)

        assert loss.item() < 1e-6

    def test_latent_weight(self):
        """Test latent weight affects loss."""
        x1_hat = torch.randn(16, 48, 128)
        x2_hat = torch.randn(16, 48, 128)
        z1 = torch.randn(2048, 12)
        z2 = torch.randn(2048, 12) + 1.0  # Different

        loss_low = jae1_loss_fn(x1_hat, x2_hat, z1, z2, x1_hat, x2_hat, latent_weight=0.1)
        loss_high = jae1_loss_fn(x1_hat, x2_hat, z1, z2, x1_hat, x2_hat, latent_weight=10.0)

        assert loss_high > loss_low


class TestJAE2Loss:
    """Tests for JAE2 loss function."""

    def test_basic(self):
        """Test basic loss computation."""
        recons = [torch.randn(8, 48, 64) for _ in range(5)]
        latents = [torch.randn(8, 12) for _ in range(5)]
        targets = [torch.randn(8, 48, 64) for _ in range(5)]

        loss = jae2_loss_fn(recons, latents, targets)

        assert loss.ndim == 0
        assert loss >= 0

    def test_with_vicreg(self):
        """Test with VICReg loss."""
        recons = [torch.randn(8, 48, 64) for _ in range(5)]
        latents = [torch.randn(8, 12) for _ in range(5)]
        targets = [torch.randn(8, 48, 64) for _ in range(5)]

        loss_mse = jae2_loss_fn(recons, latents, targets, use_vicreg=False)
        loss_vicreg = jae2_loss_fn(recons, latents, targets, use_vicreg=True)

        # Both should be valid
        assert loss_mse >= 0
        assert loss_vicreg >= 0


class TestHuberLoss:
    """Tests for Huber reconstruction loss."""

    def test_basic(self):
        """Test basic computation."""
        recons = [torch.randn(8, 48, 64) for _ in range(3)]
        targets = [torch.randn(8, 48, 64) for _ in range(3)]

        loss = huber_reconstruction_loss(recons, targets)

        assert loss.ndim == 0
        assert loss >= 0

    def test_empty_list(self):
        """Test empty list returns zero."""
        loss = huber_reconstruction_loss([], [])
        assert loss.item() == 0.0

    def test_perfect_reconstruction(self):
        """Test zero loss for perfect match."""
        recons = [torch.randn(8, 48, 64) for _ in range(3)]
        loss = huber_reconstruction_loss(recons, recons)
        assert loss.item() < 1e-6


class TestVICRegLoss:
    """Tests for VICReg loss."""

    def test_basic(self):
        """Test basic computation."""
        latents = [torch.randn(32, 12) for _ in range(5)]
        loss = vicreg_loss(latents)

        assert loss.ndim == 0
        assert loss >= 0

    def test_single_view(self):
        """Test single view returns zero."""
        latents = [torch.randn(32, 12)]
        loss = vicreg_loss(latents)
        assert loss.item() == 0.0

    def test_identical_latents(self):
        """Test identical latents have low invariance loss."""
        z = torch.randn(32, 12)
        latents = [z, z.clone(), z.clone()]

        # With identical latents, invariance component should be zero
        loss = vicreg_loss(latents, lambda_inv=1.0, mu_var=0.0, nu_cov=0.0)
        assert loss.item() < 1e-6


class TestMSELatentAlignment:
    """Tests for MSE latent alignment."""

    def test_basic(self):
        """Test basic computation."""
        latents = [torch.randn(32, 12) for _ in range(5)]
        loss = mse_latent_alignment(latents)

        assert loss.ndim == 0
        assert loss >= 0

    def test_single_view(self):
        """Test single view returns zero."""
        latents = [torch.randn(32, 12)]
        loss = mse_latent_alignment(latents)
        assert loss.item() == 0.0

    def test_identical_latents(self):
        """Test identical latents have zero loss."""
        z = torch.randn(32, 12)
        latents = [z, z.clone(), z.clone()]
        loss = mse_latent_alignment(latents)
        assert loss.item() < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
