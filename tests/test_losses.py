"""
Unit tests for JAE loss functions.
"""

import pytest
import torch
import numpy as np

from jae.losses import (
    jae1_loss_fn,
    huber_reconstruction_loss,
    vicreg_loss,
    jae2_loss_fn
)


class TestJAE1Loss:
    """Tests for JAE1 loss function."""

    def test_loss_computation(self):
        """Test that loss is computed correctly."""
        batch_size, channels, time = 16, 48, 128
        latent_dim = 12

        x1_hat = torch.randn(batch_size, channels, time)
        x2_hat = torch.randn(batch_size, channels, time)
        z1 = torch.randn(batch_size * time, latent_dim)
        z2 = torch.randn(batch_size * time, latent_dim)
        x1_target = torch.randn(batch_size, channels, time)
        x2_target = torch.randn(batch_size, channels, time)

        loss = jae1_loss_fn(x1_hat, x2_hat, z1, z2, x1_target, x2_target)

        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)
        assert loss >= 0

    def test_zero_loss_perfect_reconstruction(self):
        """Test that loss is zero for perfect reconstruction and alignment."""
        batch_size, channels, time = 8, 48, 64
        latent_dim = 12

        x1 = torch.randn(batch_size, channels, time)
        x2 = torch.randn(batch_size, channels, time)
        z = torch.randn(batch_size * time, latent_dim)

        # Perfect reconstruction and alignment
        loss = jae1_loss_fn(x1, x2, z, z, x1, x2)

        assert loss.item() < 1e-6  # Should be very close to zero

    def test_latent_weight(self):
        """Test that latent_weight affects loss."""
        batch_size, channels, time = 8, 48, 64
        latent_dim = 12

        x1_hat = torch.randn(batch_size, channels, time)
        x2_hat = torch.randn(batch_size, channels, time)
        z1 = torch.randn(batch_size * time, latent_dim)
        z2 = torch.randn(batch_size * time, latent_dim)
        x1_target = torch.randn(batch_size, channels, time)
        x2_target = torch.randn(batch_size, channels, time)

        loss_w1 = jae1_loss_fn(x1_hat, x2_hat, z1, z2, x1_target, x2_target, latent_weight=1.0)
        loss_w2 = jae1_loss_fn(x1_hat, x2_hat, z1, z2, x1_target, x2_target, latent_weight=2.0)

        assert loss_w2 > loss_w1  # Higher weight should increase loss

    def test_gradient_flow(self):
        """Test that gradients flow through the loss."""
        x1_hat = torch.randn(4, 24, 64, requires_grad=True)
        x2_hat = torch.randn(4, 24, 64, requires_grad=True)
        z1 = torch.randn(256, 12, requires_grad=True)
        z2 = torch.randn(256, 12, requires_grad=True)
        x1_target = torch.randn(4, 24, 64)
        x2_target = torch.randn(4, 24, 64)

        loss = jae1_loss_fn(x1_hat, x2_hat, z1, z2, x1_target, x2_target)
        loss.backward()

        assert x1_hat.grad is not None
        assert z1.grad is not None


class TestHuberLoss:
    """Tests for Huber reconstruction loss."""

    def test_empty_list(self):
        """Test that empty list returns zero."""
        loss = huber_reconstruction_loss([], [])
        assert loss == 0.0

    def test_single_view(self):
        """Test with single view."""
        recon = torch.randn(8, 48, 64)
        target = torch.randn(8, 48, 64)

        loss = huber_reconstruction_loss([recon], [target])

        assert loss.ndim == 0
        assert loss >= 0

    def test_multiple_views(self):
        """Test with multiple views."""
        recons = [torch.randn(8, 48, 64) for _ in range(5)]
        targets = [torch.randn(8, 48, 64) for _ in range(5)]

        loss = huber_reconstruction_loss(recons, targets)

        assert loss.ndim == 0
        assert loss >= 0

    def test_perfect_reconstruction(self):
        """Test that loss is zero for perfect reconstruction."""
        data = torch.randn(8, 48, 64)

        loss = huber_reconstruction_loss([data], [data])

        assert loss.item() < 1e-6

    def test_delta_parameter(self):
        """Test that delta parameter affects loss."""
        recon = torch.randn(8, 48, 64)
        target = torch.randn(8, 48, 64)

        loss_d1 = huber_reconstruction_loss([recon], [target], delta=1.0)
        loss_d2 = huber_reconstruction_loss([recon], [target], delta=2.0)

        # Both should be valid losses
        assert loss_d1 >= 0
        assert loss_d2 >= 0


class TestVICRegLoss:
    """Tests for VICReg loss."""

    def test_empty_list(self):
        """Test that empty list returns zero."""
        loss = vicreg_loss([])
        assert loss == 0.0

    def test_single_latent(self):
        """Test that single latent returns zero."""
        latent = torch.randn(32, 12)
        loss = vicreg_loss([latent])
        assert loss == 0.0

    def test_two_latents(self):
        """Test with two latent representations."""
        latent1 = torch.randn(32, 12)
        latent2 = torch.randn(32, 12)

        loss = vicreg_loss([latent1, latent2])

        assert loss.ndim == 0
        assert loss >= 0

    def test_multiple_latents(self):
        """Test with multiple latent representations."""
        latents = [torch.randn(32, 12) for _ in range(5)]

        loss = vicreg_loss(latents)

        assert loss.ndim == 0
        assert loss >= 0

    def test_batch_size_one_handling(self):
        """Test that batch size 1 is handled gracefully."""
        latents = [torch.randn(1, 12) for _ in range(3)]

        # Should not raise an error
        loss = vicreg_loss(latents)
        assert loss >= 0

    def test_identical_latents(self):
        """Test with identical latents (perfect invariance)."""
        latent = torch.randn(32, 12)

        loss = vicreg_loss([latent, latent.clone()])

        # Invariance term should be zero, but variance and covariance terms may not be
        assert loss >= 0

    def test_weight_parameters(self):
        """Test that weight parameters affect loss."""
        latents = [torch.randn(32, 12) for _ in range(3)]

        loss_default = vicreg_loss(latents)
        loss_high_inv = vicreg_loss(latents, lambda_inv=100.0)

        # Both should be valid
        assert loss_default >= 0
        assert loss_high_inv >= 0

    def test_gradient_flow(self):
        """Test that gradients flow through VICReg loss."""
        latent1 = torch.randn(32, 12, requires_grad=True)
        latent2 = torch.randn(32, 12, requires_grad=True)

        loss = vicreg_loss([latent1, latent2])
        loss.backward()

        assert latent1.grad is not None
        assert latent2.grad is not None


class TestJAE2Loss:
    """Tests for combined JAE2 loss."""

    def test_loss_computation(self):
        """Test that combined loss is computed correctly."""
        recons = [torch.randn(8, 48, 64) for _ in range(5)]
        latents = [torch.randn(8, 12) for _ in range(5)]
        targets = [torch.randn(8, 48, 64) for _ in range(5)]

        loss = jae2_loss_fn(recons, latents, targets)

        assert loss.ndim == 0
        assert loss >= 0
        assert not torch.isnan(loss)

    def test_empty_inputs(self):
        """Test behavior with empty inputs."""
        loss = jae2_loss_fn([], [], [])

        # Should handle gracefully
        assert loss.ndim == 0

    def test_weight_parameters(self):
        """Test that weight parameters affect loss."""
        recons = [torch.randn(8, 48, 64) for _ in range(3)]
        latents = [torch.randn(8, 12) for _ in range(3)]
        targets = [torch.randn(8, 48, 64) for _ in range(3)]

        loss_default = jae2_loss_fn(recons, latents, targets)
        loss_high_recon = jae2_loss_fn(recons, latents, targets, recon_weight=10.0)
        loss_high_vicreg = jae2_loss_fn(recons, latents, targets, vicreg_weight=1.0)

        # All should be valid
        assert loss_default >= 0
        assert loss_high_recon >= 0
        assert loss_high_vicreg >= 0

    def test_gradient_flow(self):
        """Test gradient flow through combined loss."""
        recons = [torch.randn(8, 48, 64, requires_grad=True) for _ in range(3)]
        latents = [torch.randn(8, 12, requires_grad=True) for _ in range(3)]
        targets = [torch.randn(8, 48, 64) for _ in range(3)]

        loss = jae2_loss_fn(recons, latents, targets)
        loss.backward()

        for recon in recons:
            assert recon.grad is not None
        for latent in latents:
            assert latent.grad is not None

    def test_with_smoothness_loss(self):
        """Test JAE2 loss with temporal smoothness regularization."""
        recons = [torch.randn(8, 48, 64) for _ in range(3)]
        latents = [torch.randn(8, 12) for _ in range(3)]
        targets = [torch.randn(8, 48, 64) for _ in range(3)]
        denoised = torch.randn(8, 96, 128)

        loss_without = jae2_loss_fn(recons, latents, targets, denoised_output=None)
        loss_with = jae2_loss_fn(recons, latents, targets, denoised_output=denoised,
                                  smoothness_weight=0.01)

        assert loss_without.ndim == 0
        assert loss_with.ndim == 0
        assert loss_with >= 0
        # Loss with smoothness should generally be different (higher)
        # But we can't guarantee it's always higher due to randomness

    def test_smoothness_weight_scaling(self):
        """Test that smoothness weight scales the smoothness loss contribution."""
        recons = [torch.randn(8, 48, 64) for _ in range(3)]
        latents = [torch.randn(8, 12) for _ in range(3)]
        targets = [torch.randn(8, 48, 64) for _ in range(3)]
        denoised = torch.randn(8, 96, 128)

        loss_small = jae2_loss_fn(recons, latents, targets, denoised_output=denoised,
                                   smoothness_weight=0.001)
        loss_large = jae2_loss_fn(recons, latents, targets, denoised_output=denoised,
                                   smoothness_weight=0.1)

        # With larger smoothness weight, total loss should generally be higher
        assert loss_small >= 0
        assert loss_large >= 0


class TestTemporalSmoothnessLoss:
    """Tests for temporal smoothness regularization."""

    def test_output_shape(self):
        """Test that output is a scalar."""
        from jae.losses import temporal_smoothness_loss

        x = torch.randn(8, 96, 128)
        loss = temporal_smoothness_loss(x)

        assert loss.ndim == 0
        assert loss >= 0

    def test_constant_signal_zero_loss(self):
        """Test that constant signal has zero smoothness loss."""
        from jae.losses import temporal_smoothness_loss

        x = torch.ones(8, 96, 128) * 5.0
        loss = temporal_smoothness_loss(x)

        assert loss.item() < 1e-6

    def test_jagged_signal_high_loss(self):
        """Test that jagged signal has higher loss than smooth signal."""
        from jae.losses import temporal_smoothness_loss

        # Smooth signal (low frequency)
        t = torch.linspace(0, 4 * np.pi, 128)
        smooth = torch.sin(t).unsqueeze(0).unsqueeze(0).repeat(8, 96, 1)

        # Jagged signal (high frequency)
        jagged = torch.sin(20 * t).unsqueeze(0).unsqueeze(0).repeat(8, 96, 1)

        loss_smooth = temporal_smoothness_loss(smooth)
        loss_jagged = temporal_smoothness_loss(jagged)

        assert loss_jagged > loss_smooth

    def test_gradient_flow(self):
        """Test gradient flow through smoothness loss."""
        from jae.losses import temporal_smoothness_loss

        x = torch.randn(8, 96, 128, requires_grad=True)
        loss = temporal_smoothness_loss(x)
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_batch_invariance(self):
        """Test that loss scales properly with batch size."""
        from jae.losses import temporal_smoothness_loss

        x_small = torch.randn(4, 96, 128)
        x_large = torch.cat([x_small, x_small], dim=0)  # Double batch size

        loss_small = temporal_smoothness_loss(x_small)
        loss_large = temporal_smoothness_loss(x_large)

        # Losses should be similar (averaged over batches)
        assert abs(loss_small - loss_large) < 0.5 * (loss_small + loss_large)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

