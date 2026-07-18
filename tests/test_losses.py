"""
Unit tests for JAE loss functions.
"""

import pytest
import torch

from pyjae.losses import (
    huber_reconstruction_loss,
    jae1_loss_fn,
    jae2_jepa_loss_fn,
    jae2_loss_fn,
    jepa_loss,
    mse_latent_alignment,
    vicreg_loss,
    vicreg_reg,
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


class TestVICRegReg:
    """Tests for standalone VICReg variance/covariance regularization terms."""

    def test_collapsed_var_term_large(self):
        """Identical rows (zero variance) should trigger a large variance hinge."""
        z = torch.ones(32, 8)
        var_term, _ = vicreg_reg(z, gamma=1.0)
        assert var_term.item() > 0.9

    def test_wellspread_var_term_near_zero(self):
        """Unit-variance, well-spread Z should have a near-zero variance hinge."""
        torch.manual_seed(0)
        z = torch.randn(20000, 16)
        var_term, _ = vicreg_reg(z, gamma=1.0)
        assert var_term.item() < 0.05

    def test_decorrelated_cov_term_near_zero(self):
        """Independent dimensions should have a near-zero covariance term."""
        torch.manual_seed(0)
        z = torch.randn(20000, 16)
        _, cov_term = vicreg_reg(z)
        assert cov_term.item() < 0.05

    def test_correlated_cov_term_larger(self):
        """Strongly correlated dimensions should raise the covariance term."""
        torch.manual_seed(0)
        base = torch.randn(20000, 1)
        noise = 0.01 * torch.randn(20000, 1)
        z_correlated = torch.cat([base, base + noise], dim=1)
        z_decorrelated = torch.randn(20000, 2)

        _, cov_corr = vicreg_reg(z_correlated)
        _, cov_decorr = vicreg_reg(z_decorrelated)

        assert cov_corr.item() > cov_decorr.item()

    def test_returns_scalar_tensors(self):
        """Returns a (var_term, cov_term) tuple of scalar tensors."""
        z = torch.randn(32, 8)
        var_term, cov_term = vicreg_reg(z)
        assert var_term.ndim == 0
        assert cov_term.ndim == 0


class TestJEPALoss:
    """Tests for the JEPA prediction loss."""

    def test_identical_zero(self):
        """Identical pred and target give zero loss."""
        x = torch.randn(16, 32)
        loss = jepa_loss(x, x.clone())
        assert loss.item() < 1e-6

    def test_stop_grad_blocks_target_gradient(self):
        """With stop_grad=True, gradient flows to pred but not to target."""
        pred = torch.randn(8, 4, requires_grad=True)
        target = torch.randn(8, 4, requires_grad=True)

        loss = jepa_loss(pred, target, stop_grad=True)
        loss.backward()

        assert pred.grad is not None
        assert target.grad is None

    def test_no_stop_grad_allows_target_gradient(self):
        """With stop_grad=False, gradient flows to both pred and target."""
        pred = torch.randn(8, 4, requires_grad=True)
        target = torch.randn(8, 4, requires_grad=True)

        loss = jepa_loss(pred, target, stop_grad=False)
        loss.backward()

        assert pred.grad is not None
        assert target.grad is not None


class TestJAE2JEPALoss:
    """Tests for jae2_jepa_loss_fn."""

    def test_positive_scalar(self):
        """Returns a positive scalar loss."""
        pred_tokens = torch.randn(8, 16)
        target_tokens = torch.randn(8, 16)
        z_ctx = torch.randn(32, 16)
        z_tgt = torch.randn(32, 16)

        loss = jae2_jepa_loss_fn(pred_tokens, target_tokens, z_ctx, z_tgt)

        assert loss.ndim == 0
        assert loss.item() > 0

    def test_differentiable(self):
        """Gradients flow to pred_tokens and Z_ctx."""
        pred_tokens = torch.randn(8, 16, requires_grad=True)
        target_tokens = torch.randn(8, 16)
        z_ctx = torch.randn(32, 16, requires_grad=True)
        z_tgt = torch.randn(32, 16)

        loss = jae2_jepa_loss_fn(pred_tokens, target_tokens, z_ctx, z_tgt)
        loss.backward()

        assert pred_tokens.grad is not None
        assert z_ctx.grad is not None


class TestJAE1LossDefaultLatentWeight:
    """Tests for the jae1_loss_fn default latent_weight change (2.0 -> 1.0)."""

    def test_default_is_unweighted_sum(self):
        """Perfect reconstruction and aligned latents give ~0 loss with default weight."""
        x1 = torch.randn(16, 48, 128)
        x2 = torch.randn(16, 48, 128)
        z = torch.randn(2048, 12)

        loss = jae1_loss_fn(x1, x2, z, z, x1, x2)
        assert loss.item() < 1e-6

    def test_default_adds_exact_latent_mse(self):
        """With mismatched latents, default weight adds exactly the latent MSE."""
        x1 = torch.randn(16, 48, 128)
        x2 = torch.randn(16, 48, 128)
        z1 = torch.randn(2048, 12)
        z2 = torch.randn(2048, 12)

        loss = jae1_loss_fn(x1, x2, z1, z2, x1, x2)
        expected_latent_mse = torch.nn.functional.mse_loss(z1, z2)

        assert loss.item() == pytest.approx(expected_latent_mse.item(), abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
