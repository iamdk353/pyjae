"""
Unit tests for JAE model architectures.
"""

import pytest
import torch

from jae.models import SimpleAutoencoder, JAE1, JAE2, UNet1D


class TestSimpleAutoencoder:
    """Tests for SimpleAutoencoder."""

    def test_forward_shape(self):
        """Test output shapes."""
        ae = SimpleAutoencoder(input_dim=48, latent_dim=12)
        x = torch.randn(16, 48)
        x_hat, z = ae(x)

        assert x_hat.shape == (16, 48)
        assert z.shape == (16, 12)

    def test_gradient_flow(self):
        """Test gradients flow through network."""
        ae = SimpleAutoencoder(input_dim=48, latent_dim=12)
        x = torch.randn(8, 48)
        x_hat, z = ae(x)
        loss = (x_hat - x).pow(2).mean()
        loss.backward()

        for param in ae.parameters():
            assert param.grad is not None


class TestJAE1:
    """Tests for JAE1 model."""

    def test_forward_shape(self):
        """Test output shapes."""
        model = JAE1(input_dim=96, latent_dim=12)
        x = torch.randn(16, 96, 128)
        x_denoised, z1, z2, x1_target, x2_target = model(x)

        assert x_denoised.shape == (16, 96, 128)
        assert z1.shape == (16 * 128, 12)
        assert z2.shape == (16 * 128, 12)
        assert x1_target.shape == (16, 48, 128)
        assert x2_target.shape == (16, 48, 128)

    def test_odd_input_error(self):
        """Test odd input channels raise error."""
        with pytest.raises(ValueError, match="even"):
            JAE1(input_dim=97, latent_dim=12)

    def test_gradient_flow(self):
        """Test gradients flow."""
        model = JAE1(input_dim=96, latent_dim=12)
        x = torch.randn(8, 96, 128)
        x_denoised, z1, z2, _, _ = model(x)
        loss = x_denoised.mean()
        loss.backward()

        for param in model.parameters():
            assert param.grad is not None

    def test_batch_sizes(self):
        """Test different batch sizes."""
        model = JAE1(input_dim=96, latent_dim=12)
        for batch_size in [1, 8, 32]:
            x = torch.randn(batch_size, 96, 128)
            x_denoised, _, _, _, _ = model(x)
            assert x_denoised.shape == (batch_size, 96, 128)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self):
        """Test CUDA support."""
        model = JAE1(input_dim=96, latent_dim=12).to('cuda')
        x = torch.randn(8, 96, 128).to('cuda')
        x_denoised, z1, _, _, _ = model(x)

        assert x_denoised.device.type == 'cuda'
        assert z1.device.type == 'cuda'


class TestUNet1D:
    """Tests for UNet1D."""

    def test_forward_shape(self):
        """Test output shapes."""
        unet = UNet1D(input_channels=48, channels=[32, 64], latent_dim=12)
        x = torch.randn(8, 48, 128)
        recon, latent = unet(x)

        assert recon.shape == (8, 48, 128)
        assert latent.shape == (8, 12)

    def test_non_negativity(self):
        """Test output is non-negative (ReLU)."""
        unet = UNet1D(input_channels=48, channels=[32, 64], latent_dim=12)
        x = torch.randn(4, 48, 128)
        recon, _ = unet(x)
        assert (recon >= 0).all()

    def test_gradient_flow(self):
        """Test gradient flow."""
        unet = UNet1D(input_channels=48, channels=[32, 64], latent_dim=12)
        x = torch.randn(4, 48, 128)
        recon, latent = unet(x)
        loss = recon.mean() + latent.mean()
        loss.backward()

        for param in unet.parameters():
            assert param.grad is not None

    def test_sequence_lengths(self):
        """Test different sequence lengths."""
        unet = UNet1D(input_channels=48, channels=[32, 64], latent_dim=12)
        for seq_len in [64, 128, 256]:
            x = torch.randn(4, 48, seq_len)
            recon, _ = unet(x)
            assert recon.shape == (4, 48, seq_len)


class TestJAE2:
    """Tests for JAE2 model."""

    def test_forward_shape(self):
        """Test output shapes."""
        model = JAE2(input_dim=96, latent_dim=12, num_networks=5, subsample_fraction=0.8)
        x = torch.randn(8, 96, 128)
        denoised, recons, latents, inputs_sub = model(x)

        assert denoised.shape == (8, 96, 128)
        assert len(recons) == 5
        assert len(latents) == 5
        for latent in latents:
            assert latent.shape == (8, 12)

    def test_invalid_subsample(self):
        """Test invalid subsample fraction."""
        with pytest.raises(ValueError):
            JAE2(input_dim=96, latent_dim=12, subsample_fraction=0.0)

    def test_gradient_flow(self):
        """Test gradient flow."""
        model = JAE2(input_dim=96, latent_dim=12, num_networks=3, subsample_fraction=1.0)
        x = torch.randn(4, 96, 128)
        denoised, recons, latents, _ = model(x)
        loss = denoised.sum() + sum(r.sum() for r in recons) + sum(l.sum() for l in latents)
        loss.backward()

        for param in model.parameters():
            assert param.grad is not None

    def test_num_networks(self):
        """Test different network counts."""
        for num_networks in [2, 3, 5, 8]:
            model = JAE2(input_dim=96, latent_dim=12, num_networks=num_networks)
            x = torch.randn(4, 96, 128)
            _, recons, latents, _ = model(x)
            assert len(recons) == num_networks
            assert len(latents) == num_networks

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self):
        """Test CUDA support."""
        model = JAE2(input_dim=96, latent_dim=12, num_networks=3).to('cuda')
        x = torch.randn(4, 96, 128).to('cuda')
        denoised, _, latents, _ = model(x)

        assert denoised.device.type == 'cuda'
        for latent in latents:
            assert latent.device.type == 'cuda'

    def test_no_nan(self):
        """Test output contains no NaN."""
        model = JAE2(input_dim=96, latent_dim=12, num_networks=3)
        x = torch.randn(4, 96, 128)
        denoised, _, _, _ = model(x)
        assert not torch.isnan(denoised).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
