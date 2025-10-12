"""
Unit tests for JAE model architectures.
"""

import pytest
import torch
import torch.nn as nn

from jae.models import SimpleAutoencoder, JAE1, JAE2, UNet1D


class TestSimpleAutoencoder:
    """Tests for SimpleAutoencoder."""

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shapes."""
        input_dim = 48
        latent_dim = 12
        batch_size = 16

        ae = SimpleAutoencoder(input_dim, latent_dim)
        x = torch.randn(batch_size, input_dim)

        x_hat, z = ae(x)

        assert x_hat.shape == (batch_size, input_dim)
        assert z.shape == (batch_size, latent_dim)

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        ae = SimpleAutoencoder(input_dim=48, latent_dim=12)
        x = torch.randn(8, 48)

        x_hat, z = ae(x)
        loss = (x_hat - x).pow(2).mean()
        loss.backward()

        # Check that gradients exist for all parameters
        for param in ae.parameters():
            assert param.grad is not None

    def test_device_placement(self):
        """Test model can be moved to different devices."""
        ae = SimpleAutoencoder(input_dim=48, latent_dim=12)

        # Test CPU
        ae_cpu = ae.to('cpu')
        x_cpu = torch.randn(4, 48)
        x_hat, z = ae_cpu(x_cpu)
        assert x_hat.device.type == 'cpu'

        # Test CUDA if available
        if torch.cuda.is_available():
            ae_cuda = ae.to('cuda')
            x_cuda = torch.randn(4, 48).to('cuda')
            x_hat, z = ae_cuda(x_cuda)
            assert x_hat.device.type == 'cuda'


class TestJAE1:
    """Tests for JAE1 model."""

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shapes."""
        input_dim = 96
        latent_dim = 12
        batch_size = 16
        seq_len = 128

        model = JAE1(input_dim, latent_dim)
        x = torch.randn(batch_size, input_dim, seq_len)

        x_denoised, z1, z2, x1_target, x2_target = model(x)

        assert x_denoised.shape == (batch_size, input_dim, seq_len)
        assert z1.shape == (batch_size * seq_len, latent_dim)
        assert z2.shape == (batch_size * seq_len, latent_dim)
        assert x1_target.shape == (batch_size, input_dim // 2, seq_len)
        assert x2_target.shape == (batch_size, input_dim // 2, seq_len)

    def test_odd_input_dim_raises_error(self):
        """Test that odd input dimensions raise ValueError."""
        with pytest.raises(ValueError, match="even number"):
            JAE1(input_dim=97, latent_dim=12)

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        model = JAE1(input_dim=96, latent_dim=12)
        x = torch.randn(8, 96, 128)

        x_denoised, z1, z2, x1_target, x2_target = model(x)
        loss = (x_denoised - x).pow(2).mean()
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None

    def test_different_batch_sizes(self):
        """Test model works with different batch sizes."""
        model = JAE1(input_dim=96, latent_dim=12)

        for batch_size in [1, 8, 32]:
            x = torch.randn(batch_size, 96, 128)
            x_denoised, _, _, _, _ = model(x)
            assert x_denoised.shape == (batch_size, 96, 128)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_support(self):
        """Test model works on CUDA."""
        model = JAE1(input_dim=96, latent_dim=12).to('cuda')
        x = torch.randn(8, 96, 128).to('cuda')

        x_denoised, z1, z2, _, _ = model(x)

        assert x_denoised.device.type == 'cuda'
        assert z1.device.type == 'cuda'


class TestUNet1D:
    """Tests for UNet1D model."""

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shapes."""
        input_channels = 48
        latent_dim = 12
        batch_size = 8
        seq_len = 128

        unet = UNet1D(input_channels, channels=[32, 64], latent_dim=latent_dim)
        x = torch.randn(batch_size, input_channels, seq_len)

        reconstruction, latent = unet(x)

        assert reconstruction.shape == (batch_size, input_channels, seq_len)
        assert latent.shape == (batch_size, latent_dim)

    def test_non_negativity_constraint(self):
        """Test that reconstruction is non-negative."""
        unet = UNet1D(input_channels=48, channels=[32, 64], latent_dim=12)
        x = torch.randn(4, 48, 128)

        reconstruction, latent = unet(x)

        # All values should be >= 0 due to ReLU
        assert (reconstruction >= 0).all()

    def test_gradient_flow(self):
        """Test gradient flow through U-Net."""
        unet = UNet1D(input_channels=48, channels=[32, 64], latent_dim=12)
        x = torch.randn(4, 48, 128)

        reconstruction, latent = unet(x)
        loss = reconstruction.mean() + latent.mean()
        loss.backward()

        for param in unet.parameters():
            assert param.grad is not None

    def test_different_sequence_lengths(self):
        """Test U-Net with different sequence lengths."""
        unet = UNet1D(input_channels=48, channels=[32, 64], latent_dim=12)

        for seq_len in [64, 128, 256]:
            x = torch.randn(4, 48, seq_len)
            reconstruction, latent = unet(x)
            assert reconstruction.shape == (4, 48, seq_len)


class TestJAE2:
    """Tests for JAE2 model."""

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shapes."""
        input_dim = 96
        latent_dim = 12
        num_networks = 5
        batch_size = 8
        seq_len = 128

        model = JAE2(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_networks=num_networks,
            subsample_fraction=0.8
        )
        x = torch.randn(batch_size, input_dim, seq_len)

        denoised, recons, latents, inputs_sub = model(x)

        assert denoised.shape == (batch_size, input_dim, seq_len)
        assert len(recons) == num_networks
        assert len(latents) == num_networks
        assert len(inputs_sub) == num_networks

        # Check latent shapes
        for latent in latents:
            assert latent.shape == (batch_size, latent_dim)

    def test_subsample_fraction_validation(self):
        """Test that invalid subsample_fraction raises error."""
        with pytest.raises(ValueError):
            JAE2(input_dim=96, latent_dim=12, subsample_fraction=0.0)

    def test_gradient_flow(self):
        """Test gradient flow through JAE2."""
        model = JAE2(input_dim=96, latent_dim=12, num_networks=3)
        x = torch.randn(4, 96, 128)

        denoised, recons, latents, inputs_sub = model(x)
        loss = denoised.mean()
        loss.backward()

        for param in model.parameters():
            assert param.grad is not None

    def test_different_num_networks(self):
        """Test JAE2 with different numbers of parallel networks."""
        for num_networks in [2, 3, 5, 8]:
            model = JAE2(input_dim=96, latent_dim=12, num_networks=num_networks)
            x = torch.randn(4, 96, 128)

            denoised, recons, latents, _ = model(x)

            assert len(recons) == num_networks
            assert len(latents) == num_networks

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_support(self):
        """Test JAE2 works on CUDA."""
        model = JAE2(input_dim=96, latent_dim=12, num_networks=3).to('cuda')
        x = torch.randn(4, 96, 128).to('cuda')

        denoised, recons, latents, _ = model(x)

        assert denoised.device.type == 'cuda'
        for latent in latents:
            assert latent.device.type == 'cuda'

    def test_reconstruction_averaging(self):
        """Test that reconstructions are properly averaged."""
        model = JAE2(input_dim=96, latent_dim=12, num_networks=3)
        x = torch.randn(4, 96, 128)

        denoised, _, _, _ = model(x)

        # Denoised output should not contain NaN or Inf
        assert not torch.isnan(denoised).any()
        assert not torch.isinf(denoised).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

