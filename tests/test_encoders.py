"""Unit tests for shared encoder/decoder building blocks in jae.models.encoders."""

import pytest
import torch

from jae.models.encoders import ConvPatchEncoder, MLPDecoder, MLPEncoder


class TestMLPEncoder:
    """Tests for MLPEncoder."""

    def test_output_shape(self):
        """Output shape is (N, latent_dim)."""
        enc = MLPEncoder(in_dim=96, latent_dim=12, hidden=(128, 64))
        x = torch.randn(16, 96)
        z = enc(x)
        assert z.shape == (16, 12)

    def test_relu_latent_non_negative(self):
        """latent_activation='relu' produces an all non-negative latent."""
        enc = MLPEncoder(in_dim=96, latent_dim=12, latent_activation="relu")
        x = torch.randn(64, 96) * 10.0
        z = enc(x)
        assert (z >= 0).all()

    def test_none_latent_can_be_negative(self):
        """latent_activation='none' allows negative latent values."""
        torch.manual_seed(0)
        found_negative = False
        enc = MLPEncoder(in_dim=96, latent_dim=12, latent_activation="none")
        for _ in range(10):
            x = torch.randn(64, 96) * 10.0
            z = enc(x)
            if (z < 0).any():
                found_negative = True
                break
        assert found_negative

    def test_tanh_latent_bounded(self):
        """latent_activation='tanh' bounds outputs to [-1, 1]."""
        enc = MLPEncoder(in_dim=96, latent_dim=12, latent_activation="tanh")
        x = torch.randn(32, 96) * 100.0
        z = enc(x)
        assert (z >= -1).all() and (z <= 1).all()

    def test_invalid_latent_activation_raises(self):
        """Unsupported latent_activation values raise ValueError."""
        with pytest.raises(ValueError):
            MLPEncoder(in_dim=96, latent_dim=12, latent_activation="sigmoid")

    def test_gradient_flow(self):
        """Gradients flow to every parameter after a backward pass."""
        enc = MLPEncoder(in_dim=96, latent_dim=12, hidden=(128, 64))
        x = torch.randn(8, 96)
        z = enc(x)
        loss = z.pow(2).mean()
        loss.backward()
        for name, param in enc.named_parameters():
            assert param.grad is not None, f"no grad for {name}"
            assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"

    def test_empty_hidden(self):
        """Works with no hidden layers (direct in_dim -> latent_dim)."""
        enc = MLPEncoder(in_dim=32, latent_dim=8, hidden=())
        x = torch.randn(4, 32)
        z = enc(x)
        assert z.shape == (4, 8)


class TestMLPDecoder:
    """Tests for MLPDecoder."""

    def test_output_shape(self):
        """Output shape is (N, out_dim)."""
        dec = MLPDecoder(latent_dim=12, out_dim=96, hidden=(64, 128))
        z = torch.randn(16, 12)
        x_hat = dec(z)
        assert x_hat.shape == (16, 96)

    def test_relu_output_non_negative(self):
        """out_activation='relu' produces an all non-negative reconstruction."""
        dec = MLPDecoder(latent_dim=12, out_dim=96, out_activation="relu")
        z = torch.randn(64, 12) * 10.0
        x_hat = dec(z)
        assert (x_hat >= 0).all()

    def test_none_output_can_be_negative(self):
        """out_activation='none' allows negative reconstructed values."""
        torch.manual_seed(0)
        found_negative = False
        dec = MLPDecoder(latent_dim=12, out_dim=96, out_activation="none")
        for _ in range(10):
            z = torch.randn(64, 12) * 10.0
            x_hat = dec(z)
            if (x_hat < 0).any():
                found_negative = True
                break
        assert found_negative

    def test_final_layer_bias_initialized_to_pos_bias(self):
        """The final linear layer's bias is initialized to pos_bias."""
        pos_bias = 0.37
        dec = MLPDecoder(latent_dim=12, out_dim=96, hidden=(64, 128), pos_bias=pos_bias)
        final_linear = [m for m in dec.net if isinstance(m, torch.nn.Linear)][-1]
        assert torch.allclose(final_linear.bias, torch.full_like(final_linear.bias, pos_bias))

    def test_invalid_out_activation_raises(self):
        """Unsupported out_activation values raise ValueError."""
        with pytest.raises(ValueError):
            MLPDecoder(latent_dim=12, out_dim=96, out_activation="tanh")

    def test_gradient_flow(self):
        """Gradients flow to every parameter after a backward pass."""
        dec = MLPDecoder(latent_dim=12, out_dim=96, hidden=(64, 128))
        z = torch.randn(8, 12)
        x_hat = dec(z)
        loss = x_hat.pow(2).mean()
        loss.backward()
        for name, param in dec.named_parameters():
            assert param.grad is not None, f"no grad for {name}"
            assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"


class TestConvPatchEncoder:
    """Tests for ConvPatchEncoder."""

    @pytest.mark.parametrize(
        "n_channels,n_patches,patch_len,d_model",
        [
            (96, 16, 8, 128),
            (48, 8, 16, 64),
            (12, 4, 4, 32),
        ],
    )
    def test_output_shapes(self, n_channels, n_patches, patch_len, d_model):
        """tokens and pooled have the documented shapes for several size combos."""
        enc = ConvPatchEncoder(
            n_channels=n_channels, n_patches=n_patches, patch_len=patch_len, d_model=d_model
        )
        t = n_patches * patch_len
        x = torch.randn(4, n_channels, t)
        tokens, pooled = enc(x)
        assert tokens.shape == (4, n_channels, n_patches, d_model)
        assert pooled.shape == (4, n_patches, d_model)

    def test_pooled_is_mean_over_channels(self):
        """pooled equals the mean of tokens over the channel dimension."""
        enc = ConvPatchEncoder(n_channels=12, n_patches=4, patch_len=4, d_model=32)
        x = torch.randn(2, 12, 16)
        tokens, pooled = enc(x)
        assert torch.allclose(pooled, tokens.mean(dim=1), atol=1e-5)

    def test_truncates_longer_input(self):
        """Input longer than n_patches * patch_len is truncated, not erroring."""
        enc = ConvPatchEncoder(n_channels=12, n_patches=4, patch_len=4, d_model=32)
        x = torch.randn(2, 12, 20)  # expected T = 16, this is longer
        tokens, pooled = enc(x)
        assert tokens.shape == (2, 12, 4, 32)
        assert pooled.shape == (2, 4, 32)

    def test_pads_shorter_input(self):
        """Input shorter than n_patches * patch_len is zero-padded, not erroring."""
        enc = ConvPatchEncoder(n_channels=12, n_patches=4, patch_len=4, d_model=32)
        x = torch.randn(2, 12, 10)  # expected T = 16, this is shorter
        tokens, pooled = enc(x)
        assert tokens.shape == (2, 12, 4, 32)
        assert pooled.shape == (2, 4, 32)

    def test_gradient_flow(self):
        """Gradients flow to every parameter, including the channel-mixing layer."""
        enc = ConvPatchEncoder(n_channels=96, n_patches=16, patch_len=8, d_model=128)
        x = torch.randn(2, 96, 128)
        tokens, pooled = enc(x)
        loss = tokens.pow(2).mean() + pooled.pow(2).mean()
        loss.backward()
        for name, param in enc.named_parameters():
            assert param.grad is not None, f"no grad for {name}"
            assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"

    def test_batch_independence(self):
        """Different batch sizes work and per-sample outputs are unaffected by batch size."""
        enc = ConvPatchEncoder(n_channels=12, n_patches=4, patch_len=4, d_model=32)
        enc.eval()
        x = torch.randn(1, 12, 16)
        with torch.no_grad():
            tokens_single, pooled_single = enc(x)
            tokens_batched, pooled_batched = enc(x.repeat(5, 1, 1))
        assert torch.allclose(tokens_batched[0], tokens_single[0], atol=1e-5)
        assert torch.allclose(pooled_batched[0], pooled_single[0], atol=1e-5)
