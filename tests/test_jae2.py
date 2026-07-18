"""Unit tests for the JEPA-style JAE2 model (src/pyjae/models/jae2.py)."""

import math

import pytest
import torch

from pyjae.models.jae2 import JAE2, JAE2Output


def _low_rank_batch(
    n_samples: int, n_channels: int, n_timepoints: int, rank: int = 3, noise_std: float = 0.05
) -> torch.Tensor:
    """Build a tiny low-rank-plus-noise signal for the anti-collapse smoke test.

    Every sample is a random per-channel linear combination of `rank` shared sine
    bases, plus i.i.d. Gaussian noise. This gives the encoder real shared
    structure to learn (so representations should stay spread out / non-trivial)
    without needing real neural data.

    Args:
        n_samples: Number of samples to generate.
        n_channels: Number of channels per sample.
        n_timepoints: Number of timepoints per sample.
        rank: Number of shared sine bases (the "true" rank of the signal).
        noise_std: Standard deviation of additive Gaussian noise.

    Returns:
        Tensor of shape (n_samples, n_channels, n_timepoints).
    """
    gen = torch.Generator().manual_seed(1234)
    t = torch.linspace(0, 4 * math.pi, n_timepoints)
    basis = torch.stack([torch.sin((k + 1) * t) for k in range(rank)], dim=0)  # (rank, T)
    loadings = torch.randn(n_samples, n_channels, rank, generator=gen)
    x = torch.einsum("bcr,rt->bct", loadings, basis)
    x = x + noise_std * torch.randn(n_samples, n_channels, n_timepoints, generator=gen)
    return x


def _effective_rank(z: torch.Tensor) -> float:
    """Compute the effective rank (exp of the entropy of normalized singular values).

    Reference: RankMe (Garrido et al. 2023). A collapsed embedding matrix has one
    dominant singular value (entropy ~= 0, effective rank ~= 1); a healthy,
    spread-out embedding has effective rank closer to min(N, D).

    Args:
        z: Embedding matrix, shape (N, D).

    Returns:
        Effective rank as a Python float.
    """
    z_centered = z - z.mean(dim=0)
    singular_values = torch.linalg.svdvals(z_centered)
    p = singular_values / singular_values.sum().clamp_min(1e-12)
    p = p[p > 1e-12]
    entropy = -(p * p.log()).sum()
    return torch.exp(entropy).item()


class TestJAE2Forward:
    """Forward-pass shape and sanity tests."""

    def test_forward_shapes_and_no_nan(self):
        """forward on (8, 96, 128): all output shapes match the documented contract."""
        model = JAE2(input_dim=96)
        x = torch.randn(8, 96, 128)
        out = model(x)

        assert isinstance(out, JAE2Output)
        assert out.denoised.shape == (8, 96, 128)
        assert out.pred_tokens.shape == out.target_tokens.shape
        assert out.pred_tokens.shape[1] == model.d_model
        assert out.z_context.shape == out.z_target.shape == (8 * model.n_patches, model.latent_dim)

        for tensor in out:
            assert torch.isfinite(tensor).all()

    def test_forward_m_matches_target_grid_size(self):
        """M (pred/target token count) equals batch_size * n_target_grid_positions."""
        model = JAE2(input_dim=16, n_timepoints=32, patch_len=8, d_model=32, seed=0)
        x = torch.randn(3, 16, 32)
        gen = torch.Generator().manual_seed(7)
        out = model(x, generator=gen)

        gen_check = torch.Generator().manual_seed(7)
        _, target_blocks = model.mask(model.input_dim, model.n_patches, gen_check)
        union = torch.zeros_like(target_blocks[0])
        for block in target_blocks:
            union |= block
        n_targets = int(union.sum().item())

        assert out.pred_tokens.shape == (3 * n_targets, model.d_model)
        assert out.target_tokens.shape == (3 * n_targets, model.d_model)


class TestJAE2Loss:
    """Loss / gradient-flow tests."""

    def test_gradient_flow_to_all_submodules(self):
        """loss(out).backward() populates finite grads on every submodule's params."""
        model = JAE2(input_dim=24, n_timepoints=32, patch_len=8, d_model=32, latent_dim=16)
        x = torch.randn(4, 24, 32)
        out = model(x)
        loss = model.loss(out)
        loss.backward()

        submodules = {
            "encoder": model.encoder,
            "predictor": model.predictor,
            "projector": model.projector,
            "decoder_head": model.decoder_head,
        }
        for name, module in submodules.items():
            for pname, param in module.named_parameters():
                assert param.grad is not None, f"no grad for {name}.{pname}"
                assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}.{pname}"

        assert model.mask_token.grad is not None
        assert model.channel_pos_embed.weight.grad is not None
        assert model.patch_pos_embed.weight.grad is not None

    def test_loss_without_forward_raises(self):
        """Calling loss() before forward() raises, since there's no stashed input."""
        model = JAE2(input_dim=8, n_timepoints=16, patch_len=8, d_model=16)
        x = torch.randn(2, 8, 16)
        out = model(x)
        fresh_model = JAE2(input_dim=8, n_timepoints=16, patch_len=8, d_model=16)
        with pytest.raises(RuntimeError):
            fresh_model.loss(out)

    def test_stop_gradient_on_target_tokens(self):
        """Prediction term must not backprop into target_tokens (JEPA stop-grad).

        With lambda_var = lambda_cov = 0 and recon_weight = 0, the only surviving
        loss term is lambda_pred * SmoothL1(pred_tokens, stopgrad(target_tokens)).
        pyjae.losses.jepa_loss detaches its local copy of the target before the
        Smooth L1 comparison, so no gradient should ever reach `target_tokens`
        itself (retain_grad lets us inspect that directly). This is exactly what
        prevents the "collapse the target to make prediction trivial" failure
        mode: the encoder gets no signal, through this term, that would reward
        making target embeddings easy to predict (e.g. constant). Separately, the
        context/predictor path (which is NOT detached) must still receive a
        nonzero gradient, proving the model actually trains through that branch.
        """
        model = JAE2(input_dim=16, n_timepoints=32, patch_len=8, d_model=32, seed=3)
        x = torch.randn(4, 16, 32)
        out = model(x)
        out.target_tokens.retain_grad()

        loss = model.loss(out, lambda_pred=25.0, lambda_var=0.0, lambda_cov=0.0, recon_weight=0.0)
        loss.backward()

        # Stop-gradient held: no gradient ever reached target_tokens.
        assert out.target_tokens.grad is None

        # Context/predictor path is alive: encoder and predictor got real gradient.
        encoder_grad = model.encoder.patch_embed.weight.grad
        assert encoder_grad is not None
        assert torch.any(encoder_grad != 0)

        predictor_grad = model.predictor[0].weight.grad
        assert predictor_grad is not None
        assert torch.any(predictor_grad != 0)


class TestJAE2AntiCollapse:
    """Smoke test that training does not collapse the projected embeddings."""

    def test_training_reduces_loss_without_collapsing(self):
        """~60 Adam steps on tiny low-rank data: loss drops, embeddings stay spread out.

        VICReg's variance hinge (weighted by lambda_var) is exactly the mechanism
        that should prevent the trivial "collapse everything to a constant"
        shortcut that would otherwise make the JEPA prediction loss vanish for
        free. This test checks both sides: the model actually learns (loss goes
        down) and it does not learn by collapsing (per-dim std stays above a
        floor and the effective rank of z_context stays well above 1).
        """
        torch.manual_seed(0)
        model = JAE2(
            input_dim=16,
            n_timepoints=32,
            patch_len=8,
            d_model=32,
            latent_dim=16,
            predictor_dim=16,
            seed=0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x_pool = _low_rank_batch(n_samples=64, n_channels=16, n_timepoints=32)
        data_gen = torch.Generator().manual_seed(2)

        losses = []
        for _ in range(60):
            idx = torch.randint(0, x_pool.shape[0], (8,), generator=data_gen)
            x = x_pool[idx]
            out = model(x)
            loss = model.loss(out, lambda_pred=25.0, lambda_var=25.0, lambda_cov=1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0]

        model.eval()
        with torch.no_grad():
            final_out = model(x_pool)
        z = final_out.z_context

        std_mean = z.std(dim=0).mean().item()
        eff_rank = _effective_rank(z)

        assert std_mean > 1e-3, f"embedding std collapsed: {std_mean}"
        assert eff_rank > 1.5, f"effective rank collapsed: {eff_rank}"


class TestJAE2Denoise:
    """Tests for the standalone denoise() readout."""

    def test_denoise_shape_and_eval_mode(self):
        """denoise() returns (B, C, T) and runs cleanly in eval mode, no grad."""
        model = JAE2(input_dim=12, n_timepoints=24, patch_len=8, d_model=16, latent_dim=8)
        model.eval()
        x = torch.randn(3, 12, 24)
        with torch.no_grad():
            denoised = model.denoise(x)

        assert denoised.shape == (3, 12, 24)
        assert torch.isfinite(denoised).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
