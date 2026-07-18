"""Unit tests for the modular, collapse-resistant pyjae.models.jae1.JAE1."""

import torch
import torch.nn.functional as F  # noqa: N812

from pyjae.models.jae1 import JAE1, JAE1Output
from pyjae.views import ContiguousSplit


class TestJAE1ForwardShapes:
    """Tests for JAE1.forward output shapes with the default split."""

    def test_default_split_shapes_and_coverage(self):
        """(8, 96, 128) input: 2 partitions covering all 96 channels disjointly."""
        model = JAE1(input_dim=96, latent_dim=12, seed=0)
        x = torch.randn(8, 96, 128)
        out = model(x)

        assert isinstance(out, JAE1Output)
        assert out.denoised.shape == (8, 96, 128)
        assert model.n_partitions == 2
        assert len(out.latents) == 2
        assert len(out.recons) == 2
        assert len(out.targets) == 2
        assert len(out.indices) == 2

        for z in out.latents:
            assert z.shape == (8 * 128, 12)

        for idx, recon, target in zip(out.indices, out.recons, out.targets):
            n_c = idx.numel()
            assert recon.shape == (8 * 128, n_c)
            assert target.shape == (8 * 128, n_c)

        # Partitions are disjoint and, for this even channel count with the
        # default 50/50 split, cover every channel exactly once.
        all_idx = torch.cat(out.indices)
        assert all_idx.numel() == 96
        assert torch.equal(torch.sort(all_idx).values, torch.arange(96))

    def test_odd_channels_default_split_no_crash(self):
        """Odd channel count with the default split runs without error."""
        model = JAE1(input_dim=97, latent_dim=8, seed=0)
        x = torch.randn(4, 97, 32)
        out = model(x)

        assert out.denoised.shape == (4, 97, 32)
        assert len(model.partition_sizes) == model.n_partitions
        for size in model.partition_sizes:
            assert size > 0

    def test_unequal_partition_sizes_handled(self):
        """A split producing genuinely unequal partition sizes works end to end."""
        # 100 channels into 3 contiguous partitions: sizes (34, 33, 33).
        model = JAE1(input_dim=100, latent_dim=6, split=ContiguousSplit(n_partitions=3), seed=0)
        sizes = model.partition_sizes

        assert len(sizes) == 3
        assert len(set(sizes)) > 1  # not all partitions are the same size
        assert sum(sizes) == 100

        x = torch.randn(3, 100, 10)
        out = model(x)
        assert out.denoised.shape == (3, 100, 10)
        for idx, recon, target, z in zip(out.indices, out.recons, out.targets, out.latents):
            n_c = idx.numel()
            assert recon.shape == (3 * 10, n_c)
            assert target.shape == (3 * 10, n_c)
            assert z.shape == (3 * 10, 6)


class TestJAE1GradientFlow:
    """Tests that gradients reach every parameter."""

    def test_backward_populates_all_grads(self):
        model = JAE1(input_dim=96, latent_dim=12, seed=0)
        x = torch.randn(4, 96, 16)
        out = model(x)
        loss = model.loss(out)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"no grad for {name}"
            assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"


class TestJAE1SplitIsFixed:
    """Tests that the sampled channel split never changes."""

    def test_split_identical_across_forward_calls(self):
        model = JAE1(input_dim=96, latent_dim=12, seed=0)
        x1 = torch.randn(4, 96, 16)
        x2 = torch.randn(2, 96, 8)

        out1 = model(x1)
        out2 = model(x2)

        for idx1, idx2 in zip(out1.indices, out2.indices):
            assert torch.equal(idx1, idx2)

    def test_same_seed_gives_same_split_across_instances(self):
        model_a = JAE1(input_dim=96, latent_dim=12, seed=42)
        model_b = JAE1(input_dim=96, latent_dim=12, seed=42)

        for i in range(model_a.n_partitions):
            assert torch.equal(model_a._indices(i), model_b._indices(i))

    def test_different_seed_gives_different_split(self):
        model_a = JAE1(input_dim=96, latent_dim=12, seed=0)
        model_b = JAE1(input_dim=96, latent_dim=12, seed=1)

        same = all(
            torch.equal(model_a._indices(i), model_b._indices(i))
            for i in range(model_a.n_partitions)
        )
        assert not same


class TestJAE1Loss:
    """Tests for the JAE1.loss unweighted 3-term objective."""

    def test_perfectly_aligned_latents_contribute_near_zero_align_term(self):
        model = JAE1(input_dim=12, latent_dim=4, seed=0)

        z_shared = torch.randn(20, 4)
        recon = torch.zeros(20, 6)
        target = torch.zeros(20, 6)

        out = JAE1Output(
            denoised=torch.zeros(2, 12, 10),
            latents=[z_shared, z_shared.clone()],
            recons=[recon, recon.clone()],
            targets=[target, target.clone()],
            indices=[torch.arange(6), torch.arange(6, 12)],
        )

        loss = model.loss(out, latent_weight=1.0)
        assert loss.item() < 1e-6

    def test_misaligned_latents_and_recon_error_both_contribute(self):
        model = JAE1(input_dim=12, latent_dim=4, seed=0)

        out = JAE1Output(
            denoised=torch.zeros(2, 12, 10),
            latents=[torch.zeros(20, 4), torch.ones(20, 4)],
            recons=[torch.zeros(20, 6), torch.zeros(20, 6)],
            targets=[torch.ones(20, 6), torch.zeros(20, 6)],
            indices=[torch.arange(6), torch.arange(6, 12)],
        )

        loss = model.loss(out, latent_weight=1.0)
        expected_recon = F.mse_loss(torch.zeros(20, 6), torch.ones(20, 6)) + F.mse_loss(
            torch.zeros(20, 6), torch.zeros(20, 6)
        )
        expected_align = F.mse_loss(torch.zeros(20, 4), torch.ones(20, 4))
        assert torch.isclose(loss, expected_recon + expected_align, atol=1e-6)


class TestJAE1CollapseResistance:
    """Smoke test guarding against the original single-linear-layer collapse bug."""

    def test_trains_and_denoised_output_does_not_collapse(self):
        torch.manual_seed(0)

        n_samples, n_channels, t_len, true_latent_dim = 200, 12, 16, 3

        # Non-negative low-rank-plus-noise synthetic "firing rate" data: a
        # shared low-D latent, linearly mixed to more channels, plus
        # independent per-channel noise, matching the non-negative decoder
        # output activation (ReLU, for firing rates) used inside JAE1.
        latents_true = torch.rand(n_samples, true_latent_dim, t_len)
        mix = torch.rand(n_channels, true_latent_dim) + 0.5
        clean = torch.einsum("cd,ndt->nct", mix, latents_true)
        noise = 0.05 * torch.randn(n_samples, n_channels, t_len)
        x_noisy = (clean + noise).clamp_min(0.0)

        model = JAE1(input_dim=n_channels, latent_dim=4, hidden=(32, 16), seed=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        n_epochs = 150
        first_loss = None
        last_loss = None
        model.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            out = model(x_noisy)
            loss = model.loss(out)
            if epoch == 0:
                first_loss = loss.item()
            loss.backward()
            optimizer.step()
            last_loss = loss.item()

        assert first_loss is not None and last_loss is not None
        assert last_loss < 0.6 * first_loss, (
            f"loss did not meaningfully decrease: epoch1={first_loss:.4f}, final={last_loss:.4f}"
        )

        model.eval()
        with torch.no_grad():
            out = model(x_noisy)

        # (a) denoised output must retain real temporal variance per channel,
        # not collapse to a near-constant value over time.
        std_over_time = out.denoised.std(dim=2).mean().item()
        assert std_over_time > 0.01, f"denoised output looks collapsed: std={std_over_time:.5f}"

        # (b) every partition's latent must retain real variance across
        # samples/timepoints, not collapse to a single point.
        for z in out.latents:
            z_var = z.std(dim=0).mean().item()
            assert z_var > 1e-3, f"latent looks collapsed: std={z_var:.6f}"
