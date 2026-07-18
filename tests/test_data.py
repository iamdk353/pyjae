"""Unit tests for pyjae.data: simulator, evaluation controls, and splits."""

import numpy as np
import torch

from pyjae.data import (
    make_noise_only,
    phase_shuffle,
    simulate_neural_data,
    simulate_paired_noise,
    train_val_test_split,
)


class TestSimulateNeuralDataShapes:
    """Shape, dtype, and reproducibility contracts."""

    def test_shapes_and_dtype(self):
        clean, noisy, info = simulate_neural_data(
            n_samples=20, n_channels=12, n_timepoints=32, latent_dim=4, seed=0
        )
        assert clean.shape == (20, 12, 32)
        assert noisy.shape == (20, 12, 32)
        assert clean.dtype == torch.float32
        assert noisy.dtype == torch.float32
        assert info["W"].shape == (12, 4)
        assert info["latents"].shape == (20, 4, 32)
        assert info["latent_dim"] == 4

    def test_info_dict_keys(self):
        _, _, info = simulate_neural_data(n_samples=10, latent_dim=3, seed=0)
        for key in (
            "W",
            "latents",
            "latent_dim",
            "alpha",
            "snr_db_per_channel",
            "noise_std_per_channel",
        ):
            assert key in info

    def test_reproducible_with_same_seed(self):
        clean1, noisy1, _ = simulate_neural_data(n_samples=15, latent_dim=3, seed=123)
        clean2, noisy2, _ = simulate_neural_data(n_samples=15, latent_dim=3, seed=123)
        assert torch.allclose(clean1, clean2)
        assert torch.allclose(noisy1, noisy2)

    def test_different_seed_differs(self):
        clean1, noisy1, _ = simulate_neural_data(n_samples=15, latent_dim=3, seed=1)
        clean2, noisy2, _ = simulate_neural_data(n_samples=15, latent_dim=3, seed=2)
        assert not torch.allclose(clean1, clean2)
        assert not torch.allclose(noisy1, noisy2)


class TestLatentDiversity:
    """Latents must span latent_dim, not collapse to one shared waveform."""

    def test_smooth_dynamics_full_rank(self):
        _, _, info = simulate_neural_data(
            n_samples=5, n_channels=20, n_timepoints=128, latent_dim=6, dynamics="smooth", seed=0
        )
        latents = info["latents"].numpy().astype(np.float64)
        for i in range(latents.shape[0]):
            rank = np.linalg.matrix_rank(latents[i])
            assert rank == 6, f"sample {i} latent rank {rank} != latent_dim 6"

    def test_iid_dynamics_full_rank(self):
        _, _, info = simulate_neural_data(
            n_samples=5, n_channels=20, n_timepoints=128, latent_dim=6, dynamics="iid", seed=0
        )
        latents = info["latents"].numpy().astype(np.float64)
        for i in range(latents.shape[0]):
            rank = np.linalg.matrix_rank(latents[i])
            assert rank == 6, f"sample {i} latent rank {rank} != latent_dim 6"

    def test_latent_dims_not_identical(self):
        """Different latent dimensions within a sample must not share a shape."""
        _, _, info = simulate_neural_data(
            n_samples=3, n_channels=10, n_timepoints=64, latent_dim=5, dynamics="smooth", seed=0
        )
        latents = info["latents"].numpy()
        sample0 = latents[0]  # (latent_dim, n_timepoints)
        for d in range(1, sample0.shape[0]):
            assert not np.allclose(sample0[0], sample0[d]), "latent dims must not be identical"

    def test_samples_not_identical(self):
        """Different samples must not share the same latent trajectory."""
        _, _, info = simulate_neural_data(
            n_samples=3, n_channels=10, n_timepoints=64, latent_dim=4, dynamics="smooth", seed=0
        )
        latents = info["latents"].numpy()
        assert not np.allclose(latents[0], latents[1])


class TestSNRAccuracy:
    """Achieved SNR must track the requested snr_db within tolerance."""

    def test_achieved_snr_within_tolerance(self):
        for target_snr in (0.0, 5.0, 10.0):
            _, _, info = simulate_neural_data(
                n_samples=200,
                n_channels=16,
                n_timepoints=128,
                latent_dim=6,
                snr_db=target_snr,
                seed=42,
            )
            achieved = float(np.mean(info["snr_db_per_channel"]))
            assert abs(achieved - target_snr) < 1.5, (
                f"target {target_snr} dB, achieved mean {achieved:.2f} dB"
            )

    def test_per_channel_snr_all_close(self):
        target_snr = 5.0
        _, _, info = simulate_neural_data(
            n_samples=200, n_channels=16, n_timepoints=128, latent_dim=6, snr_db=target_snr, seed=7
        )
        per_channel = info["snr_db_per_channel"]
        assert np.all(np.abs(per_channel - target_snr) < 3.0)


class TestNonlinearity:
    """Eq1 nonlinearity should meaningfully change the clean signal."""

    def test_nonlinear_vs_linear_differ(self):
        clean_lin, _, _ = simulate_neural_data(n_samples=20, latent_dim=4, nonlinear=False, seed=0)
        clean_nonlin, _, _ = simulate_neural_data(
            n_samples=20, latent_dim=4, nonlinear=True, alpha=16.0, seed=0
        )
        assert not torch.allclose(clean_lin, clean_nonlin)

    def test_different_alpha_differ(self):
        clean_a, _, _ = simulate_neural_data(
            n_samples=20, latent_dim=4, nonlinear=True, alpha=4.0, seed=0
        )
        clean_b, _, _ = simulate_neural_data(
            n_samples=20, latent_dim=4, nonlinear=True, alpha=16.0, seed=0
        )
        assert not torch.allclose(clean_a, clean_b)


class TestPerChannelNormalization:
    """Per-channel max-normalization (not global) before the nonlinearity."""

    def test_pre_nonlinear_channel_max_is_one(self):
        # nonlinear=False exposes the normalized-but-not-yet-nonlinear signal.
        clean, _, _ = simulate_neural_data(
            n_samples=30, n_channels=10, n_timepoints=64, latent_dim=4, nonlinear=False, seed=0
        )
        channel_max = clean.numpy().max(axis=(0, 2))
        np.testing.assert_allclose(channel_max, np.ones(10), atol=1e-5)

    def test_channel_maxes_are_not_uniform_scale_only(self):
        """Different channels get independently-fit scale (not one global max)."""
        clean, _, _ = simulate_neural_data(
            n_samples=30, n_channels=10, n_timepoints=64, latent_dim=4, nonlinear=False, seed=0
        )
        # Every channel's own max is ~1: this could only be true for ALL
        # channels simultaneously under per-channel (not global) normalization.
        channel_max = clean.numpy().max(axis=(0, 2))
        assert np.all(channel_max > 0.999) and np.all(channel_max < 1.001)

    def test_sane_value_range_nonlinear(self):
        clean, noisy, _ = simulate_neural_data(n_samples=20, latent_dim=4, seed=0)
        assert torch.isfinite(clean).all()
        assert torch.isfinite(noisy).all()
        assert clean.max() <= 1.01
        assert clean.min() >= -1.01


class TestSimulatePairedNoise:
    """Shared clean/latents, independent noise draws, for cvPCA."""

    def test_shapes_and_repeats(self):
        clean, noisy_list, info = simulate_paired_noise(
            n_samples=10, n_channels=8, n_timepoints=16, latent_dim=3, n_repeats=3, seed=5
        )
        assert clean.shape == (10, 8, 16)
        assert len(noisy_list) == 3
        for noisy in noisy_list:
            assert noisy.shape == (10, 8, 16)
        assert info["n_repeats"] == 3

    def test_noise_draws_are_independent(self):
        _, noisy_list, _ = simulate_paired_noise(
            n_samples=10, n_channels=8, n_timepoints=16, latent_dim=3, n_repeats=2, seed=5
        )
        assert not torch.allclose(noisy_list[0], noisy_list[1])

    def test_reproducible_with_same_seed(self):
        clean1, noisy_list1, _ = simulate_paired_noise(
            n_samples=10, n_channels=8, n_timepoints=16, latent_dim=3, n_repeats=2, seed=9
        )
        clean2, noisy_list2, _ = simulate_paired_noise(
            n_samples=10, n_channels=8, n_timepoints=16, latent_dim=3, n_repeats=2, seed=9
        )
        assert torch.allclose(clean1, clean2)
        for a, b in zip(noisy_list1, noisy_list2, strict=True):
            assert torch.allclose(a, b)


class TestPhaseShuffle:
    """Theiler surrogate: preserves per-channel power spectrum, changes waveform."""

    def test_preserves_power_spectrum(self):
        x = torch.randn(4, 6, 64)
        shuffled = phase_shuffle(x, seed=0)

        mag_orig = torch.fft.fft(x, dim=-1).abs()
        mag_shuffled = torch.fft.fft(shuffled, dim=-1).abs()
        assert torch.allclose(mag_orig, mag_shuffled, atol=1e-3, rtol=1e-3)

    def test_changes_time_domain_signal(self):
        x = torch.randn(4, 6, 64)
        shuffled = phase_shuffle(x, seed=0)
        assert not torch.allclose(x, shuffled)

    def test_odd_timepoints(self):
        x = torch.randn(3, 5, 33)
        shuffled = phase_shuffle(x, seed=1)
        assert shuffled.shape == x.shape
        mag_orig = torch.fft.fft(x, dim=-1).abs()
        mag_shuffled = torch.fft.fft(shuffled, dim=-1).abs()
        assert torch.allclose(mag_orig, mag_shuffled, atol=1e-3, rtol=1e-3)

    def test_reproducible_with_seed(self):
        x = torch.randn(4, 6, 64)
        s1 = phase_shuffle(x, seed=42)
        s2 = phase_shuffle(x, seed=42)
        assert torch.allclose(s1, s2)


class TestMakeNoiseOnly:
    """Zero-signal negative control with matched per-channel noise std."""

    def test_clean_is_zero_noisy_is_nonzero(self):
        clean0, noisy = make_noise_only(
            (50, 10, 20), per_channel_noise_std=np.full(10, 0.5), seed=1
        )
        assert torch.equal(clean0, torch.zeros_like(clean0))
        assert not torch.equal(noisy, torch.zeros_like(noisy))
        assert clean0.shape == (50, 10, 20)
        assert noisy.shape == (50, 10, 20)

    def test_accepts_info_dict(self):
        _, _, info = simulate_neural_data(n_samples=20, n_channels=8, n_timepoints=16, seed=0)
        clean0, noisy = make_noise_only(info, info["noise_std_per_channel"], seed=2)
        assert clean0.shape == (20, 8, 16)
        assert noisy.shape == (20, 8, 16)

    def test_matches_requested_std(self):
        std = np.full(6, 2.0)
        _, noisy = make_noise_only((500, 6, 64), per_channel_noise_std=std, seed=3)
        achieved_std = noisy.numpy().std(axis=(0, 2))
        np.testing.assert_allclose(achieved_std, std, rtol=0.1)

    def test_bad_std_length_raises(self):
        try:
            make_noise_only((10, 6, 8), per_channel_noise_std=np.ones(4), seed=0)
            raised = False
        except ValueError:
            raised = True
        assert raised


class TestTrainValTestSplit:
    """Disjoint, exhaustive partitions matching requested fractions."""

    def test_partitions_disjoint_and_cover_all(self):
        x = torch.randn(100, 5, 7)
        splits = train_val_test_split(x, fracs=(0.7, 0.15, 0.15), seed=0)

        train_idx = set(splits["train"]["idx"].tolist())
        val_idx = set(splits["val"]["idx"].tolist())
        test_idx = set(splits["test"]["idx"].tolist())

        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)
        assert train_idx | val_idx | test_idx == set(range(100))

    def test_fraction_sizes_approximately_correct(self):
        x = torch.randn(200, 3, 4)
        splits = train_val_test_split(x, fracs=(0.7, 0.15, 0.15), seed=0)
        assert splits["train"]["idx"].shape[0] == 140
        assert splits["val"]["idx"].shape[0] == 30
        assert splits["test"]["idx"].shape[0] == 30

    def test_noisy_and_clean_indexed_consistently(self):
        x = torch.randn(50, 4, 6)
        clean = torch.randn(50, 4, 6)
        splits = train_val_test_split(x, clean=clean, fracs=(0.6, 0.2, 0.2), seed=1)

        for name in ("train", "val", "test"):
            idx = splits[name]["idx"]
            assert torch.equal(splits[name]["noisy"], x[idx])
            assert torch.equal(splits[name]["clean"], clean[idx])

    def test_clean_absent_when_not_given(self):
        x = torch.randn(30, 2, 3)
        splits = train_val_test_split(x, fracs=(0.7, 0.15, 0.15), seed=0)
        assert "clean" not in splits["train"]

    def test_reproducible_with_seed(self):
        x = torch.randn(40, 3, 5)
        s1 = train_val_test_split(x, fracs=(0.7, 0.15, 0.15), seed=99)
        s2 = train_val_test_split(x, fracs=(0.7, 0.15, 0.15), seed=99)
        assert torch.equal(s1["train"]["idx"], s2["train"]["idx"])
        assert torch.equal(s1["test"]["idx"], s2["test"]["idx"])

    def test_bad_fracs_raises(self):
        x = torch.randn(10, 2, 2)
        try:
            train_val_test_split(x, fracs=(0.5, 0.3, 0.3), seed=0)
            raised = False
        except ValueError:
            raised = True
        assert raised
