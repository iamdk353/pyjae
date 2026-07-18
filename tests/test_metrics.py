"""Unit tests for pyjae.metrics: the non-gameable latent-quality metric panel."""

import numpy as np
import pytest
import torch

from pyjae.metrics import (
    alignment,
    collapse_report,
    effective_rank,
    invariance_ratio,
    lidar,
    nuisance_probe,
    participation_ratio,
    per_channel_vaf,
    uniformity,
    vaf,
)


def _rank1_matrix(n: int, d: int, seed: int = 0) -> np.ndarray:
    """A rank-1 (N, D) matrix: every row is a scalar multiple of one vector."""
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(d)
    scales = rng.standard_normal(n)  # varies so centering does not zero it out
    return np.outer(scales, direction)


def _isotropic_matrix(n: int, d: int, seed: int = 0) -> np.ndarray:
    """An isotropic Gaussian (N, D) matrix, N >> D."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d))


class TestEffectiveRank:
    def test_rank1_is_near_one(self):
        z = _rank1_matrix(n=200, d=20)
        erank = effective_rank(z)
        assert abs(erank - 1.0) < 0.1

    def test_isotropic_is_near_d(self):
        d = 20
        z = _isotropic_matrix(n=3000, d=d)
        erank = effective_rank(z)
        # SPEC: "Good: >0.7K"; isotropic Gaussian with N >> D should clear
        # this bar comfortably and stay within min(N, D) = D.
        assert erank > 0.7 * d
        assert erank <= d + 1e-6

    def test_accepts_torch_tensor(self):
        z_np = _isotropic_matrix(n=500, d=10)
        erank_np = effective_rank(z_np)
        erank_torch = effective_rank(torch.tensor(z_np, dtype=torch.float32))
        assert abs(erank_np - erank_torch) < 0.05


class TestParticipationRatio:
    def test_rank1_is_near_one(self):
        z = _rank1_matrix(n=200, d=20)
        pr = participation_ratio(z)
        assert abs(pr - 1.0) < 0.1

    def test_isotropic_is_near_d(self):
        d = 20
        z = _isotropic_matrix(n=3000, d=d)
        pr = participation_ratio(z)
        assert pr > 0.7 * d
        assert pr <= d + 1.0


class TestPerChannelVaf:
    def test_identical_signals_give_mean_one(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal((8, 12, 64))
        result = per_channel_vaf(y, y.copy())
        assert result["mean"] == pytest.approx(1.0)
        assert result["per_channel"].shape == (12,)
        assert np.allclose(result["per_channel"], 1.0)

    def test_per_channel_length_matches_channels(self):
        rng = np.random.default_rng(1)
        y_true = rng.standard_normal((5, 7, 30))
        y_pred = y_true + rng.standard_normal((5, 7, 30)) * 0.1
        result = per_channel_vaf(y_true, y_pred)
        assert result["per_channel"].shape == (7,)
        for key in ("mean", "weighted_mean", "median", "worst"):
            assert isinstance(result[key], float)

    def test_denoised_to_channel_mean_gives_zero_vaf(self):
        rng = np.random.default_rng(2)
        b, c, t = 10, 6, 40
        y_true = rng.standard_normal((b, c, t)) * 3.0 + 1.0
        channel_mean = y_true.mean(axis=(0, 2), keepdims=True)
        y_pred = np.broadcast_to(channel_mean, y_true.shape)
        result = per_channel_vaf(y_true, y_pred)
        assert np.allclose(result["per_channel"], 0.0, atol=1e-8)
        assert abs(result["mean"]) < 1e-8

    def test_accepts_torch_tensor(self):
        rng = np.random.default_rng(3)
        y_true = torch.tensor(rng.standard_normal((4, 5, 20)), dtype=torch.float32)
        y_pred = y_true.clone()
        result = per_channel_vaf(y_true, y_pred)
        assert np.allclose(result["per_channel"], 1.0, atol=1e-5)


class TestVaf:
    def test_identical_is_one(self):
        rng = np.random.default_rng(4)
        y = rng.standard_normal((100,))
        assert vaf(y, y.copy()) == pytest.approx(1.0)

    def test_constant_prediction_worse_than_perfect(self):
        rng = np.random.default_rng(5)
        y_true = rng.standard_normal((200,))
        y_pred_perfect = y_true.copy()
        y_pred_mean = np.full_like(y_true, y_true.mean())
        assert vaf(y_true, y_pred_perfect) > vaf(y_true, y_pred_mean)
        assert abs(vaf(y_true, y_pred_mean)) < 1e-8


class TestInvarianceRatio:
    def test_identical_nonzero_variance_gives_ir_near_one(self):
        rng = np.random.default_rng(6)
        z = rng.standard_normal((100, 8))
        result = invariance_ratio(z, z.copy())
        assert result["ir"] == pytest.approx(1.0, abs=1e-6)
        assert result["denom"] > 0.0

    def test_collapsed_constant_has_near_zero_denom(self):
        const_row = np.array([1.0, -2.0, 0.5, 3.0])
        z = np.tile(const_row, (50, 1))
        result = invariance_ratio(z, z.copy())
        assert result["denom"] < 1e-8

    def test_diverging_pairs_lower_ir(self):
        rng = np.random.default_rng(7)
        z1 = rng.standard_normal((200, 10))
        z2 = z1 + rng.standard_normal((200, 10)) * 5.0
        result = invariance_ratio(z1, z2)
        assert result["ir"] < 0.9


class TestLidar:
    def test_between_sample_variance_dominates(self):
        rng = np.random.default_rng(8)
        n, d, v = 60, 4, 5
        sample_means = rng.standard_normal((n, d)) * 5.0
        views = []
        for _ in range(v):
            noise = rng.standard_normal((n, d)) * 0.01
            views.append(sample_means + noise)
        val = lidar(views)
        assert val > 1.5

    def test_collapsed_views_give_lidar_near_one(self):
        n, d, v = 60, 4, 5
        constant = np.zeros((n, d))
        views = [constant.copy() for _ in range(v)]
        val = lidar(views)
        assert abs(val - 1.0) < 0.05

    def test_accepts_torch_tensor_views(self):
        rng = np.random.default_rng(9)
        n, d, v = 40, 3, 4
        sample_means = rng.standard_normal((n, d)) * 4.0
        views = [
            torch.tensor(sample_means + rng.standard_normal((n, d)) * 0.01, dtype=torch.float32)
            for _ in range(v)
        ]
        val = lidar(views)
        assert val > 1.0


class TestAlignmentUniformity:
    def test_collapsed_uniformity_is_worst_value(self):
        z = np.tile(np.array([1.0, 2.0, 3.0]), (50, 1))
        val = uniformity(z)
        assert abs(val) < 1e-6

    def test_spread_out_uniformity_is_strongly_negative(self):
        rng = np.random.default_rng(10)
        z = rng.standard_normal((200, 50))
        val = uniformity(z)
        assert val < -1.0

    def test_alignment_zero_for_identical_pairs(self):
        rng = np.random.default_rng(11)
        z = rng.standard_normal((100, 8))
        assert alignment(z, z.copy()) == pytest.approx(0.0, abs=1e-10)

    def test_alignment_positive_for_diverging_pairs(self):
        rng = np.random.default_rng(12)
        z1 = rng.standard_normal((100, 8))
        z2 = rng.standard_normal((100, 8))
        assert alignment(z1, z2) > 0.1

    def test_uniformity_subsamples_for_large_n(self):
        rng = np.random.default_rng(13)
        z = rng.standard_normal((2500, 16))
        val = uniformity(z)
        assert val < -0.1


class TestNuisanceProbe:
    def test_informative_labels_beat_chance(self):
        rng = np.random.default_rng(14)
        n_per_class, d, n_classes = 60, 6, 3
        chunks = []
        labels = []
        for k in range(n_classes):
            center = np.full(d, k * 10.0)
            chunks.append(rng.standard_normal((n_per_class, d)) * 0.5 + center)
            labels.extend([k] * n_per_class)
        z = np.concatenate(chunks, axis=0)
        labels = np.array(labels)
        acc = nuisance_probe(z, labels, n_splits=5)
        assert acc > 2.0 / n_classes

    def test_uninformative_labels_near_chance(self):
        rng = np.random.default_rng(15)
        n, d, n_classes = 180, 10, 3
        z = rng.standard_normal((n, d))
        labels = rng.integers(0, n_classes, size=n)
        acc = nuisance_probe(z, labels, n_splits=5)
        # Loose bound: chance is 1/3; allow generous slack for finite-sample noise.
        assert acc < (1.0 / n_classes) + 0.25


class TestCollapseReport:
    def test_collapsed_embedding_flagged(self):
        n, d, v = 80, 10, 3
        const_row = np.linspace(-1.0, 1.0, d)
        z = np.tile(const_row, (n, 1))
        views = [z.copy() for _ in range(v)]
        report = collapse_report(z, views=views, Z_pair=(z.copy(), z.copy()))
        assert report["collapsed"] is True
        assert report["effective_rank"] <= 1.5
        assert len(report["reasons"]) > 0

    def test_healthy_isotropic_embedding_not_flagged(self):
        rng = np.random.default_rng(16)
        n, d, v = 300, 10, 3
        base = rng.standard_normal((n, d))
        views = [base + rng.standard_normal((n, d)) * 0.001 for _ in range(v)]
        report = collapse_report(base, views=views, Z_pair=(views[0], views[1]))
        assert report["collapsed"] is False
        assert report["effective_rank"] > 0.7 * d

    def test_report_contains_individual_metrics(self):
        rng = np.random.default_rng(17)
        n, d = 100, 6
        z = rng.standard_normal((n, d))
        labels = rng.integers(0, 2, size=n)
        report = collapse_report(z, Z_pair=(z, z + rng.standard_normal((n, d)) * 0.01))
        report_with_probe = collapse_report(
            z, Z_pair=(z, z + rng.standard_normal((n, d)) * 0.01), nuisance_labels=labels
        )
        for key in ("effective_rank", "participation_ratio", "uniformity", "collapsed"):
            assert key in report
        assert "nuisance_probe_accuracy" in report_with_probe
