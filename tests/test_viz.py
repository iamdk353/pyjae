"""Tests for pyjae.viz visualization utilities."""

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
import torch

from pyjae.viz import (
    plot_channel_splits,
    plot_denoising_comparison,
    plot_latent_manifold,
    plot_per_channel_vaf,
    plot_training_history,
)


@pytest.fixture(autouse=True)
def close_plots():
    """Ensure matplotlib figures are closed after each test."""
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


def test_plot_denoising_comparison_numpy():
    clean = np.random.randn(5, 8, 50).astype(np.float32)
    noisy = clean + 0.5 * np.random.randn(5, 8, 50).astype(np.float32)
    pred_jae = clean + 0.1 * np.random.randn(5, 8, 50).astype(np.float32)

    fig, axes = plot_denoising_comparison(
        clean, noisy, {"JAE": pred_jae}, channels=[0, 1], time_range=(0, 30)
    )
    assert fig is not None
    assert len(axes) == 2


def test_plot_denoising_comparison_tensor():
    clean = torch.randn(8, 50)
    noisy = clean + 0.5 * torch.randn(8, 50)
    pred_pca = clean + 0.2 * torch.randn(8, 50)

    fig, axes = plot_denoising_comparison(clean, noisy, {"PCA": pred_pca}, n_channels_to_plot=3)
    assert fig is not None
    assert len(axes) == 3


def test_plot_latent_manifold_2d():
    latents = np.random.randn(100, 2)
    fig, ax = plot_latent_manifold(latents, dim=2, title="2D Latents Test")
    assert fig is not None
    assert ax is not None


def test_plot_latent_manifold_3d():
    latents = torch.randn(10, 50, 4)  # (N, T, D)
    color_by = torch.arange(50)
    fig, ax = plot_latent_manifold(latents, color_by=color_by, dim=3, title="3D Latents Test")
    assert fig is not None
    assert ax is not None


def test_plot_per_channel_vaf_bars():
    scores = {"JAE1": 0.85, "PCA": 0.65, "FA": 0.70}
    fig, ax = plot_per_channel_vaf(scores)
    assert fig is not None
    assert ax is not None


def test_plot_per_channel_vaf_lines():
    scores = {
        "JAE1": np.array([0.9, 0.85, 0.88, 0.92]),
        "PCA": np.array([0.7, 0.65, 0.68, 0.72]),
    }
    fig, ax = plot_per_channel_vaf(scores)
    assert fig is not None
    assert ax is not None


def test_plot_channel_splits():
    view_indices = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    fig, ax = plot_channel_splits(view_indices, n_channels=6)
    assert fig is not None
    assert ax is not None


def test_plot_training_history():
    history = {
        "total_loss": [1.0, 0.8, 0.6, 0.4],
        "recon_loss": [0.8, 0.6, 0.4, 0.3],
    }
    fig, ax = plot_training_history(history)
    assert fig is not None
    assert ax is not None
