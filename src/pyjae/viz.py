"""Visualization utilities for pyjae neural signal denoising and manifold learning."""

from typing import Any

import numpy as np
import torch


def _to_numpy(data: Any) -> np.ndarray:
    """Convert input (torch.Tensor or numpy.ndarray) to float32 NumPy array."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().astype(np.float32)
    return np.asarray(data, dtype=np.float32)


def _get_plt():
    """Lazily import matplotlib.pyplot with a user-friendly error message if missing."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        return plt
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for pyjae.viz visualization utilities. "
            "Install it via `pip install matplotlib` or `uv add matplotlib`."
        ) from err


def plot_denoising_comparison(
    clean: Any,
    noisy: Any,
    predictions: dict[str, Any],
    channels: list[int] | None = None,
    time_range: tuple[int, int] | None = None,
    n_channels_to_plot: int = 4,
    sample_idx: int = 0,
    figsize: tuple[float, float] = (12, 8),
    title: str = "Signal Denoising Comparison",
):
    """Plot multi-channel signal overlays comparing raw noisy signals, clean ground truth, and predictions.

    Parameters
    ----------
    clean : array-like or Tensor
        Clean ground-truth signal of shape (N, C, T) or (C, T).
    noisy : array-like or Tensor
        Noisy input signal of shape (N, C, T) or (C, T).
    predictions : dict[str, array-like or Tensor]
        Dictionary mapping method names (e.g., 'JAE1', 'PCA', 'FA') to predicted signals.
    channels : list of int, optional
        List of channel indices to plot. If None, plots the first `n_channels_to_plot`.
    time_range : tuple of (int, int), optional
        (start_time, end_time) slice for the time dimension.
    n_channels_to_plot : int, default=4
        Number of channels to display if `channels` is None.
    sample_idx : int, default=0
        Sample index to plot if input is 3D (N, C, T).
    figsize : tuple of float, default=(12, 8)
        Matplotlib figure size.
    title : str, default="Signal Denoising Comparison"
        Main figure title.

    Returns
    -------
    fig, axes : matplotlib.figure.Figure, np.ndarray of Axes
    """
    plt = _get_plt()

    clean_arr = _to_numpy(clean)
    noisy_arr = _to_numpy(noisy)

    if clean_arr.ndim == 3:
        clean_arr = clean_arr[sample_idx]
        noisy_arr = noisy_arr[sample_idx]

    pred_arrs = {}
    for name, pred in predictions.items():
        arr = _to_numpy(pred)
        if arr.ndim == 3:
            arr = arr[sample_idx]
        pred_arrs[name] = arr

    n_channels, n_time = clean_arr.shape

    if channels is None:
        channels = list(range(min(n_channels_to_plot, n_channels)))

    if time_range is None:
        t_start, t_end = 0, n_time
    else:
        t_start, t_end = time_range

    time_vec = np.arange(t_start, t_end)

    n_plots = len(channels)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    if n_plots == 1:
        axes = np.array([axes])

    for i, ch in enumerate(channels):
        ax = axes[i]
        ax.plot(
            time_vec,
            noisy_arr[ch, t_start:t_end],
            color="gray",
            alpha=0.4,
            label="Noisy" if i == 0 else "",
            linewidth=1.0,
        )
        ax.plot(
            time_vec,
            clean_arr[ch, t_start:t_end],
            color="black",
            linestyle="--",
            label="Clean Ground Truth" if i == 0 else "",
            linewidth=1.5,
        )

        colors = plt.cm.Set1(np.linspace(0, 1, max(len(pred_arrs), 1)))
        for j, (name, pred) in enumerate(pred_arrs.items()):
            ax.plot(
                time_vec,
                pred[ch, t_start:t_end],
                color=colors[j],
                label=name if i == 0 else "",
                linewidth=1.5,
            )

        ax.set_ylabel(f"Ch {ch}")
        ax.grid(True, linestyle=":", alpha=0.6)

    axes[-1].set_xlabel("Timepoint")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98))
    fig.tight_layout(rect=[0, 0, 0.85, 0.95])

    return fig, axes


def plot_latent_manifold(
    latents: Any,
    color_by: Any | None = None,
    dim: int = 2,
    sample_idx: int = 0,
    title: str = "Latent Manifold Representation",
    figsize: tuple[float, float] = (8, 6),
):
    """Plot 2D or 3D scatter / trajectory of low-dimensional latent embeddings.

    Parameters
    ----------
    latents : array-like or Tensor
        Latent tensor of shape (N, D), (N, T, D), or (D, T).
    color_by : array-like or Tensor, optional
        Values to color-code points by (e.g. time steps or condition labels).
    dim : int, default=2
        Dimensionality of plot (2 or 3).
    sample_idx : int, default=0
        Sample index if latents is 3D (N, T, D).
    title : str, default="Latent Manifold Representation"
        Figure title.
    figsize : tuple of float, default=(8, 6)
        Matplotlib figure size.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    plt = _get_plt()

    arr = _to_numpy(latents)
    if arr.ndim == 3:
        arr = arr[sample_idx]  # Shape (T, D)

    if arr.ndim == 1:
        arr = arr[:, np.newaxis]

    n_points, n_dims = arr.shape

    if color_by is not None:
        c_vals = _to_numpy(color_by)
        if c_vals.ndim > 1:
            c_vals = c_vals.ravel()
    else:
        c_vals = np.arange(n_points)

    fig = plt.figure(figsize=figsize)

    if dim == 3 and n_dims >= 3:
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], c=c_vals, cmap="viridis", alpha=0.8, s=25)
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color="gray", alpha=0.3, linewidth=0.8)
        ax.set_xlabel("Latent 1")
        ax.set_ylabel("Latent 2")
        ax.set_zlabel("Latent 3")
    else:
        ax = fig.add_subplot(111)
        sc = ax.scatter(arr[:, 0], arr[:, 1], c=c_vals, cmap="viridis", alpha=0.8, s=25)
        ax.plot(arr[:, 0], arr[:, 1], color="gray", alpha=0.3, linewidth=0.8)
        ax.set_xlabel("Latent 1")
        ax.set_ylabel("Latent 2" if n_dims > 1 else "Index")
        ax.grid(True, linestyle=":", alpha=0.6)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Time / Color Variable")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()

    return fig, ax


def plot_per_channel_vaf(
    vaf_scores: dict[str, Any],
    title: str = "Per-Channel Variance Accounted For (VAF)",
    figsize: tuple[float, float] = (10, 5),
):
    """Plot grouped bar charts or per-channel VAF distributions comparing models.

    Parameters
    ----------
    vaf_scores : dict[str, Any]
        Dictionary mapping method names to either scalar mean VAF, 1D array of per-channel VAFs,
        or a metrics dict containing 'per_channel' key.
    title : str, default="Per-Channel Variance Accounted For (VAF)"
        Figure title.
    figsize : tuple of float, default=(10, 5)
        Matplotlib figure size.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    plt = _get_plt()

    parsed_vafs = {}
    for name, score in vaf_scores.items():
        if isinstance(score, dict) and "per_channel" in score:
            parsed_vafs[name] = _to_numpy(score["per_channel"])
        elif isinstance(score, (list, np.ndarray, torch.Tensor)):
            arr = _to_numpy(score)
            if arr.ndim > 0:
                parsed_vafs[name] = arr
            else:
                parsed_vafs[name] = np.array([float(arr)])
        else:
            parsed_vafs[name] = np.array([float(score)])

    fig, ax = plt.subplots(figsize=figsize)

    has_multi_channel = any(len(arr) > 1 for arr in parsed_vafs.values())

    if has_multi_channel:
        for name, arr in parsed_vafs.items():
            if len(arr) > 1:
                ax.plot(arr, label=f"{name} (mean={np.mean(arr):.3f})", alpha=0.8, linewidth=1.5)
            else:
                ax.axhline(arr[0], linestyle="--", label=f"{name} ({arr[0]:.3f})", alpha=0.7)
        ax.set_xlabel("Channel Index")
        ax.set_ylabel("VAF (1.0 = Perfect)")
        ax.set_ylim(-0.1, 1.05)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right")
    else:
        names = list(parsed_vafs.keys())
        means = [float(np.mean(arr)) for arr in parsed_vafs.values()]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))

        bars = ax.bar(names, means, color=colors, alpha=0.85, edgecolor="black", width=0.5)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        ax.set_ylabel("Mean VAF")
        ax.set_ylim(-0.1, 1.05)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    ax.set_title(title, fontweight="bold")
    fig.tight_layout()

    return fig, ax


def plot_channel_splits(
    view_indices: list[Any],
    n_channels: int,
    title: str = "Channel Partition / View Assignment",
    figsize: tuple[float, float] = (8, 4),
):
    """Plot a binary grid visualization of channel splits across views.

    Parameters
    ----------
    view_indices : list of index arrays
        List of index arrays or tensors indicating channels in each view/partition.
    n_channels : int
        Total number of channels.
    title : str, default="Channel Partition / View Assignment"
        Figure title.
    figsize : tuple of float, default=(8, 4)
        Matplotlib figure size.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    plt = _get_plt()

    n_views = len(view_indices)
    grid = np.zeros((n_views, n_channels), dtype=np.float32)

    for v_idx, idxs in enumerate(view_indices):
        arr_idx = _to_numpy(idxs).astype(int)
        grid[v_idx, arr_idx] = 1.0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(grid, aspect="auto", cmap="Blues", vmin=0, vmax=1, origin="upper")

    ax.set_yticks(np.arange(n_views))
    ax.set_yticklabels([f"View {i + 1}" for i in range(n_views)])
    ax.set_xlabel("Channel Index")
    ax.set_title(title, fontweight="bold")

    fig.colorbar(
        im,
        ax=ax,
        ticks=[0, 1],
        format=plt.FuncFormatter(lambda x, _: "Active" if x == 1 else "Inactive"),
    )
    fig.tight_layout()

    return fig, ax


def plot_training_history(
    history: dict[str, list[float]],
    title: str = "Training & Loss History",
    figsize: tuple[float, float] = (8, 4),
):
    """Plot loss curves over epochs.

    Parameters
    ----------
    history : dict[str, list[float]]
        Dictionary mapping loss names (e.g. 'loss', 'recon_loss', 'latent_loss') to lists of epoch loss values.
    title : str, default="Training & Loss History"
        Figure title.
    figsize : tuple of float, default=(8, 4)
        Matplotlib figure size.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    plt = _get_plt()

    fig, ax = plt.subplots(figsize=figsize)

    for name, loss_vals in history.items():
        epochs = np.arange(1, len(loss_vals) + 1)
        ax.plot(epochs, loss_vals, label=name, linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()

    return fig, ax
