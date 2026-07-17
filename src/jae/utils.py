"""
Utility functions for JAE package.
"""

import warnings

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_device(use_gpu=True, device_id=None, verbose=True):
    """
    Get compute device (GPU/CPU).

    Parameters
    ----------
    use_gpu : bool, default=True
        Use GPU if available.
    device_id : int, optional
        Specific GPU device ID.
    verbose : bool, default=True
        Print device info.

    Returns
    -------
    device : torch.device
    """
    if use_gpu and torch.cuda.is_available():
        if device_id is not None:
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cuda")

        if verbose:
            name = torch.cuda.get_device_name(torch.cuda.current_device())
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"Using device: {device} ({name}, {mem:.1f} GB)")

    elif use_gpu and not torch.cuda.is_available():
        device = torch.device("cpu")
        if verbose:
            warnings.warn(
                "GPU requested but CUDA unavailable. Using CPU. "
                "For GPU support, install PyTorch with CUDA.",
                UserWarning,
            )
            print("Using device: cpu")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Using device: cpu")

    return device


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_vaf(y_true, y_pred):
    """
    Calculate Variance Accounted For (VAF / R²).

    This is the primary evaluation metric from the paper:
    VAF = 1 - (residual variance / total variance)

    Parameters
    ----------
    y_true : array-like
        Ground truth (noise-free) signals.
    y_pred : array-like
        Predicted (denoised) signals.

    Returns
    -------
    vaf : float
        Variance Accounted For (same as R² score).
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return 1 - (ss_res / ss_tot)


def calculate_snr(signal, reconstruction):
    """
    Calculate Signal-to-Noise Ratio in dB.

    Parameters
    ----------
    signal : array-like
        Ground truth signal.
    reconstruction : array-like
        Reconstructed signal.

    Returns
    -------
    snr : float
        SNR in decibels.
    """
    if isinstance(signal, np.ndarray):
        signal = torch.from_numpy(signal)
    if isinstance(reconstruction, np.ndarray):
        reconstruction = torch.from_numpy(reconstruction)

    signal_power = torch.mean(signal**2)
    noise_power = torch.mean((signal - reconstruction) ** 2)

    if noise_power < 1e-10:
        return float("inf")

    return (10 * torch.log10(signal_power / noise_power)).item()


def run_pca_baseline(noisy_data, latent_dim):
    """
    PCA denoising baseline.

    Projects data onto top principal components and reconstructs.

    Parameters
    ----------
    noisy_data : Tensor, shape (n_samples, n_channels, n_timepoints)
        Noisy input.
    latent_dim : int
        Number of components to retain.

    Returns
    -------
    denoised : Tensor
        PCA-denoised data.
    """
    if isinstance(noisy_data, torch.Tensor):
        noisy_data = noisy_data.numpy()

    B, C, T = noisy_data.shape
    noisy_flat = noisy_data.transpose(0, 2, 1).reshape(B * T, C)

    scaler = StandardScaler()
    noisy_scaled = scaler.fit_transform(noisy_flat)

    pca = PCA(n_components=latent_dim)
    latents = pca.fit_transform(noisy_scaled)
    recon_flat = pca.inverse_transform(latents)
    recon_flat = scaler.inverse_transform(recon_flat)

    # Non-negativity
    recon_flat = np.clip(recon_flat, 0, None)

    recon = recon_flat.reshape(B, T, C).transpose(0, 2, 1)
    return torch.tensor(recon, dtype=torch.float32)


def validate_input_data(data, name="data"):
    """
    Validate input data format.

    Parameters
    ----------
    data : array-like
        Input data to validate.
    name : str
        Name for error messages.

    Returns
    -------
    data : Tensor
        Validated tensor.
    """
    if not isinstance(data, (np.ndarray, torch.Tensor)):
        raise TypeError(f"{name} must be numpy array or torch tensor")

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()

    if data.ndim != 3:
        raise ValueError(
            f"{name} must be 3D (n_samples, n_channels, n_timepoints), got shape {data.shape}"
        )

    if torch.isnan(data).any():
        raise ValueError(f"{name} contains NaN values")

    if torch.isinf(data).any():
        raise ValueError(f"{name} contains infinite values")

    return data
