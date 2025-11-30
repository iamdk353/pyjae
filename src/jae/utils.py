"""
Utility functions for JAE package.
"""

import warnings
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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


def simulate_neural_data(
    n_samples=500,
    n_channels=96,
    n_timepoints=128,
    latent_dim=6,
    snr_db=5.0,
    nonlinear=True,
    alpha=2.0,
    smoothing_sigma=2.0,
    seed=None,
):
    """
    Generate simulated neural data following Altan et al. (2021), Fig 1.

    The procedure:
    1. Sample d latent signals from Poisson-like firing rate distribution
    2. Smooth with Gaussian kernel
    3. Linear embedding via mixing matrix W (Gaussian entries)
    4. Scale each channel to [0, 1]
    5. Optional nonlinear embedding via exponential (Eq 1)
    6. Add Gaussian noise with specified SNR

    Parameters
    ----------
    n_samples : int, default=500
        Number of samples (trials).
    n_channels : int, default=96
        Number of recording channels (N in paper).
    n_timepoints : int, default=128
        Timepoints per sample (M in paper).
    latent_dim : int, default=6
        Intrinsic dimensionality (d in paper).
    snr_db : float, default=5.0
        Signal-to-noise ratio in dB.
    nonlinear : bool, default=True
        Apply exponential nonlinearity (Eq 1).
    alpha : float, default=2.0
        Nonlinearity parameter (higher = more nonlinear).
    smoothing_sigma : float, default=2.0
        Gaussian smoothing sigma (in samples, ~50ms at 25Hz).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    clean : Tensor, shape (n_samples, n_channels, n_timepoints)
        Noise-free signals (ground truth).
    noisy : Tensor, shape (n_samples, n_channels, n_timepoints)
        Noisy signals.
    info : dict
        Metadata including SNR, latent_dim, etc.

    Examples
    --------
    >>> clean, noisy, info = simulate_neural_data(n_samples=100, snr_db=5.0)
    >>> print(f"SNR: {info['snr_db']:.1f} dB")
    """
    if seed is not None:
        np.random.seed(seed)

    # Step 1: Generate latent signals from Poisson-like distribution
    # Simulate firing rates (Poisson parameter lambda ~ exponential distribution)
    # Sample rates typically 0-50 Hz, peaked at low values
    latents = np.zeros((n_samples, latent_dim, n_timepoints))

    for i in range(n_samples):
        for d in range(latent_dim):
            # Sample from exponential-like distribution (mimics firing rate distribution)
            # Then add temporal structure
            base_rate = np.random.exponential(scale=10.0)  # Mean firing rate
            # Add slow temporal modulation
            t = np.linspace(0, 4 * np.pi, n_timepoints)
            freq = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2 * np.pi)
            modulation = 0.5 + 0.5 * np.sin(freq * t + phase)
            latents[i, d, :] = base_rate * modulation

            # Add Poisson-like variability
            latents[i, d, :] += np.random.poisson(lam=2.0, size=n_timepoints)

    # Step 2: Smooth with Gaussian kernel
    if smoothing_sigma > 0:
        for i in range(n_samples):
            for d in range(latent_dim):
                latents[i, d, :] = gaussian_filter1d(latents[i, d, :], sigma=smoothing_sigma)

    # Step 3: Linear embedding via mixing matrix W
    # W has entries from N(0, 1) as per paper
    W = np.random.randn(n_channels, latent_dim)

    # Project: X = latents @ W.T
    # Reshape for efficient computation
    latents_flat = latents.transpose(0, 2, 1).reshape(n_samples * n_timepoints, latent_dim)
    X_flat = latents_flat @ W.T
    X = X_flat.reshape(n_samples, n_timepoints, n_channels).transpose(0, 2, 1)

    # Step 4: Scale each channel to [0, 1]
    for ch in range(n_channels):
        ch_min = X[:, ch, :].min()
        ch_max = X[:, ch, :].max()
        if ch_max > ch_min:
            X[:, ch, :] = (X[:, ch, :] - ch_min) / (ch_max - ch_min)
        else:
            X[:, ch, :] = 0.5

    # Step 5: Nonlinear embedding (Eq 1 from paper)
    if nonlinear and alpha > 0:
        # x_nonlin = (exp(alpha * x) - 1) / (exp(alpha) - 1)
        # This maps [0,1] -> [0,1] with controlled nonlinearity
        X = (np.exp(alpha * X) - 1) / (np.exp(alpha) - 1)

    clean = X.copy()

    # Step 6: Add Gaussian noise with specified SNR
    signal_power = np.mean(clean**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power)

    noise = np.random.randn(*clean.shape) * noise_std
    noisy = clean + noise

    # Clip to valid range (neural signals are non-negative)
    noisy = np.clip(noisy, 0, None)

    # Compute actual SNR
    actual_snr = 10 * np.log10(signal_power / np.mean((clean - noisy) ** 2 + 1e-10))

    info = {
        "snr_db": float(actual_snr),
        "latent_dim": latent_dim,
        "n_channels": n_channels,
        "n_timepoints": n_timepoints,
        "n_samples": n_samples,
        "nonlinear": nonlinear,
        "alpha": alpha if nonlinear else 0,
    }

    return (
        torch.tensor(clean, dtype=torch.float32),
        torch.tensor(noisy, dtype=torch.float32),
        info,
    )


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
