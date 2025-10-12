"""
Utility functions for JAE package.

This module provides helper functions for device management, data generation,
and evaluation metrics.
"""

import warnings
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def get_device(use_gpu=True, device_id=None, verbose=True):
    """
    Automatically detect and configure the compute device (GPU/CPU).

    This function checks for CUDA availability and returns an appropriate device.
    It also prints helpful information about GPU status when verbose=True.

    Args:
        use_gpu (bool, optional): Whether to use GPU if available. Default: True.
        device_id (int or str, optional): Specific device ID (e.g., 0 for 'cuda:0').
            If None, uses default CUDA device. Default: None.
        verbose (bool, optional): Whether to print device information. Default: True.

    Returns:
        torch.device: The configured device object.

    Example:
        >>> device = get_device(use_gpu=True, verbose=True)
        Using device: cuda:0 (NVIDIA GeForce RTX 4090)
        >>> print(device)  # cuda:0
    """
    if use_gpu and torch.cuda.is_available():
        if device_id is not None:
            device = torch.device(f'cuda:{device_id}')
        else:
            device = torch.device('cuda')

        if verbose:
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            print(f"Using device: {device} ({device_name})")
            print(f"GPU Memory: {torch.cuda.get_device_properties(current_device).total_memory / 1024**3:.1f} GB")

    elif use_gpu and not torch.cuda.is_available():
        device = torch.device('cpu')
        if verbose:
            warnings.warn(
                "GPU requested but CUDA is not available. Falling back to CPU. "
                "For GPU support, ensure PyTorch is installed with CUDA: "
                "https://pytorch.org/get-started/locally/",
                UserWarning
            )
            print("Using device: cpu")
    else:
        device = torch.device('cpu')
        if verbose:
            print("Using device: cpu")

    return device


def get_gpu_memory_info():
    """
    Get current GPU memory usage information.

    Returns:
        dict: Dictionary containing memory statistics in GB, or None if CUDA unavailable.
            Keys: 'allocated', 'reserved', 'total'

    Example:
        >>> if torch.cuda.is_available():
        ...     mem_info = get_gpu_memory_info()
        ...     print(f"Allocated: {mem_info['allocated']:.2f} GB")
    """
    if not torch.cuda.is_available():
        return None

    current_device = torch.cuda.current_device()
    return {
        'allocated': torch.cuda.memory_allocated(current_device) / 1024**3,
        'reserved': torch.cuda.memory_reserved(current_device) / 1024**3,
        'total': torch.cuda.get_device_properties(current_device).total_memory / 1024**3
    }


def generate_simulated_data(
    n_samples=1000,
    seq_len=128,
    n_channels=96,
    latent_dim=8,
    noise_level=0.7,
    seed=42
):
    """
    Generate simulated neural data for testing and validation.

    Creates synthetic neural signals by projecting low-dimensional sinusoidal
    latent dynamics onto high-dimensional space and adding noise. This simulates
    the hypothesis that neural recordings are noisy observations of low-dimensional
    latent dynamics.

    Args:
        n_samples (int, optional): Number of samples to generate. Default: 1000.
        seq_len (int, optional): Length of each time series. Default: 128.
        n_channels (int, optional): Number of channels (neurons). Default: 96.
        latent_dim (int, optional): True latent dimensionality. Default: 8.
        noise_level (float, optional): Relative noise level (std as fraction of
            signal power). Default: 0.7.
        seed (int, optional): Random seed for reproducibility. Default: 42.

    Returns:
        tuple: (clean_data, noisy_data) where both are torch.Tensors of shape
            (n_samples, n_channels, seq_len). clean_data contains the ground truth
            signals, noisy_data contains corrupted observations.

    Example:
        >>> clean, noisy = generate_simulated_data(n_samples=100, n_channels=96)
        >>> print(clean.shape, noisy.shape)  # torch.Size([100, 96, 128])
        >>> print(f"SNR: {calculate_snr(clean, noisy):.2f} dB")
    """
    np.random.seed(seed)

    # 1. Generate smooth low-dimensional latent signals (Sinusoids)
    t = np.linspace(0, 10 * np.pi, seq_len)
    latents = np.zeros((n_samples, latent_dim, seq_len))

    for i in range(n_samples):
        for d in range(latent_dim):
            freq = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2 * np.pi)
            # Add an offset to make the base signal positive
            latents[i, d, :] = np.sin(freq * t + phase) + np.random.uniform(1.5, 2.0)

    # 2. Project to high-dimensional space using a non-negative mixing matrix
    # This mimics how neurons (non-negative firing rates) capture underlying dynamics
    projection_matrix = np.random.rand(n_channels, latent_dim)

    # Reshape for efficient projection
    latents_reshaped = latents.transpose(0, 2, 1).reshape(n_samples * seq_len, latent_dim)
    clean_data_flat = latents_reshaped @ projection_matrix.T
    clean_data = clean_data_flat.reshape(n_samples, seq_len, n_channels).transpose(0, 2, 1)

    # 3. Add Noise scaled relative to signal power
    signal_power = np.mean(clean_data**2)
    noise_std = np.sqrt(signal_power) * noise_level
    noise = np.random.randn(*clean_data.shape) * noise_std
    noisy_data = clean_data + noise

    # Ensure non-negativity (clipping at zero)
    noisy_data = np.maximum(0, noisy_data)

    return (
        torch.tensor(clean_data, dtype=torch.float32),
        torch.tensor(noisy_data, dtype=torch.float32)
    )


def calculate_snr(signal, reconstruction):
    """
    Calculate Signal-to-Noise Ratio (SNR) in decibels.

    SNR measures the quality of reconstruction by comparing signal power to
    noise (error) power. Higher SNR indicates better reconstruction.

    Args:
        signal (torch.Tensor): Ground truth signal of any shape.
        reconstruction (torch.Tensor): Reconstructed signal of same shape as signal.

    Returns:
        float: SNR in decibels (dB). Returns inf if reconstruction is perfect.

    Example:
        >>> signal = torch.randn(100, 96, 128)
        >>> noisy = signal + 0.1 * torch.randn_like(signal)
        >>> snr = calculate_snr(signal, noisy)
        >>> print(f"SNR: {snr:.2f} dB")
    """
    signal_power = torch.mean(signal**2)
    noise_power = torch.mean((signal - reconstruction)**2)

    if noise_power < 1e-10:
        return float('inf')

    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()


def calculate_r2_score(y_true, y_pred):
    """
    Calculate the coefficient of determination (R² score).

    R² measures the proportion of variance in the true signal that is predictable
    from the reconstructed signal. R² = 1 indicates perfect prediction, R² = 0
    indicates prediction no better than the mean, and R² < 0 indicates prediction
    worse than the mean.

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth values of any shape.
        y_pred (torch.Tensor or np.ndarray): Predicted values of same shape as y_true.

    Returns:
        float: R² score.

    Example:
        >>> y_true = torch.randn(100, 96, 128)
        >>> y_pred = y_true + 0.1 * torch.randn_like(y_true)
        >>> r2 = calculate_r2_score(y_true, y_pred)
        >>> print(f"R²: {r2:.4f}")
    """
    # Convert to numpy if needed
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


def calculate_correlation(signal, reconstruction):
    """
    Calculate Pearson correlation coefficient between signals.

    Args:
        signal (torch.Tensor or np.ndarray): Ground truth signal.
        reconstruction (torch.Tensor or np.ndarray): Reconstructed signal.

    Returns:
        float: Pearson correlation coefficient (between -1 and 1).

    Example:
        >>> signal = torch.randn(100, 96, 128)
        >>> recon = signal + 0.1 * torch.randn_like(signal)
        >>> corr = calculate_correlation(signal, recon)
        >>> print(f"Correlation: {corr:.4f}")
    """
    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().numpy()
    if isinstance(reconstruction, torch.Tensor):
        reconstruction = reconstruction.detach().cpu().numpy()

    signal = signal.flatten()
    reconstruction = reconstruction.flatten()

    return np.corrcoef(signal, reconstruction)[0, 1]


def run_pca_baseline(noisy_data, latent_dim):
    """
    Run PCA denoising as a linear baseline method.

    PCA projects the data onto principal components and reconstructs from a
    reduced set of components, effectively performing linear denoising.

    Args:
        noisy_data (torch.Tensor): Noisy input data, shape (n_samples, n_channels, seq_len).
        latent_dim (int): Number of principal components to retain.

    Returns:
        torch.Tensor: Denoised data of same shape as input.

    Example:
        >>> _, noisy = generate_simulated_data(n_samples=100)
        >>> denoised = run_pca_baseline(noisy, latent_dim=8)
        >>> print(denoised.shape)  # torch.Size([100, 96, 128])
    """
    B, C, T = noisy_data.shape
    # Reshape for PCA (Samples*Time, Channels)
    noisy_flat = noisy_data.permute(0, 2, 1).reshape(B * T, C).numpy()

    # Standardize data
    scaler = StandardScaler()
    noisy_scaled = scaler.fit_transform(noisy_flat)

    # Apply PCA
    pca = PCA(n_components=latent_dim)
    latents_pca = pca.fit_transform(noisy_scaled)
    recon_pca_flat = pca.inverse_transform(latents_pca)

    # Inverse transform standardization
    recon_pca_flat = scaler.inverse_transform(recon_pca_flat)

    # Ensure non-negativity
    recon_pca_flat[recon_pca_flat < 0] = 0

    # Reshape back to (B, C, T)
    recon_pca = torch.tensor(
        recon_pca_flat.reshape(B, T, C),
        dtype=torch.float32
    ).permute(0, 2, 1)

    return recon_pca


def validate_input_data(data, name="data"):
    """
    Validate input data format for JAE models.

    Args:
        data (np.ndarray or torch.Tensor): Input data to validate.
        name (str, optional): Name of the data for error messages. Default: "data".

    Raises:
        TypeError: If data is not numpy array or torch tensor.
        ValueError: If data doesn't have 3 dimensions or contains invalid values.

    Returns:
        torch.Tensor: Validated data as torch tensor.

    Example:
        >>> data = np.random.randn(10, 96, 128)
        >>> validated = validate_input_data(data, name="training_data")
    """
    # Check type
    if not isinstance(data, (np.ndarray, torch.Tensor)):
        raise TypeError(
            f"{name} must be a numpy array or torch tensor, got {type(data)}"
        )

    # Convert to tensor if needed
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()

    # Check dimensions
    if data.ndim != 3:
        raise ValueError(
            f"{name} must be 3-dimensional (n_samples, n_channels, seq_len), "
            f"got shape {data.shape}"
        )

    # Check for invalid values
    if torch.isnan(data).any():
        raise ValueError(f"{name} contains NaN values")

    if torch.isinf(data).any():
        raise ValueError(f"{name} contains infinite values")

    return data


def set_seed(seed):
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed (int): Random seed value.

    Example:
        >>> set_seed(42)
        >>> # All subsequent random operations will be reproducible
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

