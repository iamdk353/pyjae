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

    # 2. Project to high-dimensional space with NONLINEAR mixing
    # This mimics realistic neural transformations
    projection_matrix_1 = np.random.randn(n_channels, latent_dim)
    projection_matrix_2 = np.random.randn(n_channels, latent_dim)
    projection_matrix_1 = projection_matrix_1 / (np.linalg.norm(projection_matrix_1, axis=1, keepdims=True) + 1e-8)
    projection_matrix_2 = projection_matrix_2 / (np.linalg.norm(projection_matrix_2, axis=1, keepdims=True) + 1e-8)

    # Reshape for efficient projection
    latents_reshaped = latents.transpose(0, 2, 1).reshape(n_samples * seq_len, latent_dim)
    
    # Nonlinear mixing: linear + tanh + polynomial
    linear_part = latents_reshaped @ projection_matrix_1.T
    nonlinear_part = np.tanh(latents_reshaped @ projection_matrix_2.T)
    
    poly_features = []
    for i in range(min(3, latent_dim)):
        for j in range(i+1, min(4, latent_dim)):
            poly_features.append(latents_reshaped[:, i] * latents_reshaped[:, j])
    
    if poly_features:
        poly_matrix = np.random.randn(n_channels, len(poly_features))
        poly_matrix = poly_matrix / (np.linalg.norm(poly_matrix, axis=0, keepdims=True) + 1e-8)
        poly_part = np.column_stack(poly_features) @ poly_matrix.T
        clean_data_flat = 0.5 * linear_part + 0.3 * nonlinear_part + 0.2 * poly_part
    else:
        clean_data_flat = 0.6 * linear_part + 0.4 * nonlinear_part
    
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


def generate_neural_data_with_transients(
    n_samples=1000,
    seq_len=128,
    n_channels=96,
    latent_dim=8,
    noise_level=0.5,
    n_transients=5,
    normalize=True,
    seed=42
):
    """
    Generate neural data with sharp transients (spikes) that benefit from U-Net.

    This scenario favors JAE2 (U-Net) over JAE1 (FC) because:
    - Sharp transients require temporal context
    - Skip connections preserve spike timing
    - FC networks process each timepoint independently

    Args:
        n_samples (int): Number of samples.
        seq_len (int): Sequence length.
        n_channels (int): Number of channels.
        latent_dim (int): Latent dimensionality.
        noise_level (float): Noise level (0-1).
        n_transients (int): Number of transient events per sample.
        normalize (bool): Whether to normalize data.
        seed (int): Random seed.

    Returns:
        tuple: (clean_data, noisy_data, snr_db)
    """
    np.random.seed(seed)

    # Generate smooth baseline
    t = np.linspace(0, 8 * np.pi, seq_len)
    latents = np.zeros((n_samples, latent_dim, seq_len))

    for i in range(n_samples):
        for d in range(latent_dim):
            # Smooth baseline
            freq = np.random.uniform(0.3, 1.0)
            phase = np.random.uniform(0, 2 * np.pi)
            latents[i, d, :] = np.sin(freq * t + phase)

            # Add sharp transients (spike-like events)
            for _ in range(n_transients):
                spike_time = np.random.randint(10, seq_len - 10)
                spike_width = np.random.randint(2, 5)
                spike_amp = np.random.uniform(2.0, 4.0)
                # Gaussian spike
                spike = spike_amp * np.exp(-0.5 * ((np.arange(seq_len) - spike_time) / spike_width) ** 2)
                latents[i, d, :] += spike

    # Project to high-dimensional space with NONLINEAR mixing
    # Create two projection matrices for nonlinear combination
    projection_matrix_1 = np.random.randn(n_channels, latent_dim)
    projection_matrix_2 = np.random.randn(n_channels, latent_dim)
    if normalize:
        # Normalize rows to prevent explosion
        projection_matrix_1 = projection_matrix_1 / (np.linalg.norm(projection_matrix_1, axis=1, keepdims=True) + 1e-8)
        projection_matrix_2 = projection_matrix_2 / (np.linalg.norm(projection_matrix_2, axis=1, keepdims=True) + 1e-8)

    latents_reshaped = latents.transpose(0, 2, 1).reshape(n_samples * seq_len, latent_dim)
    
    # Nonlinear mixing: mix linear + tanh(linear) + polynomial interactions
    linear_part = latents_reshaped @ projection_matrix_1.T
    nonlinear_part = np.tanh(latents_reshaped @ projection_matrix_2.T)
    
    # Add polynomial interactions (products of latent dimensions)
    poly_features = []
    for i in range(min(3, latent_dim)):
        for j in range(i+1, min(4, latent_dim)):
            poly_features.append(latents_reshaped[:, i] * latents_reshaped[:, j])
    
    if poly_features:
        poly_matrix = np.random.randn(n_channels, len(poly_features))
        if normalize:
            poly_matrix = poly_matrix / (np.linalg.norm(poly_matrix, axis=0, keepdims=True) + 1e-8)
        poly_part = np.column_stack(poly_features) @ poly_matrix.T
        # Combine: 50% linear + 30% nonlinear + 20% polynomial
        clean_data_flat = 0.5 * linear_part + 0.3 * nonlinear_part + 0.2 * poly_part
    else:
        # Combine: 60% linear + 40% nonlinear
        clean_data_flat = 0.6 * linear_part + 0.4 * nonlinear_part
    
    clean_data = clean_data_flat.reshape(n_samples, seq_len, n_channels).transpose(0, 2, 1)

    # Normalize to prevent activation explosions
    if normalize:
        # Z-score normalization per channel
        mean = clean_data.mean()
        std = clean_data.std() + 1e-8
        clean_data = (clean_data - mean) / std
        # Shift to positive range
        clean_data = clean_data - clean_data.min() + 0.5

    # Add heterogeneous Gaussian noise
    signal_power = np.mean(clean_data ** 2)
    noise_std = np.sqrt(signal_power) * noise_level
    noise = np.random.randn(*clean_data.shape) * noise_std
    noisy_data = clean_data + noise

    # Ensure non-negativity
    noisy_data = np.maximum(0, noisy_data)

    # Calculate actual SNR
    snr_db = 10 * np.log10(np.mean(clean_data ** 2) / np.mean((clean_data - noisy_data) ** 2))

    return (
        torch.tensor(clean_data, dtype=torch.float32),
        torch.tensor(noisy_data, dtype=torch.float32),
        float(snr_db)
    )


def generate_heterogeneous_noise_data(
    n_samples=1000,
    seq_len=128,
    n_channels=96,
    latent_dim=8,
    base_noise_level=0.3,
    noise_heterogeneity=0.5,
    normalize=True,
    seed=42
):
    """
    Generate data with heterogeneous noise across channels.

    This scenario favors JAE2 (multi-view sampling) over JAE1 (fixed split) because:
    - Noise varies across channels
    - Random subsampling better adapts to noise distribution
    - Fixed 50/50 split may separate channels with similar noise

    Args:
        n_samples (int): Number of samples.
        seq_len (int): Sequence length.
        n_channels (int): Number of channels.
        latent_dim (int): Latent dimensionality.
        base_noise_level (float): Base noise level.
        noise_heterogeneity (float): How much noise varies across channels (0-1).
        normalize (bool): Whether to normalize.
        seed (int): Random seed.

    Returns:
        tuple: (clean_data, noisy_data, channel_snrs)
    """
    np.random.seed(seed)

    # Generate latent dynamics
    t = np.linspace(0, 10 * np.pi, seq_len)
    latents = np.zeros((n_samples, latent_dim, seq_len))

    for i in range(n_samples):
        for d in range(latent_dim):
            freq = np.random.uniform(0.5, 1.5)
            phase = np.random.uniform(0, 2 * np.pi)
            latents[i, d, :] = np.sin(freq * t + phase)

    # Project with NONLINEAR mixing (same as transients scenario)
    projection_matrix_1 = np.random.randn(n_channels, latent_dim)
    projection_matrix_2 = np.random.randn(n_channels, latent_dim)
    if normalize:
        projection_matrix_1 = projection_matrix_1 / (np.linalg.norm(projection_matrix_1, axis=1, keepdims=True) + 1e-8)
        projection_matrix_2 = projection_matrix_2 / (np.linalg.norm(projection_matrix_2, axis=1, keepdims=True) + 1e-8)

    latents_reshaped = latents.transpose(0, 2, 1).reshape(n_samples * seq_len, latent_dim)
    
    # Nonlinear mixing
    linear_part = latents_reshaped @ projection_matrix_1.T
    nonlinear_part = np.tanh(latents_reshaped @ projection_matrix_2.T)
    
    poly_features = []
    for i in range(min(3, latent_dim)):
        for j in range(i+1, min(4, latent_dim)):
            poly_features.append(latents_reshaped[:, i] * latents_reshaped[:, j])
    
    if poly_features:
        poly_matrix = np.random.randn(n_channels, len(poly_features))
        if normalize:
            poly_matrix = poly_matrix / (np.linalg.norm(poly_matrix, axis=0, keepdims=True) + 1e-8)
        poly_part = np.column_stack(poly_features) @ poly_matrix.T
        clean_data_flat = 0.5 * linear_part + 0.3 * nonlinear_part + 0.2 * poly_part
    else:
        clean_data_flat = 0.6 * linear_part + 0.4 * nonlinear_part
    
    clean_data = clean_data_flat.reshape(n_samples, seq_len, n_channels).transpose(0, 2, 1)

    if normalize:
        mean = clean_data.mean()
        std = clean_data.std() + 1e-8
        clean_data = (clean_data - mean) / std
        clean_data = clean_data - clean_data.min() + 0.5

    # Add heterogeneous noise (different noise levels per channel)
    channel_noise_levels = base_noise_level * (1 + noise_heterogeneity * np.random.randn(n_channels))
    channel_noise_levels = np.clip(channel_noise_levels, 0.1, 2.0)

    noisy_data = clean_data.copy()
    signal_power = np.mean(clean_data ** 2, axis=(0, 2))  # Per channel

    for ch in range(n_channels):
        noise_std = np.sqrt(signal_power[ch]) * channel_noise_levels[ch]
        noise = np.random.randn(n_samples, seq_len) * noise_std
        noisy_data[:, ch, :] += noise

    noisy_data = np.maximum(0, noisy_data)

    # Calculate per-channel SNR
    channel_snrs = np.zeros(n_channels)
    for ch in range(n_channels):
        sig_pow = np.mean(clean_data[:, ch, :] ** 2)
        noise_pow = np.mean((clean_data[:, ch, :] - noisy_data[:, ch, :]) ** 2)
        channel_snrs[ch] = 10 * np.log10(sig_pow / (noise_pow + 1e-10))

    return (
        torch.tensor(clean_data, dtype=torch.float32),
        torch.tensor(noisy_data, dtype=torch.float32),
        channel_snrs
    )


def generate_nonstationary_data(
    n_samples=1000,
    seq_len=128,
    n_channels=96,
    latent_dim=8,
    noise_level=0.5,
    regime_changes=3,
    normalize=True,
    seed=42
):
    """
    Generate data with non-stationary dynamics (regime changes).

    This favors JAE2 because:
    - Temporal convolutions adapt to changing dynamics
    - Multiple views capture different aspects of regimes
    - VICReg handles rotated representations across regimes

    Args:
        n_samples (int): Number of samples.
        seq_len (int): Sequence length.
        n_channels (int): Number of channels.
        latent_dim (int): Latent dimensionality.
        noise_level (float): Noise level.
        regime_changes (int): Number of regime changes per sample.
        normalize (bool): Whether to normalize.
        seed (int): Random seed.

    Returns:
        tuple: (clean_data, noisy_data, regime_boundaries)
    """
    np.random.seed(seed)

    latents = np.zeros((n_samples, latent_dim, seq_len))
    regime_boundaries = np.sort(np.random.randint(10, seq_len - 10, regime_changes))

    for i in range(n_samples):
        # Create regime boundaries
        boundaries = [0] + list(regime_boundaries) + [seq_len]

        for regime_idx in range(len(boundaries) - 1):
            start = boundaries[regime_idx]
            end = boundaries[regime_idx + 1]
            t_segment = np.linspace(0, 5 * np.pi, end - start)

            for d in range(latent_dim):
                # Different frequency and phase per regime
                freq = np.random.uniform(0.3, 2.0)
                phase = np.random.uniform(0, 2 * np.pi)
                amplitude = np.random.uniform(0.5, 2.0)
                latents[i, d, start:end] = amplitude * np.sin(freq * t_segment + phase)

    # Smooth transitions between regimes
    for i in range(n_samples):
        for boundary in regime_boundaries:
            window_size = 5
            for d in range(latent_dim):
                if boundary - window_size >= 0 and boundary + window_size < seq_len:
                    # Apply gaussian smoothing at boundaries
                    weights = np.exp(-0.5 * (np.arange(-window_size, window_size + 1) / 2) ** 2)
                    weights = weights / weights.sum()
                    segment = latents[i, d, boundary - window_size:boundary + window_size + 1]
                    latents[i, d, boundary - window_size:boundary + window_size + 1] = np.convolve(
                        segment, weights, mode='same'
                    )

    # Project with NONLINEAR mixing
    projection_matrix_1 = np.random.randn(n_channels, latent_dim)
    projection_matrix_2 = np.random.randn(n_channels, latent_dim)
    if normalize:
        projection_matrix_1 = projection_matrix_1 / (np.linalg.norm(projection_matrix_1, axis=1, keepdims=True) + 1e-8)
        projection_matrix_2 = projection_matrix_2 / (np.linalg.norm(projection_matrix_2, axis=1, keepdims=True) + 1e-8)

    latents_reshaped = latents.transpose(0, 2, 1).reshape(n_samples * seq_len, latent_dim)
    
    # Nonlinear mixing
    linear_part = latents_reshaped @ projection_matrix_1.T
    nonlinear_part = np.tanh(latents_reshaped @ projection_matrix_2.T)
    
    poly_features = []
    for i in range(min(3, latent_dim)):
        for j in range(i+1, min(4, latent_dim)):
            poly_features.append(latents_reshaped[:, i] * latents_reshaped[:, j])
    
    if poly_features:
        poly_matrix = np.random.randn(n_channels, len(poly_features))
        if normalize:
            poly_matrix = poly_matrix / (np.linalg.norm(poly_matrix, axis=0, keepdims=True) + 1e-8)
        poly_part = np.column_stack(poly_features) @ poly_matrix.T
        clean_data_flat = 0.5 * linear_part + 0.3 * nonlinear_part + 0.2 * poly_part
    else:
        clean_data_flat = 0.6 * linear_part + 0.4 * nonlinear_part
    
    clean_data = clean_data_flat.reshape(n_samples, seq_len, n_channels).transpose(0, 2, 1)

    if normalize:
        mean = clean_data.mean()
        std = clean_data.std() + 1e-8
        clean_data = (clean_data - mean) / std
        clean_data = clean_data - clean_data.min() + 0.5

    # Add noise
    signal_power = np.mean(clean_data ** 2)
    noise_std = np.sqrt(signal_power) * noise_level
    noise = np.random.randn(*clean_data.shape) * noise_std
    noisy_data = clean_data + noise
    noisy_data = np.maximum(0, noisy_data)

    return (
        torch.tensor(clean_data, dtype=torch.float32),
        torch.tensor(noisy_data, dtype=torch.float32),
        regime_boundaries
    )


def generate_data_scenarios(scenario='easy', n_samples=500, seed=42):
    """
    Generate data for different difficulty scenarios.

    Scenarios:
    - 'easy': Smooth sinusoids, homogeneous noise (JAE1 and JAE2 similar)
    - 'transients': Sharp transients favor JAE2 (U-Net)
    - 'heterogeneous': Heterogeneous noise favors JAE2 (multi-view)
    - 'nonstationary': Regime changes favor JAE2 (temporal + VICReg)
    - 'high_noise': High noise level challenges both

    Args:
        scenario (str): Scenario name.
        n_samples (int): Number of samples.
        seed (int): Random seed.

    Returns:
        dict: Contains 'clean', 'noisy', 'info' keys with data and metadata.
    """
    if scenario == 'easy':
        clean, noisy = generate_simulated_data(
            n_samples=n_samples,
            seq_len=128,
            n_channels=96,
            latent_dim=8,
            noise_level=0.3,
            seed=seed
        )
        snr = calculate_snr(clean, noisy)
        return {
            'clean': clean,
            'noisy': noisy,
            'info': {'snr_db': snr, 'description': 'Smooth sinusoids, low noise'}
        }

    elif scenario == 'transients':
        clean, noisy, snr = generate_neural_data_with_transients(
            n_samples=n_samples,
            seq_len=128,
            n_channels=96,
            latent_dim=8,
            noise_level=0.5,
            n_transients=5,
            seed=seed
        )
        return {
            'clean': clean,
            'noisy': noisy,
            'info': {'snr_db': snr, 'description': 'Sharp transients (spikes), favors U-Net'}
        }

    elif scenario == 'heterogeneous':
        clean, noisy, channel_snrs = generate_heterogeneous_noise_data(
            n_samples=n_samples,
            seq_len=128,
            n_channels=96,
            latent_dim=8,
            base_noise_level=0.3,
            noise_heterogeneity=0.8,
            seed=seed
        )
        avg_snr = np.mean(channel_snrs)
        return {
            'clean': clean,
            'noisy': noisy,
            'info': {
                'snr_db': float(avg_snr),
                'channel_snrs': channel_snrs,
                'description': 'Heterogeneous noise across channels, favors multi-view'
            }
        }

    elif scenario == 'nonstationary':
        clean, noisy, boundaries = generate_nonstationary_data(
            n_samples=n_samples,
            seq_len=128,
            n_channels=96,
            latent_dim=8,
            noise_level=0.4,
            regime_changes=3,
            seed=seed
        )
        snr = calculate_snr(clean, noisy)
        return {
            'clean': clean,
            'noisy': noisy,
            'info': {
                'snr_db': snr,
                'regime_boundaries': boundaries,
                'description': 'Non-stationary dynamics, favors temporal convolutions'
            }
        }

    elif scenario == 'high_noise':
        clean, noisy = generate_simulated_data(
            n_samples=n_samples,
            seq_len=128,
            n_channels=96,
            latent_dim=8,
            noise_level=1.2,  # High noise
            seed=seed
        )
        snr = calculate_snr(clean, noisy)
        return {
            'clean': clean,
            'noisy': noisy,
            'info': {'snr_db': snr, 'description': 'High noise level, challenging for both'}
        }

    else:
        raise ValueError(f"Unknown scenario: {scenario}. Choose from: easy, transients, heterogeneous, nonstationary, high_noise")


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


def plot_signals(
    signals_dict,
    channel_idx=0,
    sample_idx=0,
    time_range=None,
    title=None,
    figsize=(15, 6),
    show_legend=True,
    save_path=None
):
    """
    Plot multiple signals for comparison.

    Args:
        signals_dict (dict): Dictionary of {label: signal} where signal is
            (n_samples, n_channels, seq_len) tensor or array.
        channel_idx (int, optional): Channel to plot. Default: 0.
        sample_idx (int, optional): Sample to plot. Default: 0.
        time_range (tuple, optional): (start, end) time indices. Default: None (all).
        title (str, optional): Plot title. Default: auto-generated.
        figsize (tuple, optional): Figure size. Default: (15, 6).
        show_legend (bool, optional): Whether to show legend. Default: True.
        save_path (str, optional): Path to save figure. Default: None.

    Returns:
        matplotlib.figure.Figure: The figure object.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> clean, noisy = generate_simulated_data(n_samples=10)
        >>> denoised = model.denoise(noisy)
        >>> plot_signals({
        ...     'Ground Truth': clean,
        ...     'Noisy': noisy,
        ...     'Denoised': denoised
        ... }, channel_idx=0, sample_idx=0)
        >>> plt.show()
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    fig, ax = plt.subplots(figsize=figsize)

    # Color scheme
    colors = {
        'Ground Truth': 'black',
        'Clean': 'black',
        'Noisy': 'gray',
        'JAE1': 'blue',
        'JAE2': 'red',
        'PCA': 'purple',
        'Denoised': 'green'
    }
    line_styles = {
        'Ground Truth': '-',
        'Clean': '-',
        'Noisy': '-',
        'JAE1': '--',
        'JAE2': '-',
        'PCA': ':',
        'Denoised': '-'
    }
    line_widths = {
        'Ground Truth': 2.5,
        'Clean': 2.5,
        'Noisy': 1.5,
        'JAE1': 2,
        'JAE2': 2.5,
        'PCA': 2,
        'Denoised': 2
    }
    alphas = {
        'Ground Truth': 1.0,
        'Clean': 1.0,
        'Noisy': 0.6,
        'JAE1': 0.9,
        'JAE2': 1.0,
        'PCA': 0.8,
        'Denoised': 0.9
    }

    for label, signal in signals_dict.items():
        # Convert to numpy if needed
        if isinstance(signal, torch.Tensor):
            signal = signal.detach().cpu().numpy()

        # Extract the specific channel and sample
        data = signal[sample_idx, channel_idx, :]

        # Apply time range if specified
        if time_range is not None:
            start, end = time_range
            data = data[start:end]
            t = np.arange(start, end)
        else:
            t = np.arange(len(data))

        # Get style parameters
        color = colors.get(label, None)
        linestyle = line_styles.get(label, '-')
        linewidth = line_widths.get(label, 2)
        alpha = alphas.get(label, 1.0)

        ax.plot(t, data, label=label, color=color, linestyle=linestyle,
                linewidth=linewidth, alpha=alpha)

    ax.set_xlabel('Time (samples)', fontsize=12)
    ax.set_ylabel('Signal Amplitude', fontsize=12)

    if title is None:
        title = f'Signal Comparison (Sample {sample_idx}, Channel {channel_idx})'
    ax.set_title(title, fontsize=14, fontweight='bold')

    if show_legend:
        ax.legend(loc='best', fontsize=11, framealpha=0.9)

    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_multichannel_signals(
    signal,
    n_channels_to_plot=8,
    sample_idx=0,
    title='Multi-channel Neural Signals',
    figsize=(15, 10),
    save_path=None
):
    """
    Plot multiple channels of neural signals in a grid.

    Args:
        signal (torch.Tensor or np.ndarray): Signal data of shape
            (n_samples, n_channels, seq_len).
        n_channels_to_plot (int, optional): Number of channels to display. Default: 8.
        sample_idx (int, optional): Sample index. Default: 0.
        title (str, optional): Plot title. Default: 'Multi-channel Neural Signals'.
        figsize (tuple, optional): Figure size. Default: (15, 10).
        save_path (str, optional): Path to save figure. Default: None.

    Returns:
        matplotlib.figure.Figure: The figure object.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> clean, noisy = generate_simulated_data(n_samples=10)
        >>> plot_multichannel_signals(noisy, n_channels_to_plot=8)
        >>> plt.show()
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().numpy()

    n_channels = min(n_channels_to_plot, signal.shape[1])
    n_cols = 2
    n_rows = (n_channels + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    axes = axes.flatten()

    for i in range(n_channels):
        data = signal[sample_idx, i, :]
        t = np.arange(len(data))

        axes[i].plot(t, data, color='steelblue', linewidth=1.5, alpha=0.8)
        axes[i].set_title(f'Channel {i}', fontsize=10, fontweight='bold')
        axes[i].set_ylabel('Amplitude', fontsize=9)
        axes[i].grid(True, alpha=0.3)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

    # Set x-label for bottom plots
    for i in range(n_channels - n_cols, n_channels):
        if i < len(axes):
            axes[i].set_xlabel('Time (samples)', fontsize=9)

    # Hide unused subplots
    for i in range(n_channels, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_snr_comparison(
    results_dict,
    title='Denoising Performance Comparison',
    figsize=(10, 6),
    save_path=None
):
    """
    Plot SNR comparison across different methods.

    Args:
        results_dict (dict): Dictionary of {method_name: snr_value}.
        title (str, optional): Plot title. Default: 'Denoising Performance Comparison'.
        figsize (tuple, optional): Figure size. Default: (10, 6).
        save_path (str, optional): Path to save figure. Default: None.

    Returns:
        matplotlib.figure.Figure: The figure object.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> results = {
        ...     'Noisy': 5.2,
        ...     'PCA': 8.5,
        ...     'JAE1': 12.3,
        ...     'JAE2': 15.7
        ... }
        >>> plot_snr_comparison(results)
        >>> plt.show()
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    fig, ax = plt.subplots(figsize=figsize)

    methods = list(results_dict.keys())
    snr_values = list(results_dict.values())

    colors = ['gray', 'purple', 'blue', 'red', 'green', 'orange']
    bars = ax.bar(methods, snr_values, color=colors[:len(methods)], alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar, value in zip(bars, snr_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{value:.2f} dB', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_channel_snr_distribution(
    channel_snrs,
    title='Channel-wise SNR Distribution',
    figsize=(12, 5),
    save_path=None
):
    """
    Plot distribution of SNR across channels.

    Args:
        channel_snrs (np.ndarray or list): SNR values for each channel.
        title (str, optional): Plot title. Default: 'Channel-wise SNR Distribution'.
        figsize (tuple, optional): Figure size. Default: (12, 5).
        save_path (str, optional): Path to save figure. Default: None.

    Returns:
        matplotlib.figure.Figure: The figure object.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> _, _, channel_snrs = generate_heterogeneous_noise_data(n_samples=100)
        >>> plot_channel_snr_distribution(channel_snrs)
        >>> plt.show()
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: SNR per channel
    channels = np.arange(len(channel_snrs))
    ax1.scatter(channels, channel_snrs, alpha=0.6, s=30, color='steelblue')
    ax1.axhline(np.mean(channel_snrs), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(channel_snrs):.2f} dB')
    ax1.set_xlabel('Channel Index', fontsize=11)
    ax1.set_ylabel('SNR (dB)', fontsize=11)
    ax1.set_title('SNR per Channel', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Histogram
    ax2.hist(channel_snrs, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(channel_snrs), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(channel_snrs):.2f} dB')
    ax2.axvline(np.median(channel_snrs), color='green', linestyle='--',
                linewidth=2, label=f'Median: {np.median(channel_snrs):.2f} dB')
    ax2.set_xlabel('SNR (dB)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('SNR Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_training_history(
    history_dict,
    title='Training History',
    figsize=(12, 5),
    save_path=None
):
    """
    Plot training and validation loss curves.

    Args:
        history_dict (dict): Dictionary with 'train_loss' and optionally 'val_loss'.
        title (str, optional): Plot title. Default: 'Training History'.
        figsize (tuple, optional): Figure size. Default: (12, 5).
        save_path (str, optional): Path to save figure. Default: None.

    Returns:
        matplotlib.figure.Figure: The figure object.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> model = JAE(latent_dim=12)
        >>> model.fit(data, epochs=100, validation_split=0.2)
        >>> history = model.get_training_history()
        >>> plot_training_history(history)
        >>> plt.show()
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    fig, ax = plt.subplots(figsize=figsize)

    epochs = np.arange(1, len(history_dict['train_loss']) + 1)

    ax.plot(epochs, history_dict['train_loss'], label='Training Loss',
            color='blue', linewidth=2, marker='o', markersize=3, alpha=0.8)

    if 'val_loss' in history_dict and len(history_dict['val_loss']) > 0:
        ax.plot(epochs, history_dict['val_loss'], label='Validation Loss',
                color='red', linewidth=2, marker='s', markersize=3, alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_regime_changes(
    signal,
    regime_boundaries,
    sample_idx=0,
    channel_idx=0,
    title='Non-stationary Signal with Regime Changes',
    figsize=(15, 5),
    save_path=None
):
    """
    Plot signal with regime change boundaries highlighted.

    Args:
        signal (torch.Tensor or np.ndarray): Signal of shape (n_samples, n_channels, seq_len).
        regime_boundaries (list or np.ndarray): Indices where regimes change.
        sample_idx (int, optional): Sample to plot. Default: 0.
        channel_idx (int, optional): Channel to plot. Default: 0.
        title (str, optional): Plot title. Default: 'Non-stationary Signal with Regime Changes'.
        figsize (tuple, optional): Figure size. Default: (15, 5).
        save_path (str, optional): Path to save figure. Default: None.

    Returns:
        matplotlib.figure.Figure: The figure object.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> clean, noisy, boundaries = generate_nonstationary_data(n_samples=10)
        >>> plot_regime_changes(noisy, boundaries, sample_idx=0, channel_idx=0)
        >>> plt.show()
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)

    data = signal[sample_idx, channel_idx, :]
    t = np.arange(len(data))

    ax.plot(t, data, color='steelblue', linewidth=2, alpha=0.8, label='Signal')

    # Highlight regime boundaries
    colors = plt.cm.Set3(np.linspace(0, 1, len(regime_boundaries) + 1))
    boundaries_extended = [0] + list(regime_boundaries) + [len(data)]

    for i in range(len(boundaries_extended) - 1):
        start = boundaries_extended[i]
        end = boundaries_extended[i + 1]
        ax.axvspan(start, end, alpha=0.2, color=colors[i], label=f'Regime {i+1}')

    # Draw vertical lines at boundaries
    for boundary in regime_boundaries:
        ax.axvline(boundary, color='red', linestyle='--', linewidth=2, alpha=0.7)

    ax.set_xlabel('Time (samples)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Signal Amplitude', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

