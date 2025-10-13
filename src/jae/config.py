"""
Configuration and default hyperparameters for JAE models.

This module provides default configurations, validation functions, and warning
systems to ensure sensible parameter choices while allowing full customization.
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class JAE1Config:
    """
    Default configuration for JAE1 (Original) model.

    Attributes:
        latent_dim (Optional[int]): Latent space dimensionality. If None, will be
            auto-detected as input_channels // 8.
        dropout_p (float): Dropout probability at input layer. Default: 0.05.
        latent_weight (float): Weight for latent alignment loss. Default: 1.0.
        learning_rate (float): Learning rate for optimizer. Default: 0.001.
        weight_decay (float): Weight decay (L2 regularization). Default: 1e-5.
        epochs (int): Number of training epochs. Default: 100.
        batch_size (int): Batch size for training. Default: 32.
    """
    latent_dim: Optional[int] = None
    dropout_p: float = 0.05
    latent_weight: float = 1.0
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 100
    batch_size: int = 32


@dataclass
class JAE2Config:
    """
    Default configuration for JAE2 (Modernized) model.

    Attributes:
        latent_dim (Optional[int]): Latent space dimensionality. If None, will be
            auto-detected as input_channels // 8.
        num_networks (int): Number of parallel networks. Default: 5.
        subsample_fraction (float): Fraction of neurons to sample per network.
            Default: 0.8 (80%).
        unet_channels (List[int]): Channel dimensions for U-Net encoder levels.
            Default: [32, 64].
        recon_weight (float): Weight for reconstruction loss. Default: 1.0.
        vicreg_weight (float): Weight for VICReg loss. Default: 0.1.
        huber_delta (float): Delta parameter for Huber loss. Default: 1.0.
        lambda_inv (float): VICReg invariance weight. Default: 25.0.
        mu_var (float): VICReg variance weight. Default: 25.0.
        nu_cov (float): VICReg covariance weight. Default: 1.0.
        learning_rate (float): Learning rate for optimizer. Default: 0.0005.
        weight_decay (float): Weight decay (L2 regularization). Default: 1e-5.
        epochs (int): Number of training epochs. Default: 100.
        batch_size (int): Batch size for training. Default: 32.
    """
    latent_dim: Optional[int] = None
    num_networks: int = 5
    subsample_fraction: float = 0.8
    unet_channels: List[int] = field(default_factory=lambda: [32, 64])
    recon_weight: float = 1.0
    vicreg_weight: float = 0.1
    huber_delta: float = 1.0
    lambda_inv: float = 25.0
    mu_var: float = 25.0
    nu_cov: float = 1.0
    learning_rate: float = 0.0005
    weight_decay: float = 1e-5
    epochs: int = 100
    batch_size: int = 32


def get_default_config(model_type='jae2'):
    """
    Get default configuration for a model type.

    Args:
        model_type (str): Either 'jae1' or 'jae2'. Default: 'jae2'.

    Returns:
        JAE1Config or JAE2Config: Default configuration object.

    Raises:
        ValueError: If model_type is not 'jae1' or 'jae2'.

    Example:
        >>> config = get_default_config('jae2')
        >>> print(config.num_networks)  # 5
    """
    if model_type.lower() == 'jae1':
        return JAE1Config()
    elif model_type.lower() == 'jae2':
        return JAE2Config()
    else:
        raise ValueError(f"model_type must be 'jae1' or 'jae2', got '{model_type}'")


def auto_detect_latent_dim(input_channels, warn=True):
    """
    Automatically detect a reasonable latent dimensionality.

    Uses the heuristic: latent_dim = max(input_channels // 8, 4)

    Args:
        input_channels (int): Number of input channels (neurons).
        warn (bool, optional): Whether to issue a warning. Default: True.

    Returns:
        int: Suggested latent dimensionality.

    Example:
        >>> latent_dim = auto_detect_latent_dim(96)
        WARNING: latent_dim not specified. Auto-detecting as 12 (input_channels // 8).
        >>> print(latent_dim)  # 12
    """
    latent_dim = max(input_channels // 8, 4)

    if warn:
        warnings.warn(
            f"latent_dim not specified. Auto-detecting as {latent_dim} "
            f"(input_channels // 8). For better results, consider tuning this parameter "
            f"based on your data's intrinsic dimensionality.",
            UserWarning
        )

    return latent_dim


def validate_model_params(model_type, input_channels, latent_dim, **kwargs):
    """
    Validate model parameters and provide helpful warnings.

    Args:
        model_type (str): Model type ('jae1' or 'jae2').
        input_channels (int): Number of input channels.
        latent_dim (int): Latent dimensionality.
        **kwargs: Additional model-specific parameters.

    Raises:
        ValueError: If parameters are invalid.
        UserWarning: If parameters might lead to suboptimal performance.

    Example:
        >>> validate_model_params('jae1', 96, 12)
        >>> # No errors or warnings
    """
    # Validate model type
    if model_type.lower() not in ['jae1', 'jae2']:
        raise ValueError(f"model_type must be 'jae1' or 'jae2', got '{model_type}'")

    # Validate input_channels
    if input_channels < 2:
        raise ValueError(f"input_channels must be at least 2, got {input_channels}")

    if model_type.lower() == 'jae1' and input_channels % 2 != 0:
        raise ValueError(
            f"JAE1 requires an even number of input channels for 50/50 split. "
            f"Got {input_channels} channels."
        )

    # Validate latent_dim
    if latent_dim < 1:
        raise ValueError(f"latent_dim must be at least 1, got {latent_dim}")

    if latent_dim >= input_channels:
        warnings.warn(
            f"latent_dim ({latent_dim}) is >= input_channels ({input_channels}). "
            f"This defeats the purpose of dimensionality reduction and may lead to overfitting.",
            UserWarning
        )

    # JAE2-specific validations
    if model_type.lower() == 'jae2':
        num_networks = kwargs.get('num_networks', 5)
        subsample_fraction = kwargs.get('subsample_fraction', 0.8)

        if num_networks < 2:
            warnings.warn(
                f"num_networks ({num_networks}) < 2. JAE2 benefits from multiple views. "
                f"Consider using at least 3-5 networks.",
                UserWarning
            )

        if subsample_fraction <= 0 or subsample_fraction > 1:
            raise ValueError(
                f"subsample_fraction must be in (0, 1], got {subsample_fraction}"
            )

        subsample_dim = int(input_channels * subsample_fraction)
        if subsample_dim < latent_dim:
            warnings.warn(
                f"Subsampled dimension ({subsample_dim}) < latent_dim ({latent_dim}). "
                f"This may limit model capacity. Consider increasing subsample_fraction "
                f"or decreasing latent_dim.",
                UserWarning
            )


def validate_training_params(epochs, batch_size, learning_rate):
    """
    Validate training hyperparameters.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate.

    Raises:
        ValueError: If parameters are invalid.
        UserWarning: If parameters might lead to training issues.

    Example:
        >>> validate_training_params(epochs=100, batch_size=32, learning_rate=0.001)
    """
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")

    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    if batch_size == 1:
        warnings.warn(
            "batch_size=1 may cause issues with BatchNorm and VICReg loss. "
            "Consider using batch_size >= 2.",
            UserWarning
        )

    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")

    if learning_rate > 0.1:
        warnings.warn(
            f"learning_rate ({learning_rate}) is quite high. "
            f"This may cause training instability. Typical values are 1e-4 to 1e-3.",
            UserWarning
        )

    if learning_rate < 1e-6:
        warnings.warn(
            f"learning_rate ({learning_rate}) is very low. "
            f"Training may be extremely slow or fail to converge.",
            UserWarning
        )


def suggest_batch_size(n_samples, default=32):
    """
    Suggest an appropriate batch size based on dataset size.

    Args:
        n_samples (int): Number of samples in dataset.
        default (int, optional): Default batch size. Default: 32.

    Returns:
        int: Suggested batch size.

    Example:
        >>> batch_size = suggest_batch_size(n_samples=1000)
        >>> print(batch_size)  # 32
    """
    if n_samples < default:
        suggested = max(n_samples // 4, 2)
        warnings.warn(
            f"Dataset has only {n_samples} samples. Reducing batch_size to {suggested}.",
            UserWarning
        )
        return suggested
    return default


def print_config_summary(config, model_type):
    """
    Print a formatted summary of the configuration.

    Args:
        config (JAE1Config or JAE2Config): Configuration object.
        model_type (str): Model type for display.

    Example:
        >>> config = JAE2Config(latent_dim=12)
        >>> print_config_summary(config, 'jae2')
        JAE2 Configuration:
          latent_dim: 12
          num_networks: 5
          ...
    """
    print(f"\n{model_type.upper()} Configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    print()

