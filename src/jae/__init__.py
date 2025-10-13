"""
JAE: Joint Autoencoder for Neural Signal Denoising

A modern PyTorch implementation of Joint Autoencoders for denoising high-dimensional
neural signals, based on Altan et al. (2021) with state-of-the-art enhancements.
"""

__version__ = "0.1.0"

# Main API
from .api import JAE

# Models (for advanced users)
from .models import JAE1, JAE2, SimpleAutoencoder, UNet1D

# Utilities
from .utils import (
    get_device,
    generate_simulated_data,
    generate_neural_data_with_transients,
    generate_heterogeneous_noise_data,
    generate_nonstationary_data,
    generate_data_scenarios,
    calculate_snr,
    calculate_r2_score,
    calculate_correlation,
    run_pca_baseline,
    validate_input_data,
    set_seed,
    plot_signals,
    plot_multichannel_signals,
    plot_snr_comparison,
    plot_channel_snr_distribution,
    plot_training_history,
    plot_regime_changes,
)

# Loss functions (for advanced users)
from .losses import (
    jae1_loss_fn,
    jae2_loss_fn,
    huber_reconstruction_loss,
    vicreg_loss,
    temporal_smoothness_loss,
)

# Configuration (for advanced users)
from .config import (
    JAE1Config,
    JAE2Config,
    get_default_config,
)

__all__ = [
    # Main API
    "JAE",
    # Version
    "__version__",
    # Models
    "JAE1",
    "JAE2",
    "SimpleAutoencoder",
    "UNet1D",
    # Utilities
    "get_device",
    "generate_simulated_data",
    "generate_neural_data_with_transients",
    "generate_heterogeneous_noise_data",
    "generate_nonstationary_data",
    "generate_data_scenarios",
    "calculate_snr",
    "calculate_r2_score",
    "calculate_correlation",
    "run_pca_baseline",
    "validate_input_data",
    "set_seed",
    "plot_signals",
    "plot_multichannel_signals",
    "plot_snr_comparison",
    "plot_channel_snr_distribution",
    "plot_training_history",
    "plot_regime_changes",
    # Loss functions
    "jae1_loss_fn",
    "jae2_loss_fn",
    "huber_reconstruction_loss",
    "vicreg_loss",
    "temporal_smoothness_loss",
    # Configuration
    "JAE1Config",
    "JAE2Config",
    "get_default_config",
]

