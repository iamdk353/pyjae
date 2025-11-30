"""
JAE: Joint Autoencoder for Neural Signal Denoising

A PyTorch implementation of the Joint Autoencoder from Altan et al. (2021)
for denoising high-dimensional neural signals.
"""

__version__ = "0.1.0"

from .api import JAE
from .utils import (
    get_device,
    simulate_neural_data,
    calculate_vaf,
    calculate_snr,
    run_pca_baseline,
    set_seed,
)

# For advanced users
from .models import JAE1, JAE2, SimpleAutoencoder, UNet1D
from .losses import jae1_loss_fn, jae2_loss_fn, vicreg_loss

__all__ = [
    "JAE",
    "__version__",
    # Utilities
    "get_device",
    "simulate_neural_data",
    "calculate_vaf",
    "calculate_snr",
    "run_pca_baseline",
    "set_seed",
    # Advanced
    "JAE1",
    "JAE2",
    "SimpleAutoencoder",
    "UNet1D",
    "jae1_loss_fn",
    "jae2_loss_fn",
    "vicreg_loss",
]
