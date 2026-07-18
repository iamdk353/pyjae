"""JAE: Joint Autoencoders for neural signal denoising and manifold learning.

Two model families:

- :class:`JAE1` (and the :class:`JAE` facade with ``backend="jae1"``): a modular,
  collapse-resistant channel-split autoencoder, a corrected reimplementation of
  Altan et al. (2021).
- :class:`JAE2` (facade ``backend="jepa"``): a JEPA-style joint-embedding
  predictive model that learns the underlying manifold by predicting
  masked-region embeddings in latent space.

Supporting modules: :mod:`pyjae.data` (simulation and controls), :mod:`pyjae.views`
(modular channel-split / masking strategies), :mod:`pyjae.metrics` (a non-gameable
latent-quality and invariance panel), :mod:`pyjae.baselines` (PCA, Factor Analysis,
denoising autoencoder, Wiener oracle), and :mod:`pyjae.eval` (the benchmark harness).
"""

__version__ = "0.2.0"

from pyjae.api import JAE
from pyjae.baselines import (
    DenoisingAutoencoder,
    factor_analysis_denoise,
    mean_predictor,
    pca_denoise,
    wiener_oracle,
)
from pyjae.data import (
    make_noise_only,
    phase_shuffle,
    simulate_neural_data,
    simulate_paired_noise,
    train_val_test_split,
)
from pyjae.losses import jae1_loss_fn, jae2_jepa_loss_fn, jepa_loss, vicreg_reg
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
)
from pyjae.models import JAE1, JAE2
from pyjae.utils import (
    calculate_snr,
    calculate_vaf,
    get_device,
    run_pca_baseline,
    set_seed,
    validate_input_data,
)
from pyjae.views import (
    ContiguousSplit,
    OverlappingSplit,
    RandomDisjointSplit,
    SpatioTemporalBlockMask,
)

__all__ = [
    "__version__",
    # High-level API
    "JAE",
    # Models
    "JAE1",
    "JAE2",
    # Data + controls
    "simulate_neural_data",
    "simulate_paired_noise",
    "phase_shuffle",
    "make_noise_only",
    "train_val_test_split",
    # Views / splits
    "RandomDisjointSplit",
    "ContiguousSplit",
    "OverlappingSplit",
    "SpatioTemporalBlockMask",
    # Metrics
    "per_channel_vaf",
    "effective_rank",
    "participation_ratio",
    "lidar",
    "alignment",
    "uniformity",
    "invariance_ratio",
    "nuisance_probe",
    "collapse_report",
    # Baselines
    "pca_denoise",
    "factor_analysis_denoise",
    "mean_predictor",
    "wiener_oracle",
    "DenoisingAutoencoder",
    # Losses
    "jae1_loss_fn",
    "jae2_jepa_loss_fn",
    "jepa_loss",
    "vicreg_reg",
    # Utils
    "get_device",
    "set_seed",
    "calculate_vaf",
    "calculate_snr",
    "run_pca_baseline",
    "validate_input_data",
]
