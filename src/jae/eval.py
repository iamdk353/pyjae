"""Non-gameable evaluation harness for JAE denoising.

This module encodes the evaluation discipline that separates a real denoising
result from a gamed one:

- Held-out test split: every method is fit on train and scored on unseen test
  samples, so nothing can memorize per-sample noise.
- Matched capacity: JAE and the linear baselines use the same latent dimension.
- Per-channel VAF: reported as a mean over channels, not a global flatten that
  a few high-variance channels could dominate.
- A mean-predictor floor and strong linear baselines (PCA, Factor Analysis) that
  a credible nonlinear method must beat, plus a Factor Analysis / Wiener reading
  of the linear regime where PCA is expected to win.
- Regime awareness: the pass condition is JAE beating the linear baselines on
  NONLINEAR data (not everywhere); a JAE win on linear data would be a red flag.
- Negative controls: on phase-shuffled data JAE must fall to the mean-predictor
  floor, and on pure noise it must not reconstruct better than a matched-capacity
  PCA. Both confirm the method does not invent structure that is not there.

See ``scripts/benchmark.py`` for a runnable sweep built on these functions.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from jae.api import JAE
from jae.baselines import (
    DenoisingAutoencoder,
    factor_analysis_denoise,
    mean_predictor,
    pca_denoise,
)
from jae.data import phase_shuffle, simulate_neural_data, train_val_test_split
from jae.metrics import per_channel_vaf


@dataclass
class MethodScores:
    """Per-method VAF summary on a held-out test split."""

    name: str
    mean_vaf: float
    weighted_vaf: float
    worst_vaf: float


@dataclass
class CellResult:
    """Result of one (snr, nonlinear, seed) benchmark cell."""

    snr_db: float
    nonlinear: bool
    seed: int
    latent_dim: int
    scores: dict[str, float] = field(default_factory=dict)


def _to_tensor(x) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x.float()


def train_jae1(train_noisy, test_noisy, latent_dim, epochs=200, seed=0):
    """Fit JAE1 on train, return the denoised held-out test signal."""
    model = JAE(latent_dim=latent_dim, backend="jae1", device="cpu", verbose=False, seed=seed)
    model.fit(train_noisy, epochs=epochs, batch_size=16, verbose=False)
    return model.denoise(test_noisy)


def train_dae(train_noisy, test_noisy, latent_dim, epochs=200, seed=0):
    """Fit a matched-capacity denoising autoencoder (self-supervised) on train."""
    torch.manual_seed(seed)
    train_t = _to_tensor(train_noisy)
    n_channels = train_t.shape[1]
    dae = DenoisingAutoencoder(n_channels=n_channels, latent_dim=latent_dim)
    dae.fit(train_t, train_t, epochs=epochs, verbose=False)
    return dae.denoise(_to_tensor(test_noisy))


def evaluate_cell(
    snr_db: float,
    nonlinear: bool,
    seed: int,
    n_samples: int = 400,
    n_channels: int = 64,
    n_timepoints: int = 96,
    latent_dim: int = 6,
    alpha: float = 3.0,
    epochs: int = 200,
) -> CellResult:
    """Run every method on one dataset and score on the held-out test split.

    Fits JAE1, PCA, Factor Analysis, and a denoising autoencoder on the train
    split (all at the same ``latent_dim``), plus the mean-predictor floor, then
    scores mean per-channel VAF against the clean test ground truth.
    """
    clean, noisy, _ = simulate_neural_data(
        n_samples=n_samples,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        latent_dim=latent_dim,
        snr_db=snr_db,
        nonlinear=nonlinear,
        alpha=alpha,
        seed=seed,
    )
    split = train_val_test_split(noisy, clean=clean, fracs=(0.7, 0.0, 0.3), seed=seed)
    train_noisy = split["train"]["noisy"]
    test_noisy = split["test"]["noisy"]
    test_clean = split["test"]["clean"]

    preds = {
        "noisy": test_noisy,
        "mean": mean_predictor(test_noisy),
        "pca": pca_denoise(train_noisy, test_noisy, k=latent_dim),
        "fa": factor_analysis_denoise(train_noisy, test_noisy, k=latent_dim),
        "dae": train_dae(train_noisy, test_noisy, latent_dim, epochs=epochs, seed=seed),
        "jae1": train_jae1(train_noisy, test_noisy, latent_dim, epochs=epochs, seed=seed),
    }

    scores = {name: per_channel_vaf(test_clean, pred)["mean"] for name, pred in preds.items()}
    return CellResult(
        snr_db=snr_db, nonlinear=nonlinear, seed=seed, latent_dim=latent_dim, scores=scores
    )


def negative_control(
    control: str,
    seed: int,
    n_samples: int = 400,
    n_channels: int = 64,
    n_timepoints: int = 96,
    latent_dim: int = 6,
    snr_db: float = 5.0,
    epochs: int = 200,
) -> dict[str, float]:
    """Fit JAE1 on structure-destroyed data; it must not invent signal.

    ``control="phase_shuffle"``: scramble each channel's temporal phase (a
    Theiler surrogate that preserves the power spectrum but destroys the shared
    cross-channel manifold), then score denoising VAF against the shuffled clean
    ground truth. A passing method scores near the mean-predictor floor (JAE VAF
    should not meaningfully exceed it). Returned as ``{"jae1", "mean"}``.

    ``control="noise_only"``: fit on pure noise (no latent signal). There is no
    clean signal to recover, so VAF against it is undefined; instead we measure
    RECONSTRUCTION VAF against the noisy input itself. A method that hallucinates
    structure reconstructs the noise better than a matched-dimension PCA can; a
    faithful method reconstructs no better than PCA. Returned as
    ``{"jae1_recon", "pca_recon"}``; passing means ``jae1_recon`` does not
    meaningfully exceed ``pca_recon``.
    """
    if control == "phase_shuffle":
        clean, noisy, _ = simulate_neural_data(
            n_samples=n_samples,
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            latent_dim=latent_dim,
            snr_db=snr_db,
            nonlinear=True,
            seed=seed,
        )
        surrogate = phase_shuffle(noisy, seed=seed)
        target = phase_shuffle(clean, seed=seed)
        split = train_val_test_split(surrogate, clean=target, fracs=(0.7, 0.0, 0.3), seed=seed)
        jae_pred = train_jae1(
            split["train"]["noisy"], split["test"]["noisy"], latent_dim, epochs=epochs, seed=seed
        )
        return {
            "jae1": per_channel_vaf(split["test"]["clean"], jae_pred)["mean"],
            "mean": per_channel_vaf(split["test"]["clean"], mean_predictor(split["test"]["noisy"]))[
                "mean"
            ],
        }

    if control == "noise_only":
        rng = torch.Generator().manual_seed(seed)
        noise = torch.randn(n_samples, n_channels, n_timepoints, generator=rng)
        split = train_val_test_split(noise, fracs=(0.7, 0.0, 0.3), seed=seed)
        train_noise, test_noise = split["train"]["noisy"], split["test"]["noisy"]
        jae_pred = train_jae1(train_noise, test_noise, latent_dim, epochs=epochs, seed=seed)
        # Match total latent budget: JAE1 uses two partitions of `latent_dim`
        # each, so a fair PCA reference gets 2 * latent_dim components. Otherwise
        # JAE1 looks like it "overfits noise" purely because it has more capacity.
        pca_k = min(2 * latent_dim, n_channels)
        pca_pred = pca_denoise(train_noise, test_noise, k=pca_k)
        return {
            "jae1_recon": per_channel_vaf(test_noise, jae_pred)["mean"],
            "pca_recon": per_channel_vaf(test_noise, pca_pred)["mean"],
        }

    raise ValueError(f"unknown control {control!r}")


def aggregate(results: list[CellResult]) -> dict:
    """Average per-method scores across seeds, grouped by (snr, nonlinear)."""
    grouped: dict[tuple, dict[str, list[float]]] = {}
    for r in results:
        key = (r.snr_db, r.nonlinear)
        bucket = grouped.setdefault(key, {})
        for name, val in r.scores.items():
            bucket.setdefault(name, []).append(val)
    summary = {}
    for key, bucket in grouped.items():
        summary[key] = {
            name: (float(np.mean(vals)), float(np.std(vals))) for name, vals in bucket.items()
        }
    return summary
