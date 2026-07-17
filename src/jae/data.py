"""Paper-faithful neural-data simulator, evaluation controls, and splits.

Reimplements the generative model from Altan et al. (2021) with two fixes over
the legacy simulator in ``jae.utils``:

1. Latents are diverse across latent dimensions and across samples (each
   latent dimension is an independently drawn smooth trajectory, or an i.i.d.
   firing-rate-like signal), not a single shared sinusoid family per sample.
   This keeps the ground-truth manifold from being trivially compressible.
2. Normalization is per-channel (each channel divided by its own max), not a
   single global min-max over the whole tensor.

Generative model (matches SPEC_jae_paper.md, Fig 1 / Eq 1):
    1. Latents z in R^(d x T), diverse per latent dimension and per sample.
    2. Linear mixing W in R^(C x d), entries drawn from N(0, 1).
    3. Per-channel max-normalization (each channel divided by its own max).
    4. Optional Eq1 nonlinearity f(x) = (exp(alpha*x) - 1) / (exp(alpha) - 1).
    5. Additive per-channel Gaussian noise calibrated to a target SNR (dB):
       10 * log10(var(signal) / var(noise)) == snr_db.
"""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

_N_SINUSOID_COMPONENTS = 4


def _make_latents(
    n_samples: int,
    latent_dim: int,
    n_timepoints: int,
    dynamics: str,
    smoothing_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate diverse latent trajectories.

    Each latent dimension gets an independently drawn trajectory so the
    resulting (latent_dim, n_timepoints) matrix is close to full rank per
    sample, rather than every dimension sharing one sinusoid family.

    Args:
        n_samples: Number of samples (trials).
        latent_dim: Number of latent dimensions (d).
        n_timepoints: Timepoints per sample (M).
        dynamics: 'iid' for i.i.d. positive firing-rate-like samples that are
            then Gaussian-smoothed (uncorrelated over time before smoothing),
            or 'smooth' for latents built as a random mixture of a few
            sinusoids with independent frequency, phase, and amplitude drawn
            per latent dimension and per sample.
        smoothing_sigma: Std of the Gaussian smoothing kernel (in samples).
            Set to 0 to disable smoothing.
        rng: Seeded numpy random Generator.

    Returns:
        np.ndarray of shape (n_samples, latent_dim, n_timepoints).

    Raises:
        ValueError: If dynamics is not 'iid' or 'smooth'.
    """
    if dynamics == "iid":
        # i.i.d. positive firing-rate-like draws, uncorrelated over time.
        latents = rng.gamma(shape=2.0, scale=1.0, size=(n_samples, latent_dim, n_timepoints))
    elif dynamics == "smooth":
        # Random mixture of a few sinusoids per (sample, latent dim), each
        # with independent frequency, phase, and amplitude. Amplitudes decay
        # with component index for a 1/f-like spectral shape.
        t = np.linspace(0.0, 4.0 * np.pi, n_timepoints)
        shape = (n_samples, latent_dim, _N_SINUSOID_COMPONENTS, 1)
        freqs = rng.uniform(0.3, 3.0, size=shape)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=shape)
        decay = 1.0 / np.arange(1, _N_SINUSOID_COMPONENTS + 1)
        amps = rng.uniform(0.5, 1.5, size=shape) * decay[None, None, :, None]
        components = amps * np.sin(freqs * t[None, None, None, :] + phases)
        latents = components.sum(axis=2)
    else:
        raise ValueError(f"dynamics must be 'iid' or 'smooth', got {dynamics!r}")

    if smoothing_sigma > 0:
        latents = gaussian_filter1d(latents, sigma=smoothing_sigma, axis=-1)

    return latents


def _simulate_clean(
    n_samples: int,
    n_channels: int,
    n_timepoints: int,
    latent_dim: int,
    snr_db: float,
    nonlinear: bool,
    alpha: float,
    smoothing_sigma: float,
    dynamics: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the noise-free signal shared by the two public simulators.

    Returns:
        Tuple of (clean, w, latents, noise_std):
            clean: np.ndarray (n_samples, n_channels, n_timepoints).
            w: np.ndarray (n_channels, latent_dim), the true mixing matrix.
            latents: np.ndarray (n_samples, latent_dim, n_timepoints).
            noise_std: np.ndarray (n_channels,), per-channel noise std needed
                to hit snr_db.
    """
    latents = _make_latents(n_samples, latent_dim, n_timepoints, dynamics, smoothing_sigma, rng)
    w = rng.standard_normal((n_channels, latent_dim))

    # Linear mixing: x[n, c, t] = sum_d w[c, d] * latents[n, d, t].
    x = np.einsum("cd,ndt->nct", w, latents)

    # Per-channel max-normalization (NOT global): each channel divided by its
    # own max over all samples and timepoints.
    channel_max = x.max(axis=(0, 2), keepdims=True)
    channel_max = np.where(channel_max > 1e-12, channel_max, 1.0)
    x = x / channel_max

    if nonlinear:
        clean = (np.exp(alpha * x) - 1.0) / (np.exp(alpha) - 1.0)
    else:
        clean = x

    signal_var = clean.var(axis=(0, 2))
    noise_std = np.sqrt(signal_var / (10.0 ** (snr_db / 10.0)))

    return clean, w, latents, noise_std


def simulate_neural_data(
    n_samples: int = 500,
    n_channels: int = 96,
    n_timepoints: int = 128,
    latent_dim: int = 6,
    snr_db: float = 5.0,
    nonlinear: bool = True,
    alpha: float = 16.0,
    smoothing_sigma: float = 2.0,
    dynamics: str = "smooth",
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Simulate neural population data following Altan et al. (2021).

    Generative model: diverse latents -> linear mixing (W ~ N(0, 1)) ->
    per-channel max-normalization -> optional Eq1 nonlinearity -> per-channel
    additive Gaussian noise calibrated to snr_db.

    Args:
        n_samples: Number of samples (trials), B.
        n_channels: Number of recording channels, C.
        n_timepoints: Timepoints per sample, T.
        latent_dim: Intrinsic latent dimensionality, d.
        snr_db: Target per-channel SNR in dB: 10*log10(var(signal)/var(noise)).
        nonlinear: If True, apply the Eq1 exponential nonlinearity.
        alpha: Eq1 nonlinearity strength. Higher alpha is more nonlinear.
        smoothing_sigma: Std of Gaussian smoothing applied to latents (in
            samples). Set to 0 to disable.
        dynamics: 'smooth' (default) for random-sinusoid-mixture latents, or
            'iid' for i.i.d. firing-rate-like latents that are then smoothed.
        seed: Random seed. None draws fresh entropy (non-reproducible).

    Returns:
        Tuple of (clean, noisy, info):
            clean: torch.float32 Tensor (n_samples, n_channels, n_timepoints).
            noisy: torch.float32 Tensor (n_samples, n_channels, n_timepoints).
            info: dict with keys 'W' (true mixing matrix, Tensor), 'latents'
                (Tensor), 'latent_dim', 'alpha', 'nonlinear', 'dynamics',
                'snr_db' (target), 'snr_db_per_channel' (achieved,
                np.ndarray), 'noise_std_per_channel' (np.ndarray),
                'n_samples', 'n_channels', 'n_timepoints', 'smoothing_sigma',
                'seed'.
    """
    rng = np.random.default_rng(seed)
    clean, w, latents, noise_std = _simulate_clean(
        n_samples,
        n_channels,
        n_timepoints,
        latent_dim,
        snr_db,
        nonlinear,
        alpha,
        smoothing_sigma,
        dynamics,
        rng,
    )

    noise = rng.standard_normal(clean.shape) * noise_std[None, :, None]
    noisy = clean + noise

    signal_var = clean.var(axis=(0, 2))
    achieved_noise_var = noise.var(axis=(0, 2))
    achieved_snr_db = 10.0 * np.log10(signal_var / np.maximum(achieved_noise_var, 1e-20))

    info = {
        "W": torch.from_numpy(w.astype(np.float32)),
        "latents": torch.from_numpy(latents.astype(np.float32)),
        "latent_dim": latent_dim,
        "alpha": float(alpha),
        "nonlinear": nonlinear,
        "dynamics": dynamics,
        "snr_db": float(snr_db),
        "snr_db_per_channel": achieved_snr_db.astype(np.float32),
        "noise_std_per_channel": noise_std.astype(np.float32),
        "n_samples": n_samples,
        "n_channels": n_channels,
        "n_timepoints": n_timepoints,
        "smoothing_sigma": float(smoothing_sigma),
        "seed": seed,
    }

    return (
        torch.from_numpy(clean.astype(np.float32)),
        torch.from_numpy(noisy.astype(np.float32)),
        info,
    )


def simulate_paired_noise(
    n_samples: int = 500,
    n_channels: int = 96,
    n_timepoints: int = 128,
    latent_dim: int = 6,
    snr_db: float = 5.0,
    nonlinear: bool = True,
    alpha: float = 16.0,
    smoothing_sigma: float = 2.0,
    dynamics: str = "smooth",
    n_repeats: int = 2,
    seed: int | None = None,
) -> tuple[torch.Tensor, list, dict]:
    """Simulate paired-noise data: one clean signal, R independent noise draws.

    Used for cvPCA-style estimators that need repeated noisy observations of
    the same underlying noise-free signal to separate signal variance from
    noise variance without a biased plug-in estimate.

    Args:
        n_samples: Number of samples (trials), B.
        n_channels: Number of recording channels, C.
        n_timepoints: Timepoints per sample, T.
        latent_dim: Intrinsic latent dimensionality, d.
        snr_db: Target per-channel SNR in dB.
        nonlinear: If True, apply the Eq1 exponential nonlinearity.
        alpha: Eq1 nonlinearity strength.
        smoothing_sigma: Std of Gaussian smoothing applied to latents.
        dynamics: 'smooth' or 'iid'; see simulate_neural_data.
        n_repeats: Number of independent noise draws, R.
        seed: Random seed. None draws fresh entropy (non-reproducible).

    Returns:
        Tuple of (clean, noisy_list, info):
            clean: torch.float32 Tensor (n_samples, n_channels, n_timepoints).
            noisy_list: list of R torch.float32 Tensors, each
                (n_samples, n_channels, n_timepoints), sharing `clean` but
                each with an independent noise realization.
            info: dict, same keys as simulate_neural_data (minus the achieved
                per-call SNR) plus 'n_repeats'.
    """
    rng = np.random.default_rng(seed)
    clean, w, latents, noise_std = _simulate_clean(
        n_samples,
        n_channels,
        n_timepoints,
        latent_dim,
        snr_db,
        nonlinear,
        alpha,
        smoothing_sigma,
        dynamics,
        rng,
    )

    noisy_list = []
    for _ in range(n_repeats):
        noise = rng.standard_normal(clean.shape) * noise_std[None, :, None]
        noisy_list.append(torch.from_numpy((clean + noise).astype(np.float32)))

    info = {
        "W": torch.from_numpy(w.astype(np.float32)),
        "latents": torch.from_numpy(latents.astype(np.float32)),
        "latent_dim": latent_dim,
        "alpha": float(alpha),
        "nonlinear": nonlinear,
        "dynamics": dynamics,
        "snr_db": float(snr_db),
        "noise_std_per_channel": noise_std.astype(np.float32),
        "n_samples": n_samples,
        "n_channels": n_channels,
        "n_timepoints": n_timepoints,
        "smoothing_sigma": float(smoothing_sigma),
        "n_repeats": n_repeats,
        "seed": seed,
    }

    return torch.from_numpy(clean.astype(np.float32)), noisy_list, info


def phase_shuffle(x: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    """Theiler phase-randomization surrogate.

    Randomizes the FFT phase of each channel independently while keeping the
    magnitude spectrum fixed, then inverse-transforms back to the time
    domain. This preserves each channel's power spectrum (and therefore its
    autocorrelation structure) while destroying its specific temporal
    waveform, giving a null surrogate for timing-sensitive statistics.

    Args:
        x: Tensor (n_samples, n_channels, n_timepoints).
        seed: Random seed for the phase draws. None draws fresh entropy.

    Returns:
        torch.float32 Tensor, same shape as x, with per-channel power
        spectrum preserved but time-domain waveform randomized.
    """
    rng = np.random.default_rng(seed)
    x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    n_timepoints = x_np.shape[-1]

    spectrum = np.fft.rfft(x_np, axis=-1)
    magnitude = np.abs(spectrum)
    n_freq = spectrum.shape[-1]

    random_phase = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.shape)
    # DC (and Nyquist, if n_timepoints is even) must stay real-valued for the
    # inverse transform of a real signal; keep their original phase.
    random_phase[..., 0] = np.angle(spectrum[..., 0])
    if n_timepoints % 2 == 0:
        random_phase[..., n_freq - 1] = np.angle(spectrum[..., n_freq - 1])

    shuffled_spectrum = magnitude * np.exp(1j * random_phase)
    shuffled = np.fft.irfft(shuffled_spectrum, n=n_timepoints, axis=-1)

    return torch.from_numpy(shuffled.astype(np.float32))


def make_noise_only(
    info_or_shape,
    per_channel_noise_std,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a noise-only control: zero latent signal, matched additive noise.

    Useful as a negative control: any structure a model recovers from this
    data is an artifact of the model/estimator, not the (nonexistent) signal.

    Args:
        info_or_shape: Either an info dict from simulate_neural_data (must
            contain 'n_samples', 'n_channels', 'n_timepoints') or an explicit
            (n_samples, n_channels, n_timepoints) shape tuple.
        per_channel_noise_std: Scalar or array-like of length n_channels
            giving the noise std to use per channel (e.g. from
            info['noise_std_per_channel']) so the control matches a prior
            simulation's noise level.
        seed: Random seed. None draws fresh entropy.

    Returns:
        Tuple of (clean0, noisy):
            clean0: torch.float32 zeros, (n_samples, n_channels, n_timepoints).
            noisy: torch.float32 Tensor, same shape, pure Gaussian noise.

    Raises:
        ValueError: If per_channel_noise_std length does not match n_channels.
    """
    if isinstance(info_or_shape, dict):
        shape = (
            info_or_shape["n_samples"],
            info_or_shape["n_channels"],
            info_or_shape["n_timepoints"],
        )
    else:
        shape = tuple(info_or_shape)

    n_samples, n_channels, n_timepoints = shape

    std = np.asarray(per_channel_noise_std, dtype=np.float64)
    if std.ndim == 0:
        std = np.full(n_channels, float(std))
    if std.shape[0] != n_channels:
        raise ValueError(
            f"per_channel_noise_std length {std.shape[0]} does not match n_channels {n_channels}"
        )

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n_samples, n_channels, n_timepoints)) * std[None, :, None]

    clean0 = torch.zeros((n_samples, n_channels, n_timepoints), dtype=torch.float32)
    noisy = torch.from_numpy(noise.astype(np.float32))

    return clean0, noisy


def train_val_test_split(
    X: torch.Tensor,  # noqa: N803 (X matches the INTERFACES.md contract)
    clean: torch.Tensor | None = None,
    fracs: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int | None = None,
) -> dict:
    """Split samples into train/val/test partitions along the sample axis.

    Args:
        X: Tensor (n_samples, n_channels, n_timepoints), typically noisy data.
        clean: Optional matching clean Tensor, same shape as X, split with
            the same indices.
        fracs: (train, val, test) fractions. Must sum to 1.
        seed: Random seed for the sample permutation. None draws fresh
            entropy (non-reproducible).

    Returns:
        dict with keys 'train', 'val', 'test'. Each maps to a dict with:
            'idx': LongTensor of the original sample indices in this split.
            'noisy': X indexed by 'idx'.
            'clean': clean indexed by 'idx' (only present if clean is given).

    Raises:
        ValueError: If fracs does not have length 3, does not sum to ~1, or
            clean's sample count does not match X's.
    """
    if len(fracs) != 3:
        raise ValueError(f"fracs must have length 3 (train, val, test), got {len(fracs)}")
    if abs(sum(fracs) - 1.0) > 1e-6:
        raise ValueError(f"fracs must sum to 1.0, got {sum(fracs)}")

    n = X.shape[0]
    if clean is not None and clean.shape[0] != n:
        raise ValueError("clean and X must have the same number of samples")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_train = min(int(round(fracs[0] * n)), n)
    n_val = min(int(round(fracs[1] * n)), n - n_train)

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    splits = {}
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        idx_t = torch.from_numpy(idx.copy()).long()
        entry = {"idx": idx_t, "noisy": X[idx_t]}
        if clean is not None:
            entry["clean"] = clean[idx_t]
        splits[name] = entry

    return splits
