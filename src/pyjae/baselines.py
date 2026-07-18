"""Matched-capacity denoising baselines for the JAE evaluation protocol.

These are the baselines a credible nonlinear denoiser (JAE) must beat on a
non-gameable evaluation: a trivial per-channel mean floor, PCA (the linear
baseline under an isotropic-noise assumption), Factor Analysis (the CORRECT
linear baseline, since it models per-channel noise variance instead of
assuming it is isotropic like PCA), a matched-capacity denoising autoencoder
(DAE, to show gains beyond "just using a neural net"), and a privileged
Wiener/linear-MMSE oracle that uses the TRUE signal/noise covariances as a
ceiling on what any linear method can achieve.

All baselines follow strict split discipline: normalization statistics and
model parameters are fit on TRAIN only and applied to TEST; TEST is never
used to compute stats or choose hyperparameters. Inputs may be numpy arrays
or torch tensors of shape (B, C, T) = (n_samples, n_channels, n_timepoints);
outputs are always torch.float32 tensors of the same shape as the relevant
input.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from sklearn.decomposition import PCA, FactorAnalysis

ArrayLike = np.ndarray | torch.Tensor


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert a numpy array or torch tensor to a float64 numpy array.

    Args:
        x: Input array or tensor of any numeric dtype.

    Returns:
        A float64 numpy array with the same shape as ``x``.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _bct_to_flat(x: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    """Reshape (B, C, T) samples into per-timepoint samples (B * T, C).

    Args:
        x: Array of shape (B, C, T).

    Returns:
        Tuple of (flat array of shape (B * T, C), B, C, T).
    """
    b, c, t = x.shape
    flat = x.transpose(0, 2, 1).reshape(b * t, c)
    return flat, b, c, t


def _flat_to_bct(flat: np.ndarray, b: int, c: int, t: int) -> torch.Tensor:
    """Reshape per-timepoint samples (B * T, C) back to (B, C, T).

    Args:
        flat: Array of shape (B * T, C).
        b: Number of samples.
        c: Number of channels.
        t: Number of timepoints.

    Returns:
        A torch.float32 tensor of shape (B, C, T).
    """
    x = flat.reshape(b, t, c).transpose(0, 2, 1)
    return torch.tensor(np.ascontiguousarray(x), dtype=torch.float32)


def _channel_mean_std(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std with a floor to avoid division by zero.

    Args:
        flat: Array of shape (N, C).

    Returns:
        Tuple of (mean, std), each of shape (1, C).
    """
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def pca_denoise(
    train_X: ArrayLike,  # noqa: N803 (train_X/test_X match the INTERFACES.md contract)
    test_X: ArrayLike,  # noqa: N803
    k: int,
) -> torch.Tensor:
    """PCA denoising baseline: z-score on train, top-k PCs, reconstruct.

    Normalization statistics (per-channel mean/std) and the PCA basis are
    both fit on ``train_X`` only, then applied to ``test_X``; test data is
    never used to compute stats. PCA implicitly assumes isotropic
    per-channel noise after z-scoring, which is why Factor Analysis
    (:func:`factor_analysis_denoise`) is the better-specified linear
    baseline when per-channel noise variance differs across channels.

    Args:
        train_X: Training data, shape (B_train, C, T).
        test_X: Test data to denoise, shape (B_test, C, T).
        k: Number of principal components to retain. Silently clipped to
            at most min(n_channels, n_train_samples) if larger.

    Returns:
        Denoised test data, a torch.float32 tensor of shape (B_test, C, T).

    Raises:
        ValueError: If train_X and test_X have a different number of
            channels.
    """
    train_flat, _, c_train, _ = _bct_to_flat(_to_numpy(train_X))
    test_flat, b_test, c_test, t_test = _bct_to_flat(_to_numpy(test_X))
    if c_train != c_test:
        raise ValueError(f"channel mismatch: train has {c_train}, test has {c_test}")

    mean, std = _channel_mean_std(train_flat)
    train_z = (train_flat - mean) / std
    test_z = (test_flat - mean) / std

    k_eff = max(1, min(k, train_z.shape[0], train_z.shape[1]))
    pca = PCA(n_components=k_eff)
    pca.fit(train_z)
    test_recon_z = pca.inverse_transform(pca.transform(test_z))
    test_recon = test_recon_z * std + mean
    return _flat_to_bct(test_recon, b_test, c_test, t_test)


def factor_analysis_denoise(
    train_X: ArrayLike,  # noqa: N803 (train_X/test_X match the INTERFACES.md contract)
    test_X: ArrayLike,  # noqa: N803
    k: int,
) -> torch.Tensor:
    """Factor Analysis denoising baseline: the correct linear baseline.

    Unlike PCA, Factor Analysis models per-channel noise variance (its
    noise covariance Psi is diagonal, not isotropic) instead of assuming
    uniform noise across channels after z-scoring. It is fit on TRAIN
    (z-scored using TRAIN statistics) and applied to TEST via the
    posterior-mean reconstruction:

        Z_posterior = fa.transform(test_z)              # posterior mean latents
        X_hat_z = fa.mean_ + Z_posterior @ fa.components_
        X_hat = X_hat_z * std + mean                     # un-standardize

    This is the linear-MMSE (Wiener) solution under the FA generative
    model X = mean + Z @ components_ + noise, noise ~ N(0, Psi) with
    diagonal Psi, evaluated at the posterior mean of Z given the (z-scored)
    observation.

    Args:
        train_X: Training data, shape (B_train, C, T).
        test_X: Test data to denoise, shape (B_test, C, T).
        k: Number of latent factors. Silently clipped to at most
            n_channels if larger.

    Returns:
        Denoised test data, a torch.float32 tensor of shape (B_test, C, T).

    Raises:
        ValueError: If train_X and test_X have a different number of
            channels.
    """
    train_flat, _, c_train, _ = _bct_to_flat(_to_numpy(train_X))
    test_flat, b_test, c_test, t_test = _bct_to_flat(_to_numpy(test_X))
    if c_train != c_test:
        raise ValueError(f"channel mismatch: train has {c_train}, test has {c_test}")

    mean, std = _channel_mean_std(train_flat)
    train_z = (train_flat - mean) / std
    test_z = (test_flat - mean) / std

    k_eff = max(1, min(k, c_train))
    fa = FactorAnalysis(n_components=k_eff, random_state=0)
    fa.fit(train_z)

    z_posterior = fa.transform(test_z)
    test_recon_z = fa.mean_ + z_posterior @ fa.components_
    test_recon = test_recon_z * std + mean
    return _flat_to_bct(test_recon, b_test, c_test, t_test)


def mean_predictor(test_X: ArrayLike) -> torch.Tensor:  # noqa: N803
    """Trivial floor baseline: per-sample, per-channel mean over time.

    Each channel's time series is replaced by its own across-time mean,
    computed independently per test sample (no fitting, no train data
    needed). This is the trivial floor: any credible denoiser must beat it
    clearly and per-channel, since it captures zero temporal or
    cross-channel structure.

    Args:
        test_X: Data to summarize, shape (B, C, T).

    Returns:
        A torch.float32 tensor of shape (B, C, T), constant over the T axis.
    """
    x = _to_numpy(test_X)
    per_channel_mean = x.mean(axis=2, keepdims=True)
    out = np.broadcast_to(per_channel_mean, x.shape)
    return torch.tensor(np.ascontiguousarray(out), dtype=torch.float32)


def wiener_oracle(
    noisy_test: ArrayLike,
    signal_cov: ArrayLike,
    noise_cov: ArrayLike,
    mean: ArrayLike,
) -> torch.Tensor:
    """Privileged multivariate Wiener / linear-MMSE oracle.

    Uses the TRUE signal and noise covariances (not estimated from data),
    so this is a ceiling on what any linear method can achieve, not a
    baseline that can be fit blind. A denoiser that exceeds this oracle on
    NONLINEAR data demonstrates a real nonlinear gain; a denoiser that
    exceeds a near-perfect-info oracle by a wide margin on the same linear
    regime the oracle was built for is a leakage-bug signal, not a genuine
    win (see SPEC_eval.md).

    Applied per timepoint (row-vector form, y and mean as (C,) vectors):

        x_hat = mean + Sigma_s @ inv(Sigma_s + Sigma_n) @ (y - mean)

    Args:
        noisy_test: Noisy data to denoise, shape (B, C, T).
        signal_cov: True signal covariance, shape (C, C).
        noise_cov: True noise covariance, shape (C, C).
        mean: Per-channel mean, shape (C,).

    Returns:
        Denoised data, a torch.float32 tensor of shape (B, C, T).
    """
    flat, b, c, t = _bct_to_flat(_to_numpy(noisy_test))
    sigma_s = np.asarray(signal_cov, dtype=np.float64)
    sigma_n = np.asarray(noise_cov, dtype=np.float64)
    mean_row = np.asarray(mean, dtype=np.float64).reshape(1, c)

    # gain_t == G.T where G = Sigma_s @ inv(Sigma_s + Sigma_n); solving the
    # linear system avoids forming the matrix inverse explicitly.
    gain_t = np.linalg.solve(sigma_s + sigma_n, sigma_s)
    centered = flat - mean_row
    denoised_flat = mean_row + centered @ gain_t
    return _flat_to_bct(denoised_flat, b, c, t)


class DenoisingAutoencoder(nn.Module):
    """Matched-capacity MLP denoising autoencoder baseline.

    A single, plain-MLP autoencoder operating per-timepoint on (C,) samples,
    exactly like the linear baselines (flattened from (B, C, T) to
    (B * T, C)). Its capacity is configured via ``hidden``/``latent_dim`` so
    it can be matched to a linear baseline's k, isolating whether JAE's
    gains come from its split/joint structure rather than simply from being
    "a neural net".

    Attributes:
        encoder: Funnel MLP mapping n_channels -> ... -> latent_dim.
        decoder: Mirrored MLP mapping latent_dim -> ... -> n_channels.
        loss_history_: List of per-epoch mean training losses after fit().
    """

    def __init__(
        self,
        n_channels: int,
        latent_dim: int,
        hidden: tuple[int, ...] = (64,),
        dropout_p: float = 0.0,
    ) -> None:
        """Initialize the encoder/decoder MLPs.

        Args:
            n_channels: Number of input/output channels C.
            latent_dim: Bottleneck size (matched to a linear baseline's k).
            hidden: Hidden layer widths for the encoder; mirrored (reversed)
                for the decoder. Empty tuple gives a single linear layer
                each side.
            dropout_p: Dropout probability applied after hidden-layer
                activations. 0 disables dropout.
        """
        super().__init__()
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.encoder = self._make_mlp([n_channels, *hidden, latent_dim], dropout_p)
        self.decoder = self._make_mlp([latent_dim, *reversed(hidden), n_channels], dropout_p)
        self.loss_history_: list[float] = []

    @staticmethod
    def _make_mlp(dims: list[int], dropout_p: float) -> nn.Sequential:
        """Build a Linear/ReLU/[Dropout] MLP with no activation on the last layer.

        Args:
            dims: Layer widths, e.g. [in, hidden1, ..., out].
            dropout_p: Dropout probability after each hidden activation.

        Returns:
            An nn.Sequential MLP.
        """
        layers: list[nn.Module] = []
        n_layers = len(dims) - 1
        for i in range(n_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < n_layers - 1:
                layers.append(nn.ReLU())
                if dropout_p > 0:
                    layers.append(nn.Dropout(dropout_p))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode a batch of per-timepoint samples.

        Args:
            x: Tensor of shape (N, n_channels).

        Returns:
            Reconstruction, a tensor of shape (N, n_channels).
        """
        return self.decoder(self.encoder(x))

    def fit(
        self,
        noisy: ArrayLike,
        target: ArrayLike,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        verbose: bool = False,
    ) -> "DenoisingAutoencoder":
        """Train with AdamW to map noisy inputs to targets.

        Pass ``target=clean`` for a supervised DAE, or ``target=noisy`` for
        a self-denoising DAE (no clean labels needed).

        Args:
            noisy: Input data, shape (B, C, T).
            target: Target data, shape (B, C, T) (same shape as noisy).
            epochs: Number of training epochs.
            lr: AdamW learning rate.
            weight_decay: AdamW weight decay.
            batch_size: Minibatch size over the flattened (B * T) samples.
            verbose: If True, print the mean loss every epoch.

        Returns:
            self, for chaining, e.g.
            ``DenoisingAutoencoder(...).fit(noisy, clean).denoise(test)``.
        """
        noisy_flat, _, _, _ = _bct_to_flat(_to_numpy(noisy))
        target_flat, _, _, _ = _bct_to_flat(_to_numpy(target))
        device = next(self.parameters()).device
        noisy_t = torch.tensor(noisy_flat, dtype=torch.float32, device=device)
        target_t = torch.tensor(target_flat, dtype=torch.float32, device=device)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        n = noisy_t.shape[0]
        self.loss_history_ = []
        self.train()
        for _ in range(epochs):
            perm = torch.randperm(n, device=device)
            epoch_losses = []
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                optimizer.zero_grad()
                pred = self.forward(noisy_t[idx])
                loss = F.mse_loss(pred, target_t[idx])
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            self.loss_history_.append(float(np.mean(epoch_losses)))
            if verbose:
                print(f"epoch loss: {self.loss_history_[-1]:.6f}")
        self.eval()
        return self

    def denoise(self, X: ArrayLike) -> torch.Tensor:  # noqa: N803
        """Denoise (B, C, T) data with the trained autoencoder.

        Args:
            X: Data to denoise, shape (B, C, T).

        Returns:
            Denoised data, a torch.float32 tensor of shape (B, C, T).
        """
        flat, b, c, t = _bct_to_flat(_to_numpy(X))
        device = next(self.parameters()).device
        x_t = torch.tensor(flat, dtype=torch.float32, device=device)
        self.eval()
        with torch.no_grad():
            out = self.forward(x_t)
        return _flat_to_bct(out.detach().cpu().numpy(), b, c, t)


def cv_select_k(
    train_X: ArrayLike,  # noqa: N803 (train_X matches the INTERFACES.md contract)
    ks: list[int],
    held_out_frac: float = 0.2,
    seed: int = 0,
) -> int:
    """Select a PCA rank k by held-out reconstruction error.

    Splits TRAIN into a CV-fit fold and a held-out fold, z-scores using
    only the CV-fit fold's statistics, fits PCA at each candidate k on the
    CV-fit fold, and picks the k with the lowest reconstruction MSE on the
    held-out fold. This never looks at any "true" latent dimensionality, so
    it is a non-circular way to choose k for :func:`pca_denoise`.

    Args:
        train_X: Training data, shape (B, C, T).
        ks: Candidate values of k to evaluate.
        held_out_frac: Fraction of flattened train samples held out for CV.
        seed: Random seed for the CV split.

    Returns:
        The value from ``ks`` with the lowest held-out reconstruction MSE.

    Raises:
        ValueError: If ks is empty.
    """
    if len(ks) == 0:
        raise ValueError("ks must be non-empty")

    flat, _, _, _ = _bct_to_flat(_to_numpy(train_X))
    n = flat.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * held_out_frac)))
    val_idx, fit_idx = perm[:n_val], perm[n_val:]
    fit_data, val_data = flat[fit_idx], flat[val_idx]

    mean, std = _channel_mean_std(fit_data)
    fit_z = (fit_data - mean) / std
    val_z = (val_data - mean) / std

    best_k = ks[0]
    best_mse = np.inf
    for k in ks:
        k_eff = max(1, min(k, fit_z.shape[0], fit_z.shape[1]))
        pca = PCA(n_components=k_eff)
        pca.fit(fit_z)
        val_recon = pca.inverse_transform(pca.transform(val_z))
        mse = float(np.mean((val_recon - val_z) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_k = k
    return int(best_k)
