"""High-level API for the Joint Autoencoder (JAE).

Provides a scikit-learn-style facade over the two model backends:

- ``backend="jae1"`` (default): the modular channel-split Joint Autoencoder
  (a corrected reimplementation of Altan et al. 2021). Denoises by forcing two
  disjoint channel partitions to share a latent, rejecting independent noise.
- ``backend="jepa"``: the JEPA-style joint-embedding predictive model, which
  learns the underlying manifold by predicting masked-region embeddings in
  latent space and denoises through a lightweight decoder head.

Both backends expose ``forward`` / ``loss`` / ``denoise``, so the training loop
here is backend-agnostic.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from pyjae.metrics import per_channel_vaf, vaf
from pyjae.models import JAE1, JAE2
from pyjae.utils import get_device, validate_input_data

_BACKENDS = ("jae1", "jepa")


class JAE:
    """Joint Autoencoder for neural signal denoising and manifold learning.

    Parameters
    ----------
    latent_dim : int, optional
        Latent dimensionality. For ``backend="jae1"`` this is the shared
        bottleneck of every channel partition; if None it is auto-detected as
        ``max(input_channels // 8, 4)``. For ``backend="jepa"`` it is the
        VICReg projector dimension (default 64 if None).
    backend : {"jae1", "jepa"}, default="jae1"
        Which model to use.
    learning_rate : float, default=1e-3
        AdamW learning rate.
    weight_decay : float, default=1e-5
        AdamW weight decay.
    standardize : bool, default=True
        Per-channel z-score the input using statistics fit on the training data
        only, and invert on the denoised output. Strongly recommended: it gives
        the reconstruction loss a well-scaled gradient and avoids the
        near-constant-output local minimum on small-scale signals.
    seed : int, default=0
        Seed for the model's channel split (jae1) or mask generator (jepa).
    use_gpu : bool, default=True
        Use CUDA/MPS if available.
    device : str, optional
        Explicit device override (e.g. "cpu", "cuda:0").
    verbose : bool, default=True
        Print progress.
    **backend_kwargs
        Extra keyword arguments forwarded to the underlying model constructor
        (e.g. ``split``, ``hidden``, ``dropout_p`` for jae1; ``patch_len``,
        ``d_model``, ``predictor_dim``, ``mask`` for jepa).

    Examples
    --------
    >>> model = JAE(latent_dim=6)
    >>> model.fit(noisy, epochs=200)
    >>> denoised = model.denoise(noisy)
    """

    def __init__(
        self,
        latent_dim: int | None = None,
        backend: str = "jae1",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        standardize: bool = True,
        seed: int = 0,
        use_gpu: bool = True,
        device: str | None = None,
        verbose: bool = True,
        **backend_kwargs,
    ) -> None:
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}.")

        self.latent_dim = latent_dim
        self.backend = backend
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.standardize = standardize
        self.seed = seed
        self.verbose = verbose
        self.backend_kwargs = dict(backend_kwargs)

        self.is_fitted = False
        self.input_channels_ = None
        self.n_timepoints_ = None
        self.model = None
        self.optimizer = None
        self.training_history_ = None
        self._loss_kwargs: dict = {}
        # Per-channel standardization stats, fit on the training data only.
        self.mean_ = None
        self.std_ = None

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = get_device(use_gpu=use_gpu, verbose=verbose)

    def _initialize_model(self, input_channels: int, n_timepoints: int) -> None:
        """Build the underlying model once the data shape is known."""
        self.input_channels_ = input_channels
        self.n_timepoints_ = n_timepoints

        if self.latent_dim is None:
            self.latent_dim = 64 if self.backend == "jepa" else max(input_channels // 8, 4)
            if self.verbose:
                print(f"Auto-detected latent_dim={self.latent_dim}")

        if input_channels < 2:
            raise ValueError(f"input_channels must be >= 2, got {input_channels}.")

        if self.backend == "jae1":
            self._loss_kwargs = {
                k: self.backend_kwargs.pop(k)
                for k in ("latent_weight",)
                if k in self.backend_kwargs
            }
            self.model = JAE1(
                input_dim=input_channels,
                latent_dim=self.latent_dim,
                seed=self.seed,
                **self.backend_kwargs,
            )
        else:
            jepa_loss_keys = ("lambda_pred", "lambda_var", "lambda_cov", "recon_weight")
            self._loss_kwargs = {
                k: self.backend_kwargs.pop(k) for k in jepa_loss_keys if k in self.backend_kwargs
            }
            self.model = JAE2(
                input_dim=input_channels,
                n_timepoints=n_timepoints,
                latent_dim=self.latent_dim,
                seed=self.seed,
                **self.backend_kwargs,
            )

        self.model = self.model.to(self.device)

        if self.verbose:
            print(f"Initialized JAE (backend={self.backend}):")
            print(f"  Input channels: {input_channels}")
            print(f"  Timepoints: {n_timepoints}")
            print(f"  Latent dimension: {self.latent_dim}")

    def fit(
        self,
        X,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool | None = None,
    ) -> "JAE":
        """Train the model on noisy neural data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_channels, n_timepoints)
            Noisy neural recordings.
        epochs : int, default=100
        batch_size : int, default=32
        verbose : bool, optional
            Override the instance verbose setting.

        Returns
        -------
        self
        """
        if verbose is None:
            verbose = self.verbose

        X = validate_input_data(X, name="X")
        n_samples, n_channels, seq_len = X.shape

        if self.model is None:
            self._initialize_model(n_channels, seq_len)

        # Fit per-channel standardization on the training data only. A linear
        # decoder plus zero-mean, unit-variance inputs give the reconstruction
        # loss a well-scaled gradient; without it the tiny raw signal scale
        # stalls training in a near-constant-output local minimum.
        if self.standardize:
            self.mean_ = X.mean(dim=(0, 2), keepdim=True)
            self.std_ = X.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
        X = self._apply_standardize(X)

        if n_samples < batch_size:
            batch_size = max(n_samples // 4, 2)
            if verbose:
                print(f"Reduced batch_size to {batch_size} for small dataset")

        loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True, drop_last=True)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if verbose:
            print(f"\nTraining for {epochs} epochs...")

        self.model.train()
        self.training_history_ = {"train_loss": []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(batch)
                loss = self.model.loss(out, **self._loss_kwargs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.training_history_["train_loss"].append(avg_loss)

            if verbose and ((epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0):
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        self.is_fitted = True
        if verbose:
            print("Training complete.")
        return self

    def denoise(self, X, batch_size: int = 64):
        """Denoise neural signals.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_channels, n_timepoints)
        batch_size : int, default=64

        Returns
        -------
        denoised : ndarray, shape (n_samples, n_channels, n_timepoints)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before denoising. Call fit() first.")

        X = validate_input_data(X, name="X")
        if X.shape[1] != self.input_channels_:
            raise ValueError(f"Expected {self.input_channels_} channels, got {X.shape[1]}.")

        X = self._apply_standardize(X)
        loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
        self.model.eval()
        out_batches = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                if hasattr(self.model, "denoise"):
                    denoised = self.model.denoise(batch)
                else:
                    denoised = self.model(batch).denoised
                out_batches.append(denoised.cpu())
        denoised = torch.cat(out_batches, dim=0)
        return self._invert_standardize(denoised).numpy()

    def _apply_standardize(self, X):
        """Standardize input with the fitted per-channel stats (no-op if disabled)."""
        if not self.standardize or self.mean_ is None:
            return X
        return (X - self.mean_) / self.std_

    def _invert_standardize(self, X):
        """Undo standardization on a denoised output (no-op if disabled)."""
        if not self.standardize or self.mean_ is None:
            return X
        return X * self.std_.cpu() + self.mean_.cpu()

    def fit_denoise(self, X, **fit_params):
        """Fit and denoise in one call."""
        self.fit(X, **fit_params)
        return self.denoise(X)

    def score(self, y_true, y_pred, per_channel: bool = False):
        """Variance Accounted For (VAF / R^2) between signals.

        Parameters
        ----------
        y_true, y_pred : array-like, shape (n_samples, n_channels, n_timepoints)
        per_channel : bool, default=False
            If True, return the full per-channel report dict from
            :func:`pyjae.metrics.per_channel_vaf`; otherwise return the mean VAF.
        """
        if per_channel:
            return per_channel_vaf(y_true, y_pred)
        return vaf(y_true, y_pred)

    def get_training_history(self):
        """Return the training loss history."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained.")
        return self.training_history_

    def save(self, path: str) -> None:
        """Save the model to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "backend": self.backend,
                "latent_dim": self.latent_dim,
                "input_channels": self.input_channels_,
                "n_timepoints": self.n_timepoints_,
                "seed": self.seed,
                "backend_kwargs": self.backend_kwargs,
                "loss_kwargs": self._loss_kwargs,
                "standardize": self.standardize,
                "mean_": self.mean_,
                "std_": self.std_,
            },
            path,
        )

    def load(self, path: str) -> "JAE":
        """Load a model from disk."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.backend = checkpoint["backend"]
        self.latent_dim = checkpoint["latent_dim"]
        self.seed = checkpoint.get("seed", 0)
        self.backend_kwargs = checkpoint.get("backend_kwargs", {})
        self.standardize = checkpoint.get("standardize", False)
        self.mean_ = checkpoint.get("mean_")
        self.std_ = checkpoint.get("std_")
        self._initialize_model(checkpoint["input_channels"], checkpoint["n_timepoints"])
        self._loss_kwargs = checkpoint.get("loss_kwargs", {})
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.is_fitted = True
        return self
