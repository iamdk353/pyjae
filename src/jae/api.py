"""
High-level API for the Joint Autoencoder (JAE).

Provides a scikit-learn-style interface for neural signal denoising.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from .models import JAE1, JAE2
from .losses import jae1_loss_fn, jae2_loss_fn
from .utils import get_device, validate_input_data


class JAE:
    """
    Joint Autoencoder for neural signal denoising.

    By default, implements the original algorithm from Altan et al. (2021):
    - 50/50 random partition of channels
    - Two parallel fully-connected autoencoders with ReLU activations
    - Loss: MSE(reconstruction) + MSE(latent alignment)

    Optional enhancements can be enabled via parameters.

    Parameters
    ----------
    latent_dim : int, optional
        Dimensionality of the latent space. If None, auto-detected as
        input_channels // 8.
    dropout_p : float, default=0.05
        Dropout probability at input layer (paper default).
    learning_rate : float, default=0.001
        Learning rate for ADAM optimizer (paper default).
    weight_decay : float, default=1e-5
        L2 regularization weight.
    use_unet : bool, default=False
        Use 1D U-Net architecture instead of fully-connected layers.
    use_vicreg : bool, default=False
        Use VICReg loss instead of MSE for latent alignment.
    num_networks : int, default=2
        Number of parallel networks. Original paper uses 2.
    subsample_fraction : float, default=0.5
        Fraction of channels per network. Original paper uses 0.5 (50/50 split).
    unet_channels : list, optional
        Channel dimensions for U-Net encoder levels. Default: [32, 64].
    use_gpu : bool, default=True
        Use GPU if available.
    device : str, optional
        Specific device (e.g., 'cuda:0', 'cpu').
    verbose : bool, default=True
        Print training progress.

    Examples
    --------
    Basic usage (original paper implementation):

    >>> model = JAE(latent_dim=6)
    >>> model.fit(noisy_data, epochs=100)
    >>> denoised = model.denoise(noisy_data)

    With enhancements:

    >>> model = JAE(latent_dim=6, use_unet=True, use_vicreg=True, num_networks=5)
    """

    def __init__(
        self,
        latent_dim=None,
        dropout_p=0.05,
        learning_rate=0.001,
        weight_decay=1e-5,
        use_unet=False,
        use_vicreg=False,
        num_networks=2,
        subsample_fraction=0.5,
        unet_channels=None,
        use_gpu=True,
        device=None,
        verbose=True,
    ):
        self.latent_dim = latent_dim
        self.dropout_p = dropout_p
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_unet = use_unet
        self.use_vicreg = use_vicreg
        self.num_networks = num_networks
        self.subsample_fraction = subsample_fraction
        self.unet_channels = unet_channels or [32, 64]
        self.verbose = verbose

        self.is_fitted = False
        self.input_channels_ = None
        self.model = None
        self.optimizer = None
        self.training_history_ = None

        # Device setup
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = get_device(use_gpu=use_gpu, verbose=verbose)

    def _use_enhanced_model(self):
        """Check if we should use the enhanced (JAE2-style) model."""
        return self.use_unet or self.use_vicreg or self.num_networks > 2

    def _initialize_model(self, input_channels):
        """Initialize the model architecture."""
        self.input_channels_ = input_channels

        # Auto-detect latent_dim if not specified
        if self.latent_dim is None:
            self.latent_dim = max(input_channels // 8, 4)
            if self.verbose:
                print(f"Auto-detected latent_dim={self.latent_dim}")

        # Validate parameters
        if input_channels < 2:
            raise ValueError(f"input_channels must be >= 2, got {input_channels}")

        if not self._use_enhanced_model() and input_channels % 2 != 0:
            raise ValueError(
                f"Original JAE requires even number of channels for 50/50 split. "
                f"Got {input_channels}. Use num_networks > 2 or enable enhancements."
            )

        # Create model
        if self._use_enhanced_model():
            self.model = JAE2(
                input_dim=input_channels,
                latent_dim=self.latent_dim,
                num_networks=self.num_networks,
                subsample_fraction=self.subsample_fraction,
                unet_channels=self.unet_channels,
            )
        else:
            self.model = JAE1(
                input_dim=input_channels,
                latent_dim=self.latent_dim,
                dropout_p=self.dropout_p,
            )

        self.model = self.model.to(self.device)

        if self.verbose:
            mode = "enhanced" if self._use_enhanced_model() else "original"
            print(f"Initialized JAE ({mode} mode):")
            print(f"  Input channels: {input_channels}")
            print(f"  Latent dimension: {self.latent_dim}")
            if self._use_enhanced_model():
                print(f"  Networks: {self.num_networks}")
                print(f"  U-Net: {self.use_unet}")
                print(f"  VICReg: {self.use_vicreg}")

    def fit(self, X, epochs=100, batch_size=32, validation_split=0.0, verbose=None):
        """
        Train the JAE on noisy neural data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_channels, n_timepoints)
            Noisy neural recordings.
        epochs : int, default=100
            Number of training epochs.
        batch_size : int, default=32
            Batch size for training.
        validation_split : float, default=0.0
            Fraction of data for validation.
        verbose : bool, optional
            Override instance verbose setting.

        Returns
        -------
        self
        """
        if verbose is None:
            verbose = self.verbose

        X = validate_input_data(X, name="X")
        n_samples, n_channels, seq_len = X.shape

        if self.model is None:
            self._initialize_model(n_channels)

        # Adjust batch size if needed
        if n_samples < batch_size:
            batch_size = max(n_samples // 4, 2)
            if verbose:
                print(f"Reduced batch_size to {batch_size} for small dataset")

        # Create data loaders
        if validation_split > 0:
            n_train = int(n_samples * (1 - validation_split))
            train_dataset = TensorDataset(X[:n_train], X[:n_train])
            val_dataset = TensorDataset(X[n_train:], X[n_train:])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            train_dataset = TensorDataset(X, X)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = None

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if verbose:
            print(f"\nTraining for {epochs} epochs...")

        self.model.train()
        self.training_history_ = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0

            for batch_data, _ in train_loader:
                batch_data = batch_data.to(self.device)
                self.optimizer.zero_grad()

                # Forward pass and loss
                if self._use_enhanced_model():
                    _, reconstructions, latents, targets = self.model(batch_data)
                    loss = jae2_loss_fn(
                        reconstructions,
                        latents,
                        targets,
                        use_vicreg=self.use_vicreg,
                    )
                else:
                    x_denoised, z1, z2, x1_target, x2_target = self.model(batch_data)
                    half_dim = self.model.half_dim
                    x1_hat = x_denoised[:, :half_dim, :]
                    x2_hat = x_denoised[:, half_dim:, :]
                    loss = jae1_loss_fn(x1_hat, x2_hat, z1, z2, x1_target, x2_target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.training_history_["train_loss"].append(avg_loss)

            # Validation
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                self.training_history_["val_loss"].append(val_loss)

            # Print progress
            if verbose and ((epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0):
                msg = f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}"
                if val_loader is not None:
                    msg += f", Val: {val_loss:.4f}"
                print(msg)

        self.is_fitted = True
        if verbose:
            print("Training complete.")

        return self

    def _validate(self, val_loader):
        """Compute validation loss."""
        self.model.eval()
        val_loss = 0
        n_batches = 0

        with torch.no_grad():
            for batch_data, _ in val_loader:
                batch_data = batch_data.to(self.device)

                if self._use_enhanced_model():
                    _, reconstructions, latents, targets = self.model(batch_data)
                    loss = jae2_loss_fn(reconstructions, latents, targets, use_vicreg=self.use_vicreg)
                else:
                    x_denoised, z1, z2, x1_target, x2_target = self.model(batch_data)
                    half_dim = self.model.half_dim
                    x1_hat = x_denoised[:, :half_dim, :]
                    x2_hat = x_denoised[:, half_dim:, :]
                    loss = jae1_loss_fn(x1_hat, x2_hat, z1, z2, x1_target, x2_target)

                val_loss += loss.item()
                n_batches += 1

        self.model.train()
        return val_loss / n_batches

    def denoise(self, X, batch_size=None):
        """
        Denoise neural signals.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_channels, n_timepoints)
            Noisy neural recordings.
        batch_size : int, optional
            Batch size for inference.

        Returns
        -------
        denoised : ndarray, shape (n_samples, n_channels, n_timepoints)
            Denoised signals.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before denoising. Call fit() first.")

        X = validate_input_data(X, name="X")

        if X.shape[1] != self.input_channels_:
            raise ValueError(
                f"Expected {self.input_channels_} channels, got {X.shape[1]}"
            )

        if batch_size is None:
            batch_size = 32

        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        denoised_batches = []

        with torch.no_grad():
            for (batch_data,) in loader:
                batch_data = batch_data.to(self.device)
                denoised = self.model(batch_data)[0]
                denoised_batches.append(denoised.cpu())

        return torch.cat(denoised_batches, dim=0).numpy()

    def fit_denoise(self, X, **fit_params):
        """Fit and denoise in one call."""
        self.fit(X, **fit_params)
        return self.denoise(X)

    def score(self, y_true, y_pred):
        """
        Compute Variance Accounted For (VAF / R²) between signals.

        This is the primary evaluation metric from the paper.

        Parameters
        ----------
        y_true : array-like
            Ground truth (noise-free) signals.
        y_pred : array-like
            Predicted (denoised) signals.

        Returns
        -------
        vaf : float
            Variance Accounted For (R² score).
        """
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

    def get_training_history(self):
        """Get training loss history."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained.")
        return self.training_history_

    def save(self, path):
        """Save model to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model.")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "latent_dim": self.latent_dim,
                "input_channels": self.input_channels_,
                "dropout_p": self.dropout_p,
                "use_unet": self.use_unet,
                "use_vicreg": self.use_vicreg,
                "num_networks": self.num_networks,
                "subsample_fraction": self.subsample_fraction,
                "unet_channels": self.unet_channels,
            },
            path,
        )

    def load(self, path):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.latent_dim = checkpoint["latent_dim"]
        self.dropout_p = checkpoint.get("dropout_p", 0.05)
        self.use_unet = checkpoint.get("use_unet", False)
        self.use_vicreg = checkpoint.get("use_vicreg", False)
        self.num_networks = checkpoint.get("num_networks", 2)
        self.subsample_fraction = checkpoint.get("subsample_fraction", 0.5)
        self.unet_channels = checkpoint.get("unet_channels", [32, 64])

        self._initialize_model(checkpoint["input_channels"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.is_fitted = True
