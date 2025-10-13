"""
High-level API for JAE models.

This module provides a scikit-learn-style interface for easy neural signal denoising.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .models import JAE1, JAE2
from .losses import jae1_loss_fn, jae2_loss_fn
from .utils import get_device, validate_input_data
from .config import (
    get_default_config,
    auto_detect_latent_dim,
    validate_model_params,
    validate_training_params,
    suggest_batch_size,
    print_config_summary
)


class JAE:
    """
    Joint Autoencoder for neural signal denoising.

    This class provides a high-level, scikit-learn-style API for training and using
    joint autoencoders to denoise multi-channel neural recordings.

    Args:
        model_type (str, optional): Type of model to use ('jae1' or 'jae2').
            Default: 'jae2'.
        latent_dim (int, optional): Dimensionality of latent space. If None, will be
            auto-detected as input_channels // 8. Default: None.
        num_networks (int, optional): Number of parallel networks (JAE2 only).
            Default: 5.
        subsample_fraction (float, optional): Fraction of neurons to subsample per
            network (JAE2 only). Default: 0.8.
        unet_channels (list of int, optional): U-Net channel dimensions (JAE2 only).
            Default: [32, 64].
        dropout_p (float, optional): Dropout probability (JAE1 only). Default: 0.05.
        use_gpu (bool, optional): Whether to use GPU if available. Default: True.
        device (str or torch.device, optional): Specific device to use. If None,
            auto-detects based on use_gpu. Default: None.
        verbose (bool, optional): Whether to print training progress and warnings.
            Default: True.
        **kwargs: Additional model-specific parameters.

    Attributes:
        model_type (str): The model type being used.
        latent_dim (int): Latent space dimensionality.
        device (torch.device): Device being used for computation.
        model (nn.Module): The underlying neural network model.
        is_fitted (bool): Whether the model has been trained.

    Example:
        >>> from jae import JAE
        >>> import numpy as np
        >>>
        >>> # Generate or load your noisy neural data
        >>> noisy_data = np.random.randn(100, 96, 128)  # (samples, channels, time)
        >>>
        >>> # Create and train the model
        >>> model = JAE(latent_dim=12, use_gpu=True)
        >>> model.fit(noisy_data, epochs=100)
        >>>
        >>> # Denoise the signals
        >>> clean_data = model.denoise(noisy_data)
    """

    def __init__(
        self,
        model_type='jae2',
        latent_dim=None,
        num_networks=5,
        subsample_fraction=0.8,
        unet_channels=None,
        dropout_p=0.05,
        use_gpu=True,
        device=None,
        verbose=True,
        **kwargs
    ):
        self.model_type = model_type.lower()
        self.latent_dim = latent_dim  # May be None initially
        self.verbose = verbose
        self.is_fitted = False
        self.input_channels_ = None  # Will be set during fit

        # Get default config
        self.config = get_default_config(self.model_type)

        # Override defaults with user-provided values
        if latent_dim is not None:
            self.config.latent_dim = latent_dim

        if self.model_type == 'jae1':
            self.config.dropout_p = dropout_p
        elif self.model_type == 'jae2':
            self.config.num_networks = num_networks
            self.config.subsample_fraction = subsample_fraction
            if unet_channels is not None:
                self.config.unet_channels = unet_channels

        # Additional kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Device setup
        if device is not None:
            self.device = torch.device(device)
            if self.verbose:
                print(f"Using specified device: {self.device}")
        else:
            self.device = get_device(use_gpu=use_gpu, verbose=self.verbose)

        self.model = None
        self.optimizer = None

    def _initialize_model(self, input_channels):
        """
        Initialize the model architecture based on input dimensions.

        Args:
            input_channels (int): Number of input channels.
        """
        self.input_channels_ = input_channels

        # Auto-detect latent_dim if not specified
        if self.config.latent_dim is None:
            self.latent_dim = auto_detect_latent_dim(input_channels, warn=self.verbose)
            self.config.latent_dim = self.latent_dim
        else:
            self.latent_dim = self.config.latent_dim

        # Validate parameters
        if self.model_type == 'jae1':
            validate_model_params(
                self.model_type,
                input_channels,
                self.latent_dim
            )
        elif self.model_type == 'jae2':
            validate_model_params(
                self.model_type,
                input_channels,
                self.latent_dim,
                num_networks=self.config.num_networks,
                subsample_fraction=self.config.subsample_fraction
            )

        # Create model
        if self.model_type == 'jae1':
            self.model = JAE1(
                input_dim=input_channels,
                latent_dim=self.latent_dim,
                dropout_p=self.config.dropout_p
            )
        elif self.model_type == 'jae2':
            self.model = JAE2(
                input_dim=input_channels,
                latent_dim=self.latent_dim,
                num_networks=self.config.num_networks,
                subsample_fraction=self.config.subsample_fraction,
                unet_channels=self.config.unet_channels
            )

        self.model = self.model.to(self.device)

        if self.verbose:
            print(f"\nInitialized {self.model_type.upper()} model:")
            print(f"  Input channels: {input_channels}")
            print(f"  Latent dimension: {self.latent_dim}")
            if self.model_type == 'jae2':
                print(f"  Number of networks: {self.config.num_networks}")
                print(f"  Subsample fraction: {self.config.subsample_fraction}")

    def fit(
        self,
        X,
        epochs=None,
        batch_size=None,
        learning_rate=None,
        verbose=None,
        validation_split=0.0
    ):
        """
        Train the JAE model on noisy neural data.

        Args:
            X (np.ndarray or torch.Tensor): Noisy neural data with shape
                (n_samples, n_channels, seq_len).
            epochs (int, optional): Number of training epochs. If None, uses
                config default. Default: None.
            batch_size (int, optional): Batch size for training. If None, uses
                config default. Default: None.
            learning_rate (float, optional): Learning rate for optimizer. If None,
                uses config default. Default: None.
            verbose (bool, optional): Whether to print training progress. If None,
                uses instance verbose setting. Default: None.
            validation_split (float, optional): Fraction of data to use for validation.
                Default: 0.0 (no validation).

        Returns:
            self: Returns self for method chaining.

        Example:
            >>> model = JAE()
            >>> model.fit(noisy_data, epochs=150, batch_size=64)
            >>> # Model is now trained and ready for denoising
        """
        # Handle verbose setting
        if verbose is None:
            verbose = self.verbose

        # Validate and convert input data
        X = validate_input_data(X, name="X")
        n_samples, n_channels, seq_len = X.shape

        # Initialize model if not already done
        if self.model is None:
            self._initialize_model(n_channels)

        # Set training parameters
        if epochs is None:
            epochs = self.config.epochs
        if batch_size is None:
            batch_size = suggest_batch_size(n_samples, default=self.config.batch_size)
        if learning_rate is None:
            learning_rate = self.config.learning_rate

        # Validate training parameters
        validate_training_params(epochs, batch_size, learning_rate)

        # Create data loaders
        if validation_split > 0:
            n_train = int(n_samples * (1 - validation_split))
            X_train = X[:n_train]
            X_val = X[n_train:]

            train_dataset = TensorDataset(X_train, X_train)
            val_dataset = TensorDataset(X_val, X_val)

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True  # Important for BatchNorm
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        else:
            train_dataset = TensorDataset(X, X)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True
            )
            val_loader = None

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Training loop
        if verbose:
            print(f"\nTraining {self.model_type.upper()} model...")
            print(f"  Epochs: {epochs}")
            print(f"  Batch size: {batch_size}")
            print(f"  Learning rate: {learning_rate}")
            print(f"  Training samples: {len(train_loader.dataset)}")
            if val_loader:
                print(f"  Validation samples: {len(val_loader.dataset)}")

        self.model.train()
        self.training_history_ = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0

            for batch_idx, (batch_data, _) in enumerate(train_loader):
                batch_data = batch_data.to(self.device)
                self.optimizer.zero_grad()

                # Forward pass and loss computation
                if self.model_type == 'jae1':
                    x_denoised, z1, z2, x1_target, x2_target = self.model(batch_data)
                    half_dim = self.model.half_dim
                    x1_hat = x_denoised[:, :half_dim, :]
                    x2_hat = x_denoised[:, half_dim:, :]
                    loss = jae1_loss_fn(
                        x1_hat, x2_hat, z1, z2, x1_target, x2_target,
                        latent_weight=self.config.latent_weight
                    )

                elif self.model_type == 'jae2':
                    denoised, reconstructions, latents, targets = self.model(batch_data)
                    loss = jae2_loss_fn(
                        reconstructions, latents, targets,
                        denoised_output=denoised,  # Pass for smoothness regularization
                        recon_weight=self.config.recon_weight,
                        vicreg_weight=self.config.vicreg_weight,
                        smoothness_weight=self.config.smoothness_weight,
                        huber_delta=self.config.huber_delta,
                        lambda_inv=self.config.lambda_inv,
                        mu_var=self.config.mu_var,
                        nu_cov=self.config.nu_cov
                    )

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches
            self.training_history_['train_loss'].append(avg_train_loss)

            # Validation
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                self.training_history_['val_loss'].append(val_loss)

            # Print progress
            if verbose and ((epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0):
                msg = f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}"
                if val_loader is not None:
                    msg += f", Val Loss: {val_loss:.4f}"
                print(msg)

        self.is_fitted = True

        if verbose:
            print(f"\nTraining complete!")

        return self

    def _validate(self, val_loader):
        """Run validation and return average loss."""
        self.model.eval()
        val_loss = 0
        n_batches = 0

        with torch.no_grad():
            for batch_data, _ in val_loader:
                batch_data = batch_data.to(self.device)

                if self.model_type == 'jae1':
                    x_denoised, z1, z2, x1_target, x2_target = self.model(batch_data)
                    half_dim = self.model.half_dim
                    x1_hat = x_denoised[:, :half_dim, :]
                    x2_hat = x_denoised[:, half_dim:, :]
                    loss = jae1_loss_fn(
                        x1_hat, x2_hat, z1, z2, x1_target, x2_target,
                        latent_weight=self.config.latent_weight
                    )
                elif self.model_type == 'jae2':
                    denoised, reconstructions, latents, targets = self.model(batch_data)
                    loss = jae2_loss_fn(
                        reconstructions, latents, targets,
                        denoised_output=denoised,
                        recon_weight=self.config.recon_weight,
                        vicreg_weight=self.config.vicreg_weight,
                        smoothness_weight=self.config.smoothness_weight
                    )

                val_loss += loss.item()
                n_batches += 1

        self.model.train()
        return val_loss / n_batches

    def denoise(self, X, batch_size=None):
        """
        Denoise neural signals using the trained model.

        Args:
            X (np.ndarray or torch.Tensor): Noisy neural data with shape
                (n_samples, n_channels, seq_len).
            batch_size (int, optional): Batch size for inference. If None, uses
                training batch size. Default: None.

        Returns:
            np.ndarray: Denoised neural data with same shape as input.

        Raises:
            RuntimeError: If model hasn't been trained yet.

        Example:
            >>> model = JAE()
            >>> model.fit(train_data)
            >>> clean_test = model.denoise(test_data)
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before calling denoise(). "
                "Call fit() first or use fit_denoise()."
            )

        # Validate input
        X = validate_input_data(X, name="X")

        if X.shape[1] != self.input_channels_:
            raise ValueError(
                f"Input has {X.shape[1]} channels but model was trained with "
                f"{self.input_channels_} channels."
            )

        # Set batch size
        if batch_size is None:
            batch_size = self.config.batch_size

        # Create data loader
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Denoise
        self.model.eval()
        denoised_batches = []

        with torch.no_grad():
            for (batch_data,) in loader:
                batch_data = batch_data.to(self.device)
                # First output is always the denoised signal
                denoised = self.model(batch_data)[0]
                denoised_batches.append(denoised.cpu())

        # Concatenate and convert to numpy
        denoised = torch.cat(denoised_batches, dim=0)
        return denoised.numpy()

    def fit_denoise(self, X, **fit_params):
        """
        Convenience method to fit and denoise in one call.

        Args:
            X (np.ndarray or torch.Tensor): Noisy neural data.
            **fit_params: Parameters to pass to fit().

        Returns:
            np.ndarray: Denoised neural data.

        Example:
            >>> model = JAE()
            >>> clean_data = model.fit_denoise(noisy_data, epochs=100)
        """
        self.fit(X, **fit_params)
        return self.denoise(X)

    def get_training_history(self):
        """
        Get training history (loss curves).

        Returns:
            dict: Dictionary with 'train_loss' and optionally 'val_loss' keys.

        Raises:
            RuntimeError: If model hasn't been trained yet.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained yet.")
        return self.training_history_

    def save(self, path):
        """
        Save model weights to disk.

        Args:
            path (str): File path to save the model.

        Example:
            >>> model.save('my_jae_model.pth')
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model.")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'latent_dim': self.latent_dim,
            'input_channels': self.input_channels_,
            'config': self.config
        }, path)

        if self.verbose:
            print(f"Model saved to {path}")

    def load(self, path):
        """
        Load model weights from disk.

        Args:
            path (str): File path to load the model from.

        Example:
            >>> model = JAE()
            >>> model.load('my_jae_model.pth')
            >>> clean = model.denoise(noisy_data)
        """
        # Set weights_only=False because we save custom config objects
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Restore model type and config
        self.model_type = checkpoint['model_type']
        self.config = checkpoint['config']
        self.latent_dim = checkpoint['latent_dim']

        # Initialize model with saved architecture
        self._initialize_model(checkpoint['input_channels'])

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.is_fitted = True

        if self.verbose:
            print(f"Model loaded from {path}")

