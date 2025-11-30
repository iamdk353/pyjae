"""
Original Joint Autoencoder implementation from Altan et al. (2021).
"""

import torch
import torch.nn as nn


class SimpleAutoencoder(nn.Module):
    """
    Simple fully-connected autoencoder with ReLU activations.

    Architecture per paper: Input -> Linear -> ReLU -> Linear -> ReLU -> Output
    Dropout applied at input layer (p=0.05 default).
    """

    def __init__(self, input_dim, latent_dim, dropout_p=0.05):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor, shape (batch, input_dim)

        Returns
        -------
        x_hat : Tensor, shape (batch, input_dim)
            Reconstruction.
        z : Tensor, shape (batch, latent_dim)
            Latent representation.
        """
        x = self.dropout(x)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class JAE1(nn.Module):
    """
    Joint Autoencoder from Altan et al. (2021), Fig 2.

    Splits channels into two partitions (X1, X2), processes each through
    a separate autoencoder, and trains with loss:

        C = MSE(X1, X̂1) + MSE(X2, X̂2) + ||Z1 - Z2||²

    Parameters
    ----------
    input_dim : int
        Total number of channels. Must be even.
    latent_dim : int
        Latent space dimensionality (D in paper).
    dropout_p : float, default=0.05
        Input dropout probability.
    """

    def __init__(self, input_dim, latent_dim, dropout_p=0.05):
        super().__init__()
        if input_dim % 2 != 0:
            raise ValueError("JAE1 requires even number of input channels.")

        self.half_dim = input_dim // 2
        self.ae1 = SimpleAutoencoder(self.half_dim, latent_dim, dropout_p)
        self.ae2 = SimpleAutoencoder(self.half_dim, latent_dim, dropout_p)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor, shape (batch, channels, time)

        Returns
        -------
        x_denoised : Tensor, shape (batch, channels, time)
        z1 : Tensor, shape (batch*time, latent_dim)
        z2 : Tensor, shape (batch*time, latent_dim)
        x1_target : Tensor, shape (batch, half_dim, time)
        x2_target : Tensor, shape (batch, half_dim, time)
        """
        B, C, T = x.shape

        # Split channels into two partitions
        x1 = x[:, : self.half_dim, :]
        x2 = x[:, self.half_dim :, :]

        # Reshape for FC layers: (B, C, T) -> (B*T, C)
        x1_flat = x1.permute(0, 2, 1).reshape(B * T, self.half_dim)
        x2_flat = x2.permute(0, 2, 1).reshape(B * T, self.half_dim)

        # Process through autoencoders
        x1_hat_flat, z1 = self.ae1(x1_flat)
        x2_hat_flat, z2 = self.ae2(x2_flat)

        # Reshape back: (B*T, C) -> (B, C, T)
        x1_hat = x1_hat_flat.reshape(B, T, self.half_dim).permute(0, 2, 1)
        x2_hat = x2_hat_flat.reshape(B, T, self.half_dim).permute(0, 2, 1)

        x_denoised = torch.cat((x1_hat, x2_hat), dim=1)

        return x_denoised, z1, z2, x1, x2
