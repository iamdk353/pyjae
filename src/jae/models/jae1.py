"""
Original Joint Autoencoder (JAE1) implementation.

Based on Altan, E., Solla, S. A., Miller, L. E., & Perreault, E. J. (2021).
"Estimating the dimensionality of the manifold underlying multi-electrode 
neural recordings." PLOS Computational Biology, 17(11), e1008591.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAutoencoder(nn.Module):
    """
    A simple, shallow, fully connected autoencoder as depicted in the 2021 paper.

    This autoencoder processes time points independently using fully connected layers
    with ReLU activations and optional dropout at the input layer.

    Args:
        input_dim (int): Number of input features (channels).
        latent_dim (int): Dimensionality of the latent space.
        dropout_p (float, optional): Dropout probability at input layer. Default: 0.05.

    Attributes:
        dropout (nn.Dropout): Dropout layer applied at input.
        encoder (nn.Sequential): Encoder network (Linear + ReLU).
        decoder (nn.Sequential): Decoder network (Linear + ReLU).

    Example:
        >>> ae = SimpleAutoencoder(input_dim=48, latent_dim=12)
        >>> x = torch.randn(32, 48)  # (batch, features)
        >>> x_hat, z = ae(x)
        >>> print(x_hat.shape, z.shape)  # torch.Size([32, 48]) torch.Size([32, 12])
    """

    def __init__(self, input_dim, latent_dim, dropout_p=0.05):
        super(SimpleAutoencoder, self).__init__()
        self.dropout = nn.Dropout(dropout_p)
        # Encoder and Decoder use ReLU as specified in the paper
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, input_dim).

        Returns:
            tuple: (x_hat, z) where
                - x_hat (torch.Tensor): Reconstructed input, shape (batch, input_dim).
                - z (torch.Tensor): Latent representation, shape (batch, latent_dim).
        """
        # Dropout is applied at the input layer
        x = self.dropout(x)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class JAE1(nn.Module):
    """
    Original Joint Autoencoder (Altan et al., 2021 implementation).

    JAE1 splits input channels into two fixed groups (50/50 split) and processes
    them through two parallel autoencoders. The loss function encourages both
    accurate reconstruction and alignment of the latent representations.

    Args:
        input_dim (int): Total number of input channels. Must be even.
        latent_dim (int): Dimensionality of the latent space for each autoencoder.
        dropout_p (float, optional): Dropout probability. Default: 0.05.

    Attributes:
        half_dim (int): Half of input_dim (channels per autoencoder).
        ae1 (SimpleAutoencoder): First autoencoder (processes first half of channels).
        ae2 (SimpleAutoencoder): Second autoencoder (processes second half of channels).

    Raises:
        ValueError: If input_dim is not even.

    Example:
        >>> jae = JAE1(input_dim=96, latent_dim=12)
        >>> x = torch.randn(16, 96, 128)  # (batch, channels, time)
        >>> x_denoised, z1, z2, x1_target, x2_target = jae(x)
        >>> print(x_denoised.shape)  # torch.Size([16, 96, 128])
    """

    def __init__(self, input_dim, latent_dim, dropout_p=0.05):
        super(JAE1, self).__init__()
        if input_dim % 2 != 0:
            raise ValueError("JAE1 requires an even number of input dimensions.")

        self.half_dim = input_dim // 2
        self.ae1 = SimpleAutoencoder(self.half_dim, latent_dim, dropout_p)
        self.ae2 = SimpleAutoencoder(self.half_dim, latent_dim, dropout_p)

    def forward(self, x):
        """
        Forward pass through JAE1.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, time).

        Returns:
            tuple: (x_denoised, z1, z2, x1_target, x2_target) where
                - x_denoised (torch.Tensor): Reconstructed signal, shape (batch, channels, time).
                - z1 (torch.Tensor): Latent from first AE, shape (batch*time, latent_dim).
                - z2 (torch.Tensor): Latent from second AE, shape (batch*time, latent_dim).
                - x1_target (torch.Tensor): First half input, shape (batch, half_dim, time).
                - x2_target (torch.Tensor): Second half input, shape (batch, half_dim, time).
        """
        # Input shape: (Batch, Channels, Time)
        B, C, T = x.shape

        # Split the channels (dim=1) into two fixed sets
        x1 = x[:, :self.half_dim, :]
        x2 = x[:, self.half_dim:, :]

        # To apply FC layers independently at each time step, we must reshape.
        # (Batch, Channels, Time) -> (Batch, Time, Channels) -> (Batch*Time, Channels)
        x1_flat = x1.permute(0, 2, 1).reshape(B * T, self.half_dim)
        x2_flat = x2.permute(0, 2, 1).reshape(B * T, self.half_dim)

        # Process through networks
        x1_hat_flat, z1 = self.ae1(x1_flat)
        x2_hat_flat, z2 = self.ae2(x2_flat)

        # Reshape back to (Batch, Time, Channels) -> (Batch, Channels, Time)
        x1_hat = x1_hat_flat.reshape(B, T, self.half_dim).permute(0, 2, 1)
        x2_hat = x2_hat_flat.reshape(B, T, self.half_dim).permute(0, 2, 1)

        x_denoised = torch.cat((x1_hat, x2_hat), dim=1)

        # Return denoised output, latents (for loss), and targets (for loss)
        return x_denoised, z1, z2, x1, x2

