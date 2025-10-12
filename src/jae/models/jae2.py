"""
Modernized Joint Autoencoder (JAE2) implementation.

JAE2 incorporates several state-of-the-art enhancements:
- N-way parallel networks with stochastic neuron subsampling
- 1D U-Net architecture with skip connections for temporal context
- VICReg-compatible latent representations
- Improved robustness and generalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    """
    1D convolutional block: (Conv1d -> BatchNorm -> LeakyReLU) x 2.

    This block is the building block for the U-Net encoder and decoder paths.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Convolution kernel size. Default: 3.
        padding (int, optional): Padding size. Default: 1.

    Example:
        >>> block = ConvBlock1D(32, 64)
        >>> x = torch.randn(8, 32, 128)
        >>> out = block(x)
        >>> print(out.shape)  # torch.Size([8, 64, 128])
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock1D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        """
        Forward pass through the convolutional block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, time).

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_channels, time).
        """
        return self.block(x)


class UNet1D(nn.Module):
    """
    1D U-Net architecture for time-series denoising.

    U-Net processes temporal signals with an encoder-decoder structure and skip
    connections that preserve high-frequency details. This is particularly effective
    for neural signal denoising where spike timing matters.

    Args:
        input_channels (int): Number of input channels (neurons).
        channels (list of int, optional): Channel dimensions for each encoder level.
            Default: [64, 128].
        latent_dim (int, optional): Dimensionality of the latent representation.
            Default: 64.

    Attributes:
        enc_blocks (nn.ModuleList): Encoder convolutional blocks.
        pool (nn.MaxPool1d): Max pooling for downsampling.
        bottleneck (ConvBlock1D): Bottleneck layer at the lowest resolution.
        projector (nn.Sequential): Projects bottleneck to latent representation.
        up_convs (nn.ModuleList): Transposed convolutions for upsampling.
        dec_blocks (nn.ModuleList): Decoder convolutional blocks.
        final_conv (nn.Conv1d): Final 1x1 convolution to output channels.

    Example:
        >>> unet = UNet1D(input_channels=48, channels=[32, 64], latent_dim=12)
        >>> x = torch.randn(8, 48, 128)
        >>> recon, latent = unet(x)
        >>> print(recon.shape, latent.shape)  # torch.Size([8, 48, 128]) torch.Size([8, 12])
    """

    def __init__(self, input_channels, channels=[64, 128], latent_dim=64):
        super(UNet1D, self).__init__()
        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.pool = nn.MaxPool1d(2)
        in_ch = input_channels
        for ch in channels:
            self.enc_blocks.append(ConvBlock1D(in_ch, ch))
            in_ch = ch

        # Bottleneck
        bottleneck_channels = channels[-1] * 2
        self.bottleneck = ConvBlock1D(channels[-1], bottleneck_channels)

        # Latent Projector (Pools across time for VICReg)
        # This creates a single representation for the entire time segment.
        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            # MLP Projector Head
            nn.Linear(bottleneck_channels, bottleneck_channels),
            nn.BatchNorm1d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_channels, latent_dim)
        )

        # Decoder
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        rev_channels = channels[::-1]
        up_in_ch = bottleneck_channels

        for ch in rev_channels:
            self.up_convs.append(nn.ConvTranspose1d(up_in_ch, ch, kernel_size=2, stride=2))
            self.dec_blocks.append(ConvBlock1D(ch * 2, ch))
            up_in_ch = ch

        self.final_conv = nn.Conv1d(channels[0], input_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, time).

        Returns:
            tuple: (reconstruction, latent) where
                - reconstruction (torch.Tensor): Denoised signal, shape (batch, channels, time).
                - latent (torch.Tensor): Latent representation, shape (batch, latent_dim).
        """
        # Encoder
        skip_connections = []
        for block in self.enc_blocks:
            x = block(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck & Latent
        x = self.bottleneck(x)
        latent = self.projector(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for i in range(len(self.dec_blocks)):
            x = self.up_convs[i](x)
            # Handle potential dimension mismatch along the time axis due to pooling
            if x.shape[2] != skip_connections[i].shape[2]:
                x = F.interpolate(x, size=skip_connections[i].shape[2], mode='linear', align_corners=False)
            x = torch.cat((x, skip_connections[i]), dim=1)
            x = self.dec_blocks[i](x)

        reconstruction = self.final_conv(x)
        # Apply ReLU to match the non-negativity constraint of neural data
        reconstruction = F.relu(reconstruction)
        return reconstruction, latent


class JAE2(nn.Module):
    """
    Modernized Joint Autoencoder (JAE2).

    JAE2 extends the original JAE with multiple enhancements:
    - N parallel U-Net networks (not limited to 2)
    - Stochastic neuron subsampling during training
    - Temporal context awareness via convolutional architecture
    - Produces rotation/scale-invariant latent representations

    Args:
        input_dim (int): Total number of input channels (neurons).
        latent_dim (int, optional): Dimensionality of latent space. Default: 64.
        num_networks (int, optional): Number of parallel networks. Default: 5.
        subsample_fraction (float, optional): Fraction of neurons to sample per network.
            Default: 0.8.
        unet_channels (list of int, optional): U-Net encoder channel dimensions.
            Default: [32, 64].

    Attributes:
        input_dim (int): Number of input channels.
        subsample_dim (int): Number of channels per subsampled view.
        networks (nn.ModuleList): List of parallel U-Net networks.

    Raises:
        ValueError: If subsample_fraction is too low or input_dim too small.

    Example:
        >>> jae2 = JAE2(input_dim=96, latent_dim=12, num_networks=5)
        >>> x = torch.randn(16, 96, 128)
        >>> denoised, recons, latents, inputs = jae2(x)
        >>> print(denoised.shape)  # torch.Size([16, 96, 128])
        >>> print(len(latents))  # 5
    """

    def __init__(
        self,
        input_dim,
        latent_dim=64,
        num_networks=5,
        subsample_fraction=0.8,
        unet_channels=[32, 64]
    ):
        super(JAE2, self).__init__()
        self.input_dim = input_dim
        self.subsample_dim = int(input_dim * subsample_fraction)
        if self.subsample_dim == 0:
            raise ValueError("Subsample fraction is too low or input dimension too small.")

        self.networks = nn.ModuleList([
            UNet1D(self.subsample_dim, channels=unet_channels, latent_dim=latent_dim)
            for _ in range(num_networks)
        ])

    def forward(self, x):
        """
        Forward pass through JAE2.

        Each network processes a randomly sampled subset of neurons. The final output
        is an average of all network reconstructions for the sampled neurons.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, time).

        Returns:
            tuple: (denoised_output, reconstructions_list, latents, inputs_subsampled) where
                - denoised_output (torch.Tensor): Averaged denoised signal, shape (batch, channels, time).
                - reconstructions_list (list): Partial reconstructions from each network.
                - latents (list): Latent representations from each network.
                - inputs_subsampled (list): Subsampled inputs to each network.
        """
        B, C, T = x.shape
        latents = []
        reconstructions_list = []
        inputs_subsampled = []

        # Prepare tensors for efficient reconstruction averaging
        full_recon = torch.zeros(B, C, T, device=x.device)
        counts = torch.zeros(1, C, 1, device=x.device)

        for network in self.networks:
            # 1. Randomly sample neurons (channels)
            indices = torch.randperm(self.input_dim, device=x.device)[:self.subsample_dim]
            x_sub = x[:, indices, :]
            inputs_subsampled.append(x_sub)

            # 2. Pass through the network
            recon, latent = network(x_sub)
            reconstructions_list.append(recon)
            latents.append(latent)

            # 3. Efficiently combine reconstructions using scatter_add_
            # Expand indices shape for broadcasting: (subsample_dim) -> (B, subsample_dim, T)
            indices_expanded = indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, T)
            full_recon.scatter_add_(1, indices_expanded, recon)
            counts[0, indices, 0] += 1

        # Average the reconstructions
        counts[counts == 0] = 1  # Avoid division by zero for unsampled neurons
        denoised_output = full_recon / counts

        # Return averaged output, and lists of partial results for loss calculation
        return denoised_output, reconstructions_list, latents, inputs_subsampled

