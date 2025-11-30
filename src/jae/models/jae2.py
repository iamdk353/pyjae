"""
Enhanced Joint Autoencoder with U-Net and multi-view support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    """1D convolutional block: (Conv -> BatchNorm -> LeakyReLU) x 2."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet1D(nn.Module):
    """
    1D U-Net for temporal signal denoising.

    Uses encoder-decoder with skip connections to preserve temporal details.
    """

    def __init__(self, input_channels, channels=None, latent_dim=64):
        super().__init__()
        if channels is None:
            channels = [64, 128]

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

        # Latent projector
        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_channels, bottleneck_channels),
            nn.BatchNorm1d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_channels, latent_dim),
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
        # Encoder
        skip_connections = []
        for block in self.enc_blocks:
            x = block(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        latent = self.projector(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for i in range(len(self.dec_blocks)):
            x = self.up_convs[i](x)
            if x.shape[2] != skip_connections[i].shape[2]:
                x = F.interpolate(x, size=skip_connections[i].shape[2], mode="linear", align_corners=False)
            x = torch.cat((x, skip_connections[i]), dim=1)
            x = self.dec_blocks[i](x)

        reconstruction = F.relu(self.final_conv(x))
        return reconstruction, latent


class JAE2(nn.Module):
    """
    Enhanced Joint Autoencoder with N parallel networks.

    Supports U-Net architecture and stochastic channel subsampling.

    Parameters
    ----------
    input_dim : int
        Number of input channels.
    latent_dim : int, default=64
        Latent space dimensionality.
    num_networks : int, default=5
        Number of parallel networks.
    subsample_fraction : float, default=0.8
        Fraction of channels per network.
    unet_channels : list, optional
        U-Net encoder channel dimensions.
    """

    def __init__(
        self,
        input_dim,
        latent_dim=64,
        num_networks=5,
        subsample_fraction=0.8,
        unet_channels=None,
    ):
        super().__init__()
        if unet_channels is None:
            unet_channels = [32, 64]

        self.input_dim = input_dim
        self.subsample_dim = int(input_dim * subsample_fraction)
        if self.subsample_dim == 0:
            raise ValueError("Subsample fraction too low or input dim too small.")

        self.networks = nn.ModuleList([
            UNet1D(self.subsample_dim, channels=unet_channels, latent_dim=latent_dim)
            for _ in range(num_networks)
        ])

    def forward(self, x):
        B, C, T = x.shape
        latents = []
        reconstructions = []
        inputs_subsampled = []

        full_recon = torch.zeros(B, C, T, device=x.device)
        counts = torch.zeros(1, C, 1, device=x.device)

        for network in self.networks:
            # Random channel subsampling
            indices = torch.randperm(self.input_dim, device=x.device)[: self.subsample_dim]
            x_sub = x[:, indices, :]
            inputs_subsampled.append(x_sub)

            # Process
            recon, latent = network(x_sub)
            reconstructions.append(recon)
            latents.append(latent)

            # Accumulate reconstruction
            indices_expanded = indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, T)
            full_recon.scatter_add_(1, indices_expanded, recon)
            counts[0, indices, 0] += 1

        # Average reconstructions
        counts[counts == 0] = 1
        denoised = full_recon / counts

        return denoised, reconstructions, latents, inputs_subsampled
