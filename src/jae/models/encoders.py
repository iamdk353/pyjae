"""Shared encoder/decoder building blocks used by the JAE1 and JAE2 models.

This module is self-contained: it does not import from other ``jae`` modules and is not
wired into ``jae.models.__init__``. Integration happens elsewhere.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

_VALID_LATENT_ACTIVATIONS = ("relu", "none", "tanh")
_VALID_OUT_ACTIVATIONS = ("relu", "none")


def _build_funnel(dims: Sequence[int]) -> tuple[nn.Sequential, nn.Linear]:
    """Build a funnel (or expanding) MLP from a sequence of layer widths.

    Consecutive widths are connected by ``nn.Linear`` layers, with ``nn.ReLU``
    inserted between every pair of linear layers (but not after the last one, so
    callers can apply their own final activation).

    Args:
        dims: Layer widths, e.g. ``(in_dim, hidden0, hidden1, out_dim)``. Must have
            length >= 2.

    Returns:
        A tuple of ``(sequential_module, final_linear_layer)``. Exposing the final
        linear layer lets callers re-initialize its bias directly.
    """
    if len(dims) < 2:
        raise ValueError("dims must contain at least an input and output width.")

    layers: list[nn.Module] = []
    final_linear: nn.Linear | None = None
    n_linears = len(dims) - 1
    for i in range(n_linears):
        linear = nn.Linear(dims[i], dims[i + 1])
        layers.append(linear)
        final_linear = linear
        if i < n_linears - 1:
            layers.append(nn.ReLU(inplace=True))
    assert final_linear is not None
    return nn.Sequential(*layers), final_linear


def _pick_num_heads(d_model: int, preferred: int = 4) -> int:
    """Pick a number of attention heads that evenly divides ``d_model``.

    Args:
        d_model: Embedding dimension the heads must evenly divide.
        preferred: Desired head count, used when it divides ``d_model`` evenly.

    Returns:
        ``preferred`` if it divides ``d_model``, otherwise the largest divisor of
        ``d_model`` that is <= ``preferred`` (falling back to 1 if none found, which
        always divides evenly).
    """
    if d_model % preferred == 0:
        return preferred
    for n_heads in range(preferred - 1, 0, -1):
        if d_model % n_heads == 0:
            return n_heads
    return 1


class MLPEncoder(nn.Module):
    """Funnel MLP encoder: input dropout followed by ``in_dim -> ... -> latent_dim``.

    ReLU is applied between all hidden linear layers. The final latent activation is
    configurable via ``latent_activation``.

    Args:
        in_dim: Input feature dimension.
        latent_dim: Output latent dimension.
        hidden: Widths of the hidden layers between ``in_dim`` and ``latent_dim``.
        dropout_p: Probability for the input dropout applied before the first layer.
        latent_activation: Activation applied to the final latent. One of ``'relu'``
            (paper-faithful, non-negative latent), ``'none'`` (linear, can be
            negative), or ``'tanh'``.
    """

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        hidden: Sequence[int] = (128, 64),
        dropout_p: float = 0.05,
        latent_activation: str = "relu",
    ) -> None:
        super().__init__()
        if latent_activation not in _VALID_LATENT_ACTIVATIONS:
            raise ValueError(
                f"latent_activation must be one of {_VALID_LATENT_ACTIVATIONS}, "
                f"got {latent_activation!r}."
            )

        self.latent_activation = latent_activation
        self.dropout = nn.Dropout(dropout_p)
        dims = [in_dim, *hidden, latent_dim]
        self.net, _ = _build_funnel(dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode inputs into the latent space.

        Args:
            x: Input tensor of shape ``(N, in_dim)``.

        Returns:
            Latent tensor ``z`` of shape ``(N, latent_dim)``.
        """
        x = self.dropout(x)
        z = self.net(x)
        if self.latent_activation == "relu":
            z = F.relu(z)
        elif self.latent_activation == "tanh":
            z = torch.tanh(z)
        return z


class MLPDecoder(nn.Module):
    """Funnel MLP decoder mirroring MLPEncoder: ``latent_dim -> ... -> out_dim``.

    ReLU is applied between all hidden linear layers. The final layer's bias is
    initialized to ``pos_bias`` (a small positive value) so the output activation
    does not start in the dead-ReLU regime.

    Args:
        latent_dim: Input latent dimension.
        out_dim: Output feature dimension.
        hidden: Widths of the hidden layers between ``latent_dim`` and ``out_dim``.
        out_activation: Activation applied to the final output. One of ``'relu'``
            (non-negative signals, e.g. firing rates) or ``'none'`` (linear).
        pos_bias: Value used to initialize the final layer's bias.
    """

    def __init__(
        self,
        latent_dim: int,
        out_dim: int,
        hidden: Sequence[int] = (64, 128),
        out_activation: str = "relu",
        pos_bias: float = 0.1,
    ) -> None:
        super().__init__()
        if out_activation not in _VALID_OUT_ACTIVATIONS:
            raise ValueError(
                f"out_activation must be one of {_VALID_OUT_ACTIVATIONS}, got {out_activation!r}."
            )

        self.out_activation = out_activation
        dims = [latent_dim, *hidden, out_dim]
        self.net, final_linear = _build_funnel(dims)
        nn.init.constant_(final_linear.bias, pos_bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents back into signal space.

        Args:
            z: Latent tensor of shape ``(N, latent_dim)``.

        Returns:
            Reconstruction ``x_hat`` of shape ``(N, out_dim)``.
        """
        x_hat = self.net(z)
        if self.out_activation == "relu":
            x_hat = F.relu(x_hat)
        return x_hat


class ConvPatchEncoder(nn.Module):
    """Depthwise 1D-conv patch embedder with a channel-mixing attention layer.

    Each channel is independently patchified along the time axis by a grouped
    (depthwise) 1D convolution with ``kernel_size = stride = patch_len``, producing a
    ``d_model``-dim embedding per channel per patch. A single multi-head
    self-attention layer then mixes information across channels, independently at
    each patch position (channels attend to channels, not to other patches), followed
    by a residual connection and LayerNorm. This is intentionally lean: one
    patch-embed step plus one mixing layer, no deep transformer stack.

    Length handling: if the input's time dimension ``T`` does not exactly equal
    ``n_patches * patch_len``, it is truncated (drop trailing samples) when longer,
    or zero-padded on the right when shorter, before patchifying.

    Args:
        n_channels: Number of input channels ``C``.
        n_patches: Expected number of patches ``P`` along the time axis.
        patch_len: Number of timepoints per patch. Also the conv kernel size/stride.
        d_model: Per-channel, per-patch embedding dimension.
    """

    def __init__(
        self,
        n_channels: int,
        n_patches: int,
        patch_len: int,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_patches = n_patches
        self.patch_len = patch_len
        self.d_model = d_model
        self.expected_len = n_patches * patch_len

        # Depthwise/grouped conv: one group per channel, each group maps its own
        # patch_len samples to a d_model-dim embedding independently of other
        # channels. Cross-channel mixing happens afterward in self.channel_mix.
        self.patch_embed = nn.Conv1d(
            in_channels=n_channels,
            out_channels=n_channels * d_model,
            kernel_size=patch_len,
            stride=patch_len,
            groups=n_channels,
        )

        n_heads = _pick_num_heads(d_model)
        self.channel_mix = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Patch-embed and channel-mix an input batch.

        Args:
            x: Input tensor of shape ``(B, C, T)``, ``C == n_channels``. If
                ``T != n_patches * patch_len`` it is truncated or zero-padded (see
                class docstring) before patchifying.

        Returns:
            A tuple ``(tokens, pooled)``:
                tokens: Per-channel, per-patch embeddings, shape
                    ``(B, C, n_patches, d_model)``.
                pooled: Per-patch summary obtained by averaging ``tokens`` over the
                    channel dimension, shape ``(B, n_patches, d_model)``.
        """
        batch_size, n_channels, t = x.shape
        if t > self.expected_len:
            x = x[..., : self.expected_len]
        elif t < self.expected_len:
            x = F.pad(x, (0, self.expected_len - t))

        # (B, C, T) -> (B, C * d_model, n_patches)
        conv_out = self.patch_embed(x)
        # Grouped conv output channels are laid out group-major, so this view
        # recovers the (channel, d_model) split exactly.
        tokens = conv_out.view(batch_size, n_channels, self.d_model, self.n_patches)
        tokens = tokens.permute(0, 1, 3, 2)  # (B, C, n_patches, d_model)

        # Mix across channels independently per patch: fold (B, n_patches) into the
        # attention batch dimension so channels only attend to other channels.
        attn_in = tokens.permute(0, 2, 1, 3).reshape(
            batch_size * self.n_patches, n_channels, self.d_model
        )
        mixed, _ = self.channel_mix(attn_in, attn_in, attn_in)
        mixed = self.norm(attn_in + mixed)
        mixed = mixed.reshape(batch_size, self.n_patches, n_channels, self.d_model)
        tokens = mixed.permute(0, 2, 1, 3)  # (B, C, n_patches, d_model)

        pooled = tokens.mean(dim=1)  # (B, n_patches, d_model)
        return tokens, pooled
