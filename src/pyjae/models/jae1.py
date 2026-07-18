"""Joint Autoencoder (JAE1): a modular, collapse-resistant channel-split autoencoder.

This reimplements the JAE1 architecture from Altan et al. (2021). The original
repo implementation collapsed to a degenerate latent because its per-partition
"autoencoder" was a single Linear -> ReLU encoder: one matrix multiply has no
capacity to represent a nonlinear compressive code, so training pushed it toward
a trivial, near-constant solution. This module fixes that by giving each
partition a real funnel MLP (:class:`~pyjae.models.encoders.MLPEncoder` /
:class:`~pyjae.models.encoders.MLPDecoder`), matching the paper's architecture.

Why the channel split is the denoising mechanism
--------------------------------------------------
JAE1 randomly partitions the recorded channels into (by default two) disjoint
subsets. Both subsets are noisy observations of the *same* underlying low-
dimensional signal (e.g. a shared population-level latent driving all
channels), but the noise corrupting each channel is independent across
channels and therefore independent across the two partitions. This is exactly
the setup exploited by Noise2Noise-style training: if two views of a signal
share the clean content but not the noise, then forcing a model trained on one
view to agree with a model trained on the other view can only be achieved by
keeping the shared, denoised content and discarding the independent noise
(matching independent noise realizations is not a low-cost strategy, so the
optimizer is not rewarded for reproducing it). Concretely, each partition gets
its own encoder/decoder pair, and the training objective adds an alignment
term that pulls the two partitions' latents together. A single autoencoder
trained on all channels at once has no such cross-view supervision signal and
nothing that penalizes it for encoding channel-specific noise into its latent;
the two-network channel split is what turns "reconstruct the input" into
"reconstruct only what's common between two independent noisy views," which is
the denoising mechanism the paper relies on (Altan et al. 2021).

Because both partitions must agree, the shared channel split is sampled once
per model (seeded, so it is reproducible) and reused for every forward pass:
training and later denoising must route the same channels through the same
sub-network for the alignment term to mean anything.
"""

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from pyjae.models.encoders import MLPDecoder, MLPEncoder
from pyjae.views import ChannelSplit, RandomDisjointSplit


class JAE1Output(NamedTuple):
    """Structured result of a :class:`JAE1` forward pass.

    Attributes:
        denoised: Full reconstruction, shape ``(B, C, T)``, assembled by
            scattering each partition's decoded output back to its original
            channel positions.
        latents: Per-partition latents, each shape ``(B*T, latent_dim)``.
        recons: Per-partition reconstructions, each shape ``(B*T, C_i)`` where
            ``C_i`` is that partition's channel count.
        targets: Per-partition targets (the input restricted to that
            partition's channels), each shape ``(B*T, C_i)``.
        indices: Per-partition channel-index LongTensors used to select/scatter
            that partition's channels.
    """

    denoised: torch.Tensor
    latents: list[torch.Tensor]
    recons: list[torch.Tensor]
    targets: list[torch.Tensor]
    indices: list[torch.Tensor]


class JAE1(nn.Module):
    """Modular, collapse-resistant Joint Autoencoder (Altan et al. 2021, Fig 2).

    Channels are partitioned once (at construction time, seeded) into one or
    more index sets via a :class:`~pyjae.views.ChannelSplit` strategy. Each
    partition gets its own funnel-MLP encoder/decoder pair
    (:class:`~pyjae.models.encoders.MLPEncoder` /
    :class:`~pyjae.models.encoders.MLPDecoder`), all sharing the same latent
    dimensionality. See the module docstring for why this channel split is
    what makes denoising possible.

    Args:
        input_dim: Total number of channels ``C`` the model expects.
        latent_dim: Shared latent bottleneck dimension ``D`` for every
            partition's encoder/decoder.
        split: A :class:`~pyjae.views.ChannelSplit` strategy used to partition
            channel indices. Defaults to
            ``RandomDisjointSplit(n_partitions=2, fraction=0.5)``, the paper's
            50/50 random disjoint split.
        hidden: Hidden-layer widths for the encoder funnel (``in_dim -> ... ->
            latent_dim``); the decoder uses the reverse of this sequence
            (``latent_dim -> ... -> out_dim``).
        dropout_p: Input dropout probability, applied inside each encoder.
        latent_activation: Activation applied to each partition's latent. Defaults
            to ``"none"`` (a linear bottleneck, like PCA). The paper used ReLU
            everywhere for a non-negative firing-rate interpretation, but a ReLU
            latent has a zero-collapse attractor on zero-mean data (the latent
            and its alignment term are both minimized by driving every code to
            zero), so ``"relu"`` is offered only for genuinely non-negative data.
        decoder_activation: Output activation of each partition's decoder.
            Defaults to ``"none"`` (linear output), which is required whenever
            the data can be negative (e.g. mean-centered or standardized signals):
            a ReLU output clips negatives to zero and cannot reconstruct them.
            Use ``"relu"`` only for strictly non-negative signals.
        seed: Seed for the ``torch.Generator`` used to sample the channel
            split. The resulting partition is fixed for the lifetime of the
            model (stored as buffers) so training and later denoising always
            route the same channels through the same sub-network, and so two
            models built with the same ``seed`` get an identical split.

    Raises:
        ValueError: If the split produces zero partitions, or if any
            partition is empty.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        split: ChannelSplit | None = None,
        hidden: tuple[int, ...] = (128, 64),
        dropout_p: float = 0.05,
        latent_activation: str = "none",
        decoder_activation: str = "none",
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden = tuple(hidden)
        self.dropout_p = dropout_p
        self.latent_activation = latent_activation
        self.decoder_activation = decoder_activation
        self.seed = seed

        split = split if split is not None else RandomDisjointSplit(n_partitions=2, fraction=0.5)

        # Sample the partition ONCE, from a generator seeded with `seed`, so the
        # split is fully reproducible and identical across every forward pass
        # (including later denoising calls) and across re-instantiated models
        # that share the same seed.
        generator = torch.Generator().manual_seed(seed)
        partitions = split(input_dim, generator)
        if len(partitions) == 0:
            raise ValueError("split produced zero partitions.")
        for i, idx in enumerate(partitions):
            if idx.numel() == 0:
                raise ValueError(
                    f"partition {i} is empty; choose a larger fraction or fewer partitions."
                )

        self.n_partitions = len(partitions)
        self._partition_names = [f"indices_{i}" for i in range(self.n_partitions)]
        for name, idx in zip(self._partition_names, partitions):
            # Registered as buffers (not plain attributes) so the split moves
            # with the model under `.to(device)` / `.cuda()` and is saved and
            # restored by `state_dict()`, keeping it fixed across the model's
            # whole lifetime, including denoising after training.
            self.register_buffer(name, idx.long())

        decoder_hidden = tuple(reversed(self.hidden))
        self.encoders = nn.ModuleList(
            [
                MLPEncoder(
                    in_dim=idx.numel(),
                    latent_dim=latent_dim,
                    hidden=self.hidden,
                    dropout_p=dropout_p,
                    latent_activation=latent_activation,
                )
                for idx in partitions
            ]
        )
        self.decoders = nn.ModuleList(
            [
                MLPDecoder(
                    latent_dim=latent_dim,
                    out_dim=idx.numel(),
                    hidden=decoder_hidden,
                    out_activation=decoder_activation,
                )
                for idx in partitions
            ]
        )

    def _indices(self, i: int) -> torch.Tensor:
        """Fetch partition ``i``'s channel-index buffer by attribute lookup.

        Args:
            i: Partition index.

        Returns:
            LongTensor of channel indices for partition ``i``.
        """
        return getattr(self, self._partition_names[i])

    @property
    def partition_sizes(self) -> list[int]:
        """Number of channels routed to each partition, in partition order."""
        return [self._indices(i).numel() for i in range(self.n_partitions)]

    def forward(self, x: torch.Tensor) -> JAE1Output:
        """Run every partition's encoder/decoder and assemble a full denoised signal.

        Args:
            x: Input tensor of shape ``(B, C, T)`` with ``C == input_dim``.

        Returns:
            A :class:`JAE1Output` with the full ``(B, C, T)`` reconstruction
            plus per-partition latents, reconstructions, targets, and indices.

        Raises:
            ValueError: If ``x`` does not have ``C == input_dim`` channels.
        """
        batch_size, n_channels, n_time = x.shape
        if n_channels != self.input_dim:
            raise ValueError(f"expected {self.input_dim} channels, got {n_channels}.")

        denoised = torch.zeros_like(x)
        latents: list[torch.Tensor] = []
        recons: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []

        for i in range(self.n_partitions):
            idx = self._indices(i)
            n_channels_i = idx.numel()

            # (B, C_i, T) -> (B*T, C_i): each timepoint of each recording
            # becomes one row so the shared MLP sees per-timepoint channel
            # vectors, matching the paper's fully-connected treatment.
            x_i = x[:, idx, :]
            x_i_flat = x_i.permute(0, 2, 1).reshape(batch_size * n_time, n_channels_i)

            z_i = self.encoders[i](x_i_flat)
            x_hat_i_flat = self.decoders[i](z_i)

            # (B*T, C_i) -> (B, C_i, T), then scatter into the full tensor at
            # this partition's original channel positions.
            x_hat_i = x_hat_i_flat.reshape(batch_size, n_time, n_channels_i).permute(0, 2, 1)
            denoised = denoised.index_copy(1, idx, x_hat_i)

            latents.append(z_i)
            recons.append(x_hat_i_flat)
            targets.append(x_i_flat)
            indices.append(idx)

        return JAE1Output(
            denoised=denoised, latents=latents, recons=recons, targets=targets, indices=indices
        )

    def loss(self, out: JAE1Output, latent_weight: float = 1.0) -> torch.Tensor:
        """Compute the unweighted-by-default JAE1 objective (paper Eq. 3).

        ``C = sum_i MSE(recons[i], targets[i]) + latent_weight * mean_{i<j} MSE(latents[i], latents[j])``

        For the paper's default of two partitions this reduces exactly to
        ``MSE(X1, X1_hat) + MSE(X2, X2_hat) + latent_weight * MSE(Z1, Z2)``,
        which is unweighted (matches the paper) when ``latent_weight=1.0``.

        Args:
            out: A :class:`JAE1Output` produced by :meth:`forward`.
            latent_weight: Weight applied to the latent-alignment term.

        Returns:
            Scalar loss tensor.
        """
        recon = sum(
            F.mse_loss(recon_i, target_i) for recon_i, target_i in zip(out.recons, out.targets)
        )

        if self.n_partitions < 2:
            align = torch.zeros((), device=out.denoised.device, dtype=out.denoised.dtype)
        else:
            pair_losses = [
                F.mse_loss(out.latents[i], out.latents[j])
                for i in range(self.n_partitions)
                for j in range(i + 1, self.n_partitions)
            ]
            align = torch.stack(pair_losses).mean()

        return recon + latent_weight * align
