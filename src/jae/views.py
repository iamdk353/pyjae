"""View/split generator layer for JAE.

This module is the core modularization point of the project: it decouples HOW
input data is partitioned into "views" from the models that consume those views.
A view generator's only job is to hand back index sets or masks; it knows nothing
about encoders, losses, or training loops.

Two families of view generators live here, unified by a single idea: a "view" is
a deterministic, generator-seeded partition of some grid of positions.

- ChannelSplit: partitions the CHANNEL axis of (B, C, T) data into disjoint or
  overlapping index sets. Used by JAE1 (the two-network channel-split
  autoencoder), which trains one sub-network per channel partition.
- JEPAMask: partitions a (channel, time-patch) grid into a visible "context"
  region and one or more masked "target" regions. Used by JAE2 (the JEPA-style
  latent predictor), which predicts target-region embeddings from the context.

Both families expose a ``__call__`` that accepts a ``torch.Generator`` so callers
get fully reproducible views. Every source of randomness in this module is drawn
from that generator; no global ``torch.rand``/``torch.randint``/``torch.randperm``
call (i.e. one without a ``generator=`` argument) is made anywhere here.
"""

from abc import ABC, abstractmethod

import torch


def _floor_partition_size(n_channels: int, fraction: float) -> int:
    """Compute an integer partition size as floor(fraction * n_channels).

    Args:
        n_channels: Total number of channels available.
        fraction: Fraction of n_channels a partition should contain.

    Returns:
        The partition size, clamped to [0, n_channels].
    """
    size = int(fraction * n_channels)
    return max(0, min(size, n_channels))


def _sample_uniform_float(bounds: tuple[float, float], generator: torch.Generator) -> float:
    """Draw a single float from Uniform(bounds[0], bounds[1]) using generator.

    Args:
        bounds: (low, high) range to sample from.
        generator: Torch generator controlling the draw.

    Returns:
        A float sampled uniformly from the given range. If low >= high, low is
        returned directly (degenerate, zero-width range).
    """
    low, high = bounds
    if high <= low:
        return low
    return torch.empty(1).uniform_(low, high, generator=generator).item()


def _sample_block_start(extent: int, size: int, generator: torch.Generator) -> int:
    """Draw a random block start offset in [0, extent - size] using generator.

    Args:
        extent: Length of the axis the block is placed on.
        size: Length of the block along that axis (must be <= extent).
        generator: Torch generator controlling the draw.

    Returns:
        A valid start index such that [start, start + size) lies within [0, extent).
    """
    span = extent - size + 1
    if span <= 1:
        return 0
    return int(torch.randint(0, span, (1,), generator=generator).item())


class ChannelSplit(ABC):
    """Abstract base for channel-split view generators (used by JAE1).

    A ChannelSplit strategy partitions the channel indices ``[0, n_channels)`` of
    (B, C, T) data into one or more index sets that JAE1 feeds to separate
    per-partition sub-networks. Implementations differ in whether partitions are
    disjoint or overlapping and in how positions are chosen (random permutation,
    contiguous blocks, etc.), but all share this call signature so JAE1 can swap
    strategies without any change to model code. This shared signature, plus
    JEPAMask below, is what makes the split/view logic in this file "modular":
    both families are just view generators seeded by a torch.Generator.
    """

    @abstractmethod
    def __call__(self, n_channels: int, generator: torch.Generator) -> list[torch.LongTensor]:
        """Generate channel-index partitions.

        Args:
            n_channels: Total number of channels to partition.
            generator: Torch generator controlling all randomness.

        Returns:
            List of 1-D LongTensor channel-index sets, one per partition.
        """
        raise NotImplementedError


class RandomDisjointSplit(ChannelSplit):
    """Randomly partitions channels into disjoint index sets. JAE1 paper default.

    A single random permutation of ``[0, n_channels)`` is drawn and cut into
    ``n_partitions`` consecutive chunks of ``floor(fraction * n_channels)``
    channels each. Because the chunks are consecutive, non-overlapping slices of
    one permutation, the partitions are disjoint by construction.

    Behavior when ``fraction * n_partitions != 1``:
        - If ``fraction * n_partitions < 1``, some channels are left out of every
          partition (they simply are not assigned to a view). Each partition
          still has exactly ``floor(fraction * n_channels)`` channels.
        - If ``fraction * n_partitions > 1``, the requested partitions would need
          more channels than the permutation has left; the later partitions are
          then silently truncated (possibly empty) rather than raising an error,
          since the permutation only has ``n_channels`` unique indices to give
          out. Callers who need every partition to be full-size should keep
          ``fraction <= 1 / n_partitions``.

    Args:
        n_partitions: Number of disjoint partitions to produce.
        fraction: Fraction of n_channels each partition should contain.
    """

    def __init__(self, n_partitions: int = 2, fraction: float = 0.5) -> None:
        if n_partitions < 1:
            raise ValueError("n_partitions must be >= 1")
        if fraction <= 0:
            raise ValueError("fraction must be > 0")
        self.n_partitions = n_partitions
        self.fraction = fraction

    def __call__(self, n_channels: int, generator: torch.Generator) -> list[torch.LongTensor]:
        perm = torch.randperm(n_channels, generator=generator)
        part_size = _floor_partition_size(n_channels, self.fraction)

        partitions = []
        start = 0
        for _ in range(self.n_partitions):
            end = start + part_size
            partitions.append(perm[start:end].clone())
            start = end
        return partitions


class ContiguousSplit(ChannelSplit):
    """Partitions channels into contiguous, disjoint blocks (no randomness).

    Splits ``[0, n_channels)`` in index order into ``n_partitions`` contiguous
    blocks of as-equal-as-possible size: the first ``n_channels % n_partitions``
    blocks get one extra channel, matching ``numpy.array_split`` semantics. The
    result is deterministic; the ``generator`` argument is accepted only for
    interface consistency with ChannelSplit and is not used.

    Args:
        n_partitions: Number of contiguous, disjoint partitions to produce.
    """

    def __init__(self, n_partitions: int = 2) -> None:
        if n_partitions < 1:
            raise ValueError("n_partitions must be >= 1")
        self.n_partitions = n_partitions

    def __call__(self, n_channels: int, generator: torch.Generator) -> list[torch.LongTensor]:
        del generator  # unused: this strategy is fully deterministic
        indices = torch.arange(n_channels)
        base_size, remainder = divmod(n_channels, self.n_partitions)

        partitions = []
        start = 0
        for i in range(self.n_partitions):
            size = base_size + (1 if i < remainder else 0)
            end = start + size
            partitions.append(indices[start:end].clone())
            start = end
        return partitions


class OverlappingSplit(ChannelSplit):
    """Randomly samples channel partitions that are allowed to overlap.

    Each of ``n_partitions`` partitions is drawn independently: a fresh random
    permutation of ``[0, n_channels)`` is generated and its first
    ``floor(fraction * n_channels)`` indices become that partition. Because each
    partition is sampled independently, channels can (and, for
    ``fraction > 1 / n_partitions``, typically will) appear in more than one
    partition; overlap is permitted by design, not guaranteed for every input.

    Args:
        n_partitions: Number of partitions to produce.
        fraction: Fraction of n_channels each partition should contain.
    """

    def __init__(self, n_partitions: int = 2, fraction: float = 0.6) -> None:
        if n_partitions < 1:
            raise ValueError("n_partitions must be >= 1")
        if fraction <= 0:
            raise ValueError("fraction must be > 0")
        self.n_partitions = n_partitions
        self.fraction = fraction

    def __call__(self, n_channels: int, generator: torch.Generator) -> list[torch.LongTensor]:
        part_size = _floor_partition_size(n_channels, self.fraction)

        partitions = []
        for _ in range(self.n_partitions):
            perm = torch.randperm(n_channels, generator=generator)
            partitions.append(perm[:part_size].clone())
        return partitions


class JEPAMask(ABC):
    """Abstract base for JEPA context/target masking strategies (used by JAE2).

    A JEPAMask strategy partitions a (channel, time-patch) grid into a visible
    "context" region and one or more masked "target" regions. JAE2 encodes the
    context, predicts each target region's embedding, and compares that
    prediction against the (stop-gradient) embedding of the actual target
    tokens. Like ChannelSplit, this is a pure view generator: it returns boolean
    masks over grid positions and has no knowledge of encoders or losses. This
    shared "view generator" shape (config in __init__, randomness only via the
    passed generator in __call__) is the modularization this file provides:
    JAE1's channel split and JAE2's JEPA masking are both views, just over
    different grids (channels-only vs. channels x time-patches).
    """

    @abstractmethod
    def __call__(
        self, n_channels: int, n_patches: int, generator: torch.Generator
    ) -> tuple[torch.BoolTensor, list[torch.BoolTensor]]:
        """Generate a context mask and target masks over a (channel, patch) grid.

        Args:
            n_channels: Number of channels (grid rows).
            n_patches: Number of time patches (grid columns).
            generator: Torch generator controlling all randomness.

        Returns:
            Tuple of:
                context_mask: BoolTensor(n_channels, n_patches), True at visible
                    (context) positions.
                targets: List of BoolTensor(n_channels, n_patches), each True
                    only within one target block. Target masks never overlap
                    context_mask.
        """
        raise NotImplementedError


class SpatioTemporalBlockMask(JEPAMask):
    """JEPA-style spatiotemporal block masking for JAE2.

    Samples rectangular blocks over the (channel, time-patch) grid. The union of
    the sampled blocks becomes the TARGET region (masked, to be predicted); its
    complement becomes the CONTEXT region (visible input). This generalizes
    I-JEPA/STST-JEPA multi-block masking to mask both the channel and time axes
    at once, as in Laya/STST-JEPA for EEG-like signals.

    Each block's channel extent is drawn as ``Uniform(ch_frac)`` of n_channels
    and its time-patch extent as ``Uniform(t_frac)`` of n_patches; its position
    is drawn uniformly at random subject to fitting inside the grid. Blocks are
    sampled one at a time, up to ``n_blocks`` of them, but sampling stops early
    once the union of already-sampled blocks reaches approximately ``coverage``
    of the grid, so the target masks returned can number fewer than n_blocks.
    This keeps ``coverage`` meaningful across grid sizes and guards against a
    handful of large blocks swallowing an entire small grid.

    Degenerate cases (e.g. tiny grids) are guarded explicitly: at least one
    block is always sampled (so targets is never empty), and if a candidate
    block would leave zero visible context, it is either skipped (while other
    blocks already give a non-empty target) or, if the grid is otherwise fully
    covered, a single position is freed back into the context.

    Args:
        n_blocks: Maximum number of target blocks to sample.
        ch_frac: (low, high) range for the fraction of n_channels each block spans.
        t_frac: (low, high) range for the fraction of n_patches each block spans.
        coverage: Approximate target fraction of the grid covered by the union
            of blocks; sampling stops early once this is reached.
    """

    def __init__(
        self,
        n_blocks: int = 4,
        ch_frac: tuple[float, float] = (0.2, 0.5),
        t_frac: tuple[float, float] = (0.1, 0.4),
        coverage: float = 0.5,
    ) -> None:
        if n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")
        if not (0.0 < coverage <= 1.0):
            raise ValueError("coverage must be in (0, 1]")
        self.n_blocks = n_blocks
        self.ch_frac = ch_frac
        self.t_frac = t_frac
        self.coverage = coverage

    def __call__(
        self, n_channels: int, n_patches: int, generator: torch.Generator
    ) -> tuple[torch.BoolTensor, list[torch.BoolTensor]]:
        union = torch.zeros((n_channels, n_patches), dtype=torch.bool)
        target_area_cap = self.coverage * n_channels * n_patches
        blocks: list[torch.BoolTensor] = []

        for _ in range(self.n_blocks):
            if blocks and union.sum().item() >= target_area_cap:
                break

            block = self._sample_block(n_channels, n_patches, generator)
            candidate_union = union | block

            if blocks and bool(candidate_union.all()):
                # This block would swallow the whole grid; skip it so context
                # stays non-empty. We already have at least one target block.
                continue

            union = candidate_union
            blocks.append(block)

        if not blocks:
            # Only reachable for pathological (near-zero-size) grids.
            block = torch.zeros((n_channels, n_patches), dtype=torch.bool)
            block[0, 0] = True
            union = block.clone()
            blocks.append(block)

        if bool(union.all()):
            # Whole grid ended up masked (only possible on tiny grids where even
            # the first block is unavoidable): free one position back to context.
            union[0, 0] = False
            blocks[0][0, 0] = False

        context_mask = ~union
        return context_mask, blocks

    def _sample_block(
        self, n_channels: int, n_patches: int, generator: torch.Generator
    ) -> torch.BoolTensor:
        """Sample one rectangular block mask over the (channel, patch) grid.

        Args:
            n_channels: Number of channels (grid rows).
            n_patches: Number of time patches (grid columns).
            generator: Torch generator controlling all randomness.

        Returns:
            BoolTensor(n_channels, n_patches), True only within the block.
        """
        ch_size = self._block_extent(self.ch_frac, n_channels, generator)
        t_size = self._block_extent(self.t_frac, n_patches, generator)
        ch_start = _sample_block_start(n_channels, ch_size, generator)
        t_start = _sample_block_start(n_patches, t_size, generator)

        block = torch.zeros((n_channels, n_patches), dtype=torch.bool)
        block[ch_start : ch_start + ch_size, t_start : t_start + t_size] = True
        return block

    @staticmethod
    def _block_extent(
        frac_range: tuple[float, float], extent: int, generator: torch.Generator
    ) -> int:
        """Sample a block length along one axis, clamped to [1, extent].

        Args:
            frac_range: (low, high) fraction-of-extent range to sample from.
            extent: Length of the axis (n_channels or n_patches).
            generator: Torch generator controlling the draw.

        Returns:
            Integer block length, at least 1 and at most extent.
        """
        frac = _sample_uniform_float(frac_range, generator)
        size = int(round(frac * extent))
        return max(1, min(extent, size))
