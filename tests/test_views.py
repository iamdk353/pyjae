"""Unit tests for the view/split generator layer (src/jae/views.py)."""

import pytest
import torch

from jae.views import (
    ContiguousSplit,
    OverlappingSplit,
    RandomDisjointSplit,
    SpatioTemporalBlockMask,
)


def _gen(seed: int) -> torch.Generator:
    """Build a seeded CPU generator for reproducible tests."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class TestRandomDisjointSplit:
    """Tests for RandomDisjointSplit (JAE1 paper default)."""

    def test_default_shapes_and_sizes(self):
        """2 partitions of 48 on 96 channels with default args."""
        split = RandomDisjointSplit()
        partitions = split(96, _gen(0))

        assert len(partitions) == 2
        for part in partitions:
            assert part.dtype == torch.long
            assert part.numel() == 48

    def test_disjoint_and_covers_all_channels(self):
        """Partitions share no indices and their union covers every channel."""
        split = RandomDisjointSplit(n_partitions=2, fraction=0.5)
        part_a, part_b = split(96, _gen(1))

        set_a = set(part_a.tolist())
        set_b = set(part_b.tolist())

        assert set_a.isdisjoint(set_b)
        assert set_a | set_b == set(range(96))

    def test_reproducible_given_same_seed(self):
        """Same generator seed produces identical partitions."""
        split = RandomDisjointSplit()
        parts_1 = split(96, _gen(42))
        parts_2 = split(96, _gen(42))

        for p1, p2 in zip(parts_1, parts_2):
            assert torch.equal(p1, p2)

    def test_differs_across_seeds(self):
        """Different seeds produce different partitions (overwhelmingly likely)."""
        split = RandomDisjointSplit()
        parts_1 = split(96, _gen(1))
        parts_2 = split(96, _gen(2))

        assert not torch.equal(parts_1[0], parts_2[0])

    def test_small_fraction_leaves_channels_unassigned(self):
        """fraction * n_partitions < 1 => partitions smaller than an even split."""
        split = RandomDisjointSplit(n_partitions=2, fraction=0.3)
        part_a, part_b = split(96, _gen(0))

        assert part_a.numel() == 28  # floor(0.3 * 96)
        assert part_b.numel() == 28
        set_a, set_b = set(part_a.tolist()), set(part_b.tolist())
        assert set_a.isdisjoint(set_b)
        assert len(set_a | set_b) == 56 < 96

    def test_three_partitions_disjoint(self):
        """Generalizes cleanly to more than 2 partitions."""
        split = RandomDisjointSplit(n_partitions=3, fraction=1.0 / 3.0)
        parts = split(96, _gen(7))

        assert len(parts) == 3
        all_indices = torch.cat(parts).tolist()
        assert len(all_indices) == len(set(all_indices))


class TestContiguousSplit:
    """Tests for ContiguousSplit."""

    def test_blocks_contiguous_and_disjoint(self):
        """Each block is a contiguous run of indices; blocks are disjoint."""
        split = ContiguousSplit(n_partitions=2)
        part_a, part_b = split(96, _gen(0))

        assert torch.equal(part_a, torch.arange(0, 48))
        assert torch.equal(part_b, torch.arange(48, 96))

    def test_covers_all_channels(self):
        """Union of contiguous blocks covers every channel exactly once."""
        split = ContiguousSplit(n_partitions=4)
        parts = split(96, _gen(0))

        all_indices = torch.cat(parts).tolist()
        assert sorted(all_indices) == list(range(96))

    def test_uneven_split_distributes_remainder(self):
        """96 channels into 5 partitions: sizes differ by at most 1."""
        split = ContiguousSplit(n_partitions=5)
        parts = split(96, _gen(0))

        sizes = [p.numel() for p in parts]
        assert sum(sizes) == 96
        assert max(sizes) - min(sizes) <= 1

    def test_deterministic_regardless_of_generator_state(self):
        """No randomness: differing generators still give the same result."""
        split = ContiguousSplit(n_partitions=3)
        parts_1 = split(96, _gen(1))
        parts_2 = split(96, _gen(999))

        for p1, p2 in zip(parts_1, parts_2):
            assert torch.equal(p1, p2)


class TestOverlappingSplit:
    """Tests for OverlappingSplit."""

    def test_partition_sizes(self):
        """Each partition has floor(fraction * n_channels) channels."""
        split = OverlappingSplit(n_partitions=2, fraction=0.6)
        part_a, part_b = split(96, _gen(0))

        assert part_a.numel() == 57  # floor(0.6 * 96)
        assert part_b.numel() == 57

    def test_overlap_is_allowed(self):
        """With fraction > 1/n_partitions, overlap is expected across seeds."""
        split = OverlappingSplit(n_partitions=2, fraction=0.6)
        overlap_seen = False
        for seed in range(10):
            part_a, part_b = split(96, _gen(seed))
            if set(part_a.tolist()) & set(part_b.tolist()):
                overlap_seen = True
                break
        assert overlap_seen

    def test_reproducible_given_same_seed(self):
        """Same seed gives identical partitions."""
        split = OverlappingSplit()
        parts_1 = split(96, _gen(3))
        parts_2 = split(96, _gen(3))

        for p1, p2 in zip(parts_1, parts_2):
            assert torch.equal(p1, p2)

    def test_indices_within_range(self):
        """All sampled indices are valid channel indices."""
        split = OverlappingSplit(n_partitions=3, fraction=0.6)
        parts = split(96, _gen(0))

        for part in parts:
            assert part.min().item() >= 0
            assert part.max().item() < 96


class TestSpatioTemporalBlockMask:
    """Tests for SpatioTemporalBlockMask (JAE2 JEPA masking)."""

    def test_context_and_targets_disjoint(self):
        """No grid position is True in both the context mask and any target."""
        mask_gen = SpatioTemporalBlockMask()
        context_mask, targets = mask_gen(96, 16, _gen(0))

        assert context_mask.dtype == torch.bool
        for target in targets:
            assert target.dtype == torch.bool
            assert not bool((context_mask & target).any())

    def test_context_and_targets_nonempty(self):
        """Both context and every target block have at least one True position."""
        mask_gen = SpatioTemporalBlockMask()
        context_mask, targets = mask_gen(96, 16, _gen(0))

        assert bool(context_mask.any())
        assert len(targets) > 0
        for target in targets:
            assert bool(target.any())

    def test_context_is_exact_complement_of_target_union(self):
        """context_mask is exactly NOT (union of all target blocks)."""
        mask_gen = SpatioTemporalBlockMask()
        context_mask, targets = mask_gen(96, 16, _gen(0))

        union = torch.zeros_like(context_mask)
        for target in targets:
            union |= target
        assert torch.equal(context_mask, ~union)

    def test_coverage_in_reasonable_range(self):
        """Union of target blocks covers a plausible fraction of the grid."""
        mask_gen = SpatioTemporalBlockMask()
        n_channels, n_patches = 96, 16
        total = n_channels * n_patches

        for seed in range(5):
            context_mask, _ = mask_gen(n_channels, n_patches, _gen(seed))
            target_frac = 1.0 - (context_mask.sum().item() / total)
            assert 0.2 <= target_frac <= 0.8, f"seed={seed} target_frac={target_frac}"

    def test_reproducible_given_same_seed(self):
        """Same seed produces identical context and target masks."""
        mask_gen = SpatioTemporalBlockMask()
        ctx_1, targets_1 = mask_gen(96, 16, _gen(5))
        ctx_2, targets_2 = mask_gen(96, 16, _gen(5))

        assert torch.equal(ctx_1, ctx_2)
        assert len(targets_1) == len(targets_2)
        for t1, t2 in zip(targets_1, targets_2):
            assert torch.equal(t1, t2)

    def test_differs_across_seeds(self):
        """Different seeds produce different masks (overwhelmingly likely)."""
        mask_gen = SpatioTemporalBlockMask()
        ctx_1, _ = mask_gen(96, 16, _gen(1))
        ctx_2, _ = mask_gen(96, 16, _gen(2))

        assert not torch.equal(ctx_1, ctx_2)

    def test_small_grid_does_not_crash(self):
        """Small grids (8 channels x 4 patches) produce valid, non-empty masks."""
        mask_gen = SpatioTemporalBlockMask()
        context_mask, targets = mask_gen(8, 4, _gen(0))

        assert context_mask.shape == (8, 4)
        assert bool(context_mask.any())
        assert len(targets) > 0
        for target in targets:
            assert target.shape == (8, 4)
            assert bool(target.any())
            assert not bool((context_mask & target).any())

    def test_small_grid_many_seeds_no_degenerate_masks(self):
        """Sweep seeds on a small grid to stress the degenerate-case guards."""
        mask_gen = SpatioTemporalBlockMask()
        for seed in range(30):
            context_mask, targets = mask_gen(8, 4, _gen(seed))
            assert bool(context_mask.any()), f"empty context at seed={seed}"
            union = torch.zeros_like(context_mask)
            for target in targets:
                assert bool(target.any()), f"empty target at seed={seed}"
                union |= target
            assert bool(union.any()), f"empty target union at seed={seed}"
            assert torch.equal(context_mask, ~union)

    def test_custom_block_count_and_fractions(self):
        """Non-default n_blocks/coverage still yields valid disjoint masks."""
        mask_gen = SpatioTemporalBlockMask(
            n_blocks=3, ch_frac=(0.3, 0.4), t_frac=(0.2, 0.3), coverage=0.4
        )
        context_mask, targets = mask_gen(32, 8, _gen(0))

        assert len(targets) <= 3
        union = torch.zeros_like(context_mask)
        for target in targets:
            union |= target
        assert torch.equal(context_mask, ~union)
        assert bool(context_mask.any())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
