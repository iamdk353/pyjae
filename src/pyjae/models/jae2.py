"""JAE2: JEPA-style (Joint-Embedding Predictive Architecture) model.

Replaces the earlier U-Net-ensemble JAE2 with a single shared encoder used for
both a masked "context" view and an unmasked "target" view of the same input,
following I-JEPA / V-JEPA / Laya (LeJEPA for EEG). Context and target tokens are
compared in LATENT space (not signal space): a small predictor guesses the
target-position embeddings from a context summary plus positional queries, and
a VICReg variance/covariance term on the pooled embeddings prevents collapse.
Stop-gradient (not EMA) decorrelates the two branches, per the "Laya-style"
recipe in the design spec.

A lightweight decoder head, trained jointly via a reconstruction term, turns the
full-input encoding into a denoised signal for the VAF benchmark.
"""

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from pyjae.losses import jae2_jepa_loss_fn
from pyjae.models.encoders import ConvPatchEncoder
from pyjae.views import JEPAMask, SpatioTemporalBlockMask


def _match_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad with zeros or crop the last dimension of ``x`` to ``target_len``.

    Args:
        x: Tensor whose last dimension should be adjusted.
        target_len: Desired size of the last dimension.

    Returns:
        Tensor with the same shape as ``x`` except the last dimension, which is
        exactly ``target_len``.
    """
    length = x.shape[-1]
    if length == target_len:
        return x
    if length < target_len:
        return F.pad(x, (0, target_len - length))
    return x[..., :target_len]


class JAE2Output(NamedTuple):
    """Outputs of a single :class:`JAE2` forward pass.

    Attributes:
        denoised: Decoder-head reconstruction of the full (unmasked) input,
            shape ``(B, C, T)``.
        pred_tokens: Predictor's guesses for the target-position embeddings,
            stacked over every (batch, target-position) pair, shape
            ``(M, d_model)`` where ``M = B * n_target_positions``.
        target_tokens: Encoder tokens at the same target positions, taken from
            a separate full-input (unmasked) encoder pass, shape
            ``(M, d_model)``, same order as ``pred_tokens``. These carry a live
            autograd graph back to the encoder; the JEPA loss detaches them
            internally (see :func:`pyjae.losses.jepa_loss`).
        z_context: Projected, pooled context-branch embeddings used for the
            VICReg variance/covariance terms, shape ``(N, latent_dim)`` where
            ``N = B * n_patches``.
        z_target: Projected, pooled target-branch embeddings used for the
            VICReg variance/covariance terms, shape ``(N, latent_dim)``.
    """

    denoised: torch.Tensor
    pred_tokens: torch.Tensor
    target_tokens: torch.Tensor
    z_context: torch.Tensor
    z_target: torch.Tensor


class JAE2(nn.Module):
    """JEPA-style Joint Autoencoder: predicts masked-region embeddings, not signal.

    A single shared :class:`~pyjae.models.encoders.ConvPatchEncoder` is run twice per
    forward call:

    1. **Context pass**: the raw input is zeroed out at every timepoint that falls
       inside a masked target block (see "Masking" below), then encoded. The
       resulting tokens at the still-visible (context) grid positions are
       mean-pooled into a single per-sample "context summary" vector.
    2. **Target pass**: the *unmasked* input is encoded in a separate forward
       call through the same encoder. The tokens at the target grid positions
       from this pass are the prediction targets.

    Because both passes share one encoder (no EMA copy), the target pass still
    builds a live autograd graph. Stop-gradient is applied only inside the loss
    (via ``pyjae.losses.jepa_loss``/``jae2_jepa_loss_fn``), not by detaching eagerly
    in the model, so the pooled target embeddings (``z_target``) still receive
    gradient through the VICReg variance/covariance terms while the raw
    prediction target (``target_tokens``) does not receive gradient through the
    prediction term. See the class-level correctness note in the loss method.

    Masking:
        A single mask, shared across the whole batch, is sampled once per
        forward call over the ``(n_channels, n_patches)`` grid using ``self.mask``
        (default :class:`~pyjae.views.SpatioTemporalBlockMask`). The union of the
        sampled target blocks gives the target grid positions; its complement is
        the context. The target region is "dropped" from the context pass by
        zeroing the corresponding raw timepoints (``patch_len`` samples per
        masked grid cell) before the context encoder call, so the context
        encoder never sees target-region signal.

    Predictor design (kept intentionally small, MLP-only, no attention):
        For every target grid position ``(c, p)`` a query vector is built as
        ``mask_token + channel_embedding[c] + patch_embedding[p]`` (all
        ``d_model``-dim, learned). This query is added to the per-sample context
        summary vector (broadcast over target positions) and passed through a
        2-layer MLP (``d_model -> predictor_dim -> d_model``, GELU + LayerNorm in
        between) to produce the predicted target embedding. This is the "lean"
        option from the design spec: a single global context summary conditions
        every target query, rather than per-position cross-attention.

    Projector: a single ``Linear(d_model, latent_dim)`` projects pooled
        embeddings (mean over channels, per patch position) down to
        ``latent_dim`` for the VICReg variance/covariance terms. A single linear
        layer is enough here: VICReg regularizes second-order statistics of
        whatever space it is given, and keeping the projector shallow avoids
        adding an extra place for collapse to hide.

    Decoder head (denoising readout): a small per-token MLP
        (``Linear(d_model, d_model) -> GELU -> Linear(d_model, patch_len)``) maps
        each full-input encoder token back to its ``patch_len`` raw samples;
        tokens are reassembled along the time axis and padded/cropped to match
        the input length. This reuses the target pass's full-input encoding, so
        no third encoder call is needed.

    Args:
        input_dim: Number of input channels ``C``.
        n_timepoints: Expected number of input timepoints ``T``.
        patch_len: Timepoints per patch; also the patch conv kernel/stride.
        d_model: Encoder token embedding dimension.
        latent_dim: Output dimension of the VICReg projector.
        predictor_dim: Hidden width of the predictor MLP.
        mask: A :class:`~pyjae.views.JEPAMask` strategy. Defaults to
            ``SpatioTemporalBlockMask()``.
        seed: Seed for the default mask generator used when ``forward`` is
            called without an explicit ``generator``.
    """

    def __init__(
        self,
        input_dim: int,
        n_timepoints: int = 128,
        patch_len: int = 8,
        d_model: int = 128,
        latent_dim: int = 64,
        predictor_dim: int = 64,
        mask: JEPAMask | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if n_timepoints % patch_len != 0:
            raise ValueError(
                f"n_timepoints ({n_timepoints}) must be a multiple of patch_len ({patch_len})."
            )

        self.input_dim = input_dim
        self.n_timepoints = n_timepoints
        self.patch_len = patch_len
        self.n_patches = n_timepoints // patch_len
        self.d_model = d_model
        self.latent_dim = latent_dim

        self.encoder = ConvPatchEncoder(
            n_channels=input_dim,
            n_patches=self.n_patches,
            patch_len=patch_len,
            d_model=d_model,
        )
        self.mask: JEPAMask = mask if mask is not None else SpatioTemporalBlockMask()

        # Predictor: mask token + (channel, patch) positional embeddings define a
        # query per target grid position; the query is added to a per-sample
        # context summary and mapped through a small MLP to a predicted token.
        self.mask_token = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.channel_pos_embed = nn.Embedding(input_dim, d_model)
        self.patch_pos_embed = nn.Embedding(self.n_patches, d_model)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, predictor_dim),
            nn.GELU(),
            nn.LayerNorm(predictor_dim),
            nn.Linear(predictor_dim, d_model),
        )

        # Projector for VICReg variance/covariance terms.
        self.projector = nn.Linear(d_model, latent_dim)

        # Denoising readout head: per-token MLP back to raw patch samples.
        self.decoder_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, patch_len),
        )

        self._generator = torch.Generator().manual_seed(seed)
        self._last_input: torch.Tensor | None = None

    def _time_mask_from_grid(self, target_mask: torch.Tensor, t: int) -> torch.Tensor:
        """Expand a (channel, patch) grid mask into a (channel, time) mask.

        Args:
            target_mask: BoolTensor(n_channels, n_patches), True at masked
                target grid positions.
            t: Number of raw timepoints to expand to.

        Returns:
            BoolTensor(n_channels, t), True at every raw timepoint that falls
            inside a masked patch.
        """
        expanded = target_mask.repeat_interleave(self.patch_len, dim=1)
        return _match_length(expanded, t)

    def _decode(self, tokens: torch.Tensor, t: int) -> torch.Tensor:
        """Map full-input encoder tokens back to a denoised (B, C, T) signal.

        Args:
            tokens: Encoder tokens, shape ``(B, C, n_patches, d_model)``.
            t: Target number of raw timepoints for the output.

        Returns:
            Tensor of shape ``(B, C, t)``.
        """
        batch_size, n_channels, n_patches, _ = tokens.shape
        per_patch = self.decoder_head(tokens)  # (B, C, P, patch_len)
        flat = per_patch.reshape(batch_size, n_channels, n_patches * self.patch_len)
        return _match_length(flat, t)

    def forward(self, x: torch.Tensor, generator: torch.Generator | None = None) -> JAE2Output:
        """Run one context/target JEPA pass plus the denoising readout.

        Args:
            x: Input tensor of shape ``(B, C, T)``, ``C == input_dim``.
            generator: Torch generator used to sample the mask. Defaults to this
                model's internal generator (seeded at construction time via
                ``seed``), which advances across calls so repeated forward calls
                see different masks while remaining reproducible end-to-end.

        Returns:
            A :class:`JAE2Output` with ``denoised``, ``pred_tokens``,
            ``target_tokens``, ``z_context`` and ``z_target``.
        """
        batch_size, n_channels, t = x.shape
        if n_channels != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} channels, got {n_channels}.")

        gen = generator if generator is not None else self._generator

        context_mask, target_blocks = self.mask(self.input_dim, self.n_patches, gen)
        context_mask = context_mask.to(x.device)
        target_mask = torch.zeros_like(context_mask)
        for block in target_blocks:
            target_mask |= block.to(x.device)

        # Context pass: zero out target-region timepoints, then encode.
        time_mask = self._time_mask_from_grid(target_mask, t)
        x_context = x.masked_fill(time_mask.unsqueeze(0), 0.0)
        tokens_ctx, pooled_ctx = self.encoder(x_context)

        # Target pass: encode the full, unmasked input (separate forward call).
        tokens_full, pooled_full = self.encoder(x)

        # Context summary: mean-pool encoder tokens at visible positions only.
        context_tokens = tokens_ctx[:, context_mask]  # (B, N_ctx, d_model)
        context_summary = context_tokens.mean(dim=1)  # (B, d_model)

        # Target queries: mask token + positional embedding per target position.
        c_idx, p_idx = torch.nonzero(target_mask, as_tuple=True)
        n_targets = c_idx.shape[0]
        query = self.mask_token + self.channel_pos_embed(c_idx) + self.patch_pos_embed(p_idx)

        predictor_in = context_summary.unsqueeze(1) + query.unsqueeze(0)  # (B, N_tgt, d_model)
        pred_tokens = self.predictor(predictor_in.reshape(batch_size * n_targets, self.d_model))

        # Same grid order as `query` (both derived from `target_mask`'s nonzero order).
        target_tokens = tokens_full[:, target_mask].reshape(batch_size * n_targets, self.d_model)

        # VICReg projections: flatten (B, n_patches, d_model) -> (B * n_patches, d_model).
        z_context = self.projector(pooled_ctx.reshape(batch_size * self.n_patches, self.d_model))
        z_target = self.projector(pooled_full.reshape(batch_size * self.n_patches, self.d_model))

        denoised = self._decode(tokens_full, t)

        self._last_input = x

        return JAE2Output(
            denoised=denoised,
            pred_tokens=pred_tokens,
            target_tokens=target_tokens,
            z_context=z_context,
            z_target=z_target,
        )

    def loss(
        self,
        out: JAE2Output,
        lambda_pred: float = 25.0,
        lambda_var: float = 25.0,
        lambda_cov: float = 1.0,
        recon_weight: float = 1.0,
    ) -> torch.Tensor:
        """Compute the total JAE2 loss: JEPA prediction/VICReg plus denoising recon.

        Correctness note (stop-gradient): ``out.target_tokens`` still carries a
        live autograd graph back to the shared encoder (the target pass is not
        detached in ``forward``). ``jae2_jepa_loss_fn`` calls
        ``pyjae.losses.jepa_loss`` with ``stop_grad=True``, which detaches its
        local copy of the target before the Smooth L1 comparison; this cuts the
        gradient path from the prediction term into the target pass, so the
        encoder cannot lower the prediction loss by "meeting the predictor
        halfway" (collapsing the target embedding to something trivially easy to
        predict). ``out.z_target`` is passed through unmodified (not
        pre-detached), so the VICReg variance/covariance terms still train the
        target-branch encoder pass directly, which is required to keep that
        branch from collapsing. See ``tests/test_jae2.py`` for a unit test that
        directly checks ``target_tokens.grad`` is untouched by the prediction
        term.

        Args:
            out: Output of a prior call to ``forward`` on this model instance
                (``forward`` must be called first, since it stashes the input
                used for the reconstruction term).
            lambda_pred: Weight for the JEPA prediction (Smooth L1) term.
            lambda_var: Weight for the VICReg variance hinge term.
            lambda_cov: Weight for the VICReg covariance term.
            recon_weight: Weight for the denoising reconstruction (MSE) term.

        Returns:
            Scalar total loss tensor.
        """
        if self._last_input is None:
            raise RuntimeError("forward() must be called before loss() (no stashed input).")

        jepa = jae2_jepa_loss_fn(
            out.pred_tokens,
            out.target_tokens,
            out.z_context,
            out.z_target,
            lambda_pred=lambda_pred,
            lambda_var=lambda_var,
            lambda_cov=lambda_cov,
        )
        recon = F.mse_loss(out.denoised, self._last_input)
        return jepa + recon_weight * recon

    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the full (unmasked) input and decode a denoised signal.

        Args:
            x: Input tensor of shape ``(B, C, T)``, ``C == input_dim``.

        Returns:
            Denoised reconstruction of shape ``(B, C, T)``.
        """
        tokens_full, _ = self.encoder(x)
        return self._decode(tokens_full, x.shape[-1])
