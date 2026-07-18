"""Non-gameable latent-quality / invariance metric panel.

This module implements a panel of representation-quality metrics for
self-supervised neural-signal embeddings. The design principle (see
SPEC_metrics.md) is that no single metric can be maxed out by a degenerate
(collapsed) representation: each metric's "cheat" is caught by another.
Callers should report the whole panel together and use ``collapse_report``
to apply the gaming-resistant reading rule rather than trusting any one
number in isolation.

All functions accept either ``np.ndarray`` or ``torch.Tensor`` inputs and
convert internally to ``float64`` numpy arrays for linear algebra.
"""

from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ArrayLike = np.ndarray | torch.Tensor

# Thresholds for the collapse-detection reading rule (see collapse_report).
_ERANK_COLLAPSE_THRESHOLD = 1.5
_IR_HIGH_THRESHOLD = 0.9
_ALIGNMENT_LOW_THRESHOLD = 0.1
_UNIFORMITY_COLLAPSE_THRESHOLD = -0.05

# Number of samples above which uniformity subsamples pairs for speed.
_UNIFORMITY_SUBSAMPLE_N = 2000
_UNIFORMITY_MAX_PAIRS = 200_000


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert a numpy array or torch tensor to a float64 numpy array.

    Args:
        x: Input array or tensor of any numeric dtype.

    Returns:
        A float64 numpy array with the same shape as ``x``.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _cov(x: np.ndarray, ddof: int = 1) -> np.ndarray:
    """Compute a (D, D) covariance matrix, robust to D=1 and tiny N.

    Args:
        x: Array of shape (N, D).
        ddof: Delta degrees of freedom for the covariance normalization.

    Returns:
        Covariance matrix of shape (D, D). Returns zeros if there are not
        enough samples to estimate variance.
    """
    x = np.atleast_2d(x)
    n, d = x.shape
    if n <= ddof:
        return np.zeros((d, d))
    xc = x - x.mean(axis=0, keepdims=True)
    return (xc.T @ xc) / (n - ddof)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize each row of a 2D array.

    Args:
        x: Array of shape (N, D).
        eps: Floor applied to row norms to avoid division by zero.

    Returns:
        Row-normalized array of shape (N, D).
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def _effective_rank_from_spectrum(values: np.ndarray, eps: float) -> float:
    """Shannon-entropy effective rank of a nonnegative spectrum.

    Shared core of the RankMe formula: given a set of nonnegative values
    (singular values or eigenvalues), treat the normalized values as a
    probability distribution and return exp(entropy). This is used by both
    ``effective_rank`` (singular values of Z) and ``lidar`` (eigenvalues of
    the LDA-whitened between/within covariance).

    Args:
        values: Nonnegative spectrum values (e.g. singular or eigen values).
        eps: Small constant added to each normalized value before taking
            the log, to avoid log(0), matching the RankMe definition.

    Returns:
        exp(-sum p_k log p_k), a float in roughly [1, len(values)]. Falls
        back to 1.0 when the spectrum has ~zero total mass (fully collapsed
        input), since entropy is undefined for an all-zero distribution.
    """
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= eps:
        return 1.0
    p = values / total + eps
    entropy = -float(np.sum(p * np.log(p)))
    return float(np.exp(entropy))


def _inv_sqrtm(mat: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Inverse matrix square root of a symmetric positive-definite matrix.

    Args:
        mat: Symmetric positive-(semi)definite matrix of shape (D, D).
        floor: Minimum eigenvalue used before inverting, for numerical
            safety.

    Returns:
        Matrix M such that M @ mat @ M ~= I, computed via eigendecomposition.
    """
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, floor, None)
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


def per_channel_vaf(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, Any]:
    """Per-channel Variance Accounted For (VAF).

    VAF_c = 1 - sum_bt (y - yhat)^2 / sum_bt (y - ybar_c)^2, where ybar_c is
    the mean of channel c over both batch and time.

    Args:
        y_true: Ground-truth signal, shape (B, C, T).
        y_pred: Predicted (denoised) signal, shape (B, C, T).

    Returns:
        Dict with keys:
            per_channel: np.ndarray of shape (C,), VAF per channel.
            mean: float, unweighted mean over channels.
            weighted_mean: float, mean weighted by per-channel variance of
                y_true (channels with near-zero true variance contribute
                ~nothing).
            median: float, median over channels.
            worst: float, minimum (worst-case) channel VAF.
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)

    ss_res = np.sum((yt - yp) ** 2, axis=(0, 2))
    y_bar = np.mean(yt, axis=(0, 2), keepdims=True)
    ss_tot = np.sum((yt - y_bar) ** 2, axis=(0, 2))

    ss_tot_safe = np.where(ss_tot > 1e-12, ss_tot, 1e-12)
    per_channel = 1.0 - ss_res / ss_tot_safe

    total_weight = float(ss_tot.sum())
    if total_weight > 1e-12:
        weighted_mean = float(np.sum(ss_tot * per_channel) / total_weight)
    else:
        weighted_mean = float(np.mean(per_channel))

    return {
        "per_channel": per_channel,
        "mean": float(np.mean(per_channel)),
        "weighted_mean": weighted_mean,
        "median": float(np.median(per_channel)),
        "worst": float(np.min(per_channel)),
    }


def vaf(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Global Variance Accounted For (kept for back-compat).

    Args:
        y_true: Ground-truth signal, any shape.
        y_pred: Predicted signal, same shape as y_true.

    Returns:
        1 - SSres / SStot computed over all elements pooled together.
    """
    yt = _to_numpy(y_true).ravel()
    yp = _to_numpy(y_pred).ravel()

    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))

    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return 1.0 - ss_res / ss_tot


def effective_rank(Z: ArrayLike, eps: float = 1e-7) -> float:  # noqa: N803
    """RankMe effective rank of an embedding matrix.

    erank = exp(-sum p_k log p_k), p_k = s_k / sum(s) + eps, where s are the
    singular values of the column-centered Z. Range is [1, min(N, D)]; a
    healthy representation is typically well above 0.7 * D, while a
    collapsed one sits near 1.

    Args:
        Z: Embedding matrix, shape (N, D).
        eps: Small constant added to normalized singular values before the
            entropy computation, to avoid log(0).

    Returns:
        Effective rank as a float.
    """
    z = _to_numpy(Z)
    z = z - z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(z, compute_uv=False)
    return _effective_rank_from_spectrum(s, eps)


def participation_ratio(Z: ArrayLike) -> float:  # noqa: N803
    """Participation ratio of an embedding's covariance spectrum.

    PR = (sum lambda)^2 / sum(lambda^2), over eigenvalues lambda of
    cov(Z). Range is [1, D]; cross-checks effective_rank with a different
    weighting of the spectrum (PR is more sensitive to a few dominant
    eigenvalues, effective_rank more sensitive to the tail).

    Args:
        Z: Embedding matrix, shape (N, D).

    Returns:
        Participation ratio as a float.
    """
    z = _to_numpy(Z)
    cov = _cov(z)
    eigvals = np.clip(np.linalg.eigvalsh(cov), 0.0, None)

    s1 = float(eigvals.sum())
    s2 = float(np.sum(eigvals**2))
    if s2 <= 1e-24:
        return 1.0
    return (s1**2) / s2


def lidar(views: list[ArrayLike], delta: float = 1e-3, eps: float = 1e-7) -> float:
    """LDA-whitened effective rank ("LiDAR") across repeated views.

    Views are repeated embeddings of the same N samples under different
    augmentations/views. This measures how much rank comes from genuine
    between-sample (signal) structure versus within-sample (nuisance/noise)
    structure, rewarding the former.

    Sigma_b is the covariance, across samples, of each sample's mean
    embedding over views (between-sample scatter). Sigma_w is the average,
    over samples, of the within-view covariance for that sample, plus
    delta * I for numerical stability. Sigma_lidar = Sigma_w^{-1/2}
    Sigma_b Sigma_w^{-1/2}; the returned value is the RankMe-style
    effective rank of Sigma_lidar's eigenvalues.

    Args:
        views: List of V arrays, each shape (N, D), one per view, for the
            same N samples in the same order.
        delta: Regularization added to the within-view covariance diagonal.
        eps: Small constant passed through to the effective-rank entropy
            computation.

    Returns:
        Effective rank of Sigma_lidar. Values near 1 indicate the rank is
        dominated by nuisance/noise rather than signal; higher values
        indicate genuine between-sample structure.
    """
    arrs = [_to_numpy(v) for v in views]
    stacked = np.stack(arrs, axis=1)  # (N, V, D)
    n_samples, n_views, d = stacked.shape

    per_sample_mean = stacked.mean(axis=1)  # (N, D)
    sigma_b = _cov(per_sample_mean)

    within_sum = np.zeros((d, d))
    if n_views > 1:
        for i in range(n_samples):
            within_sum += _cov(stacked[i])
    sigma_w = within_sum / n_samples + delta * np.eye(d)

    inv_sqrt_w = _inv_sqrtm(sigma_w)
    sigma_lidar = inv_sqrt_w @ sigma_b @ inv_sqrt_w
    eigvals = np.linalg.eigvalsh(sigma_lidar)
    return _effective_rank_from_spectrum(eigvals, eps)


def alignment(Z1: ArrayLike, Z2: ArrayLike, alpha: float = 2.0) -> float:  # noqa: N803
    """Alignment loss (Wang & Isola 2020) between paired embeddings.

    Rows are L2-normalized, then the mean ||z1 - z2||^alpha is taken over
    paired rows. Lower is better (paired views map to similar points), but
    a value of exactly 0 combined with low effective_rank signals collapse
    rather than genuine invariance.

    Args:
        Z1: First view's embeddings, shape (N, D).
        Z2: Second view's embeddings, shape (N, D), row-paired with Z1.
        alpha: Exponent applied to the pairwise L2 distance.

    Returns:
        Mean ||z1 - z2||^alpha over paired, L2-normalized rows.
    """
    z1 = _l2_normalize(_to_numpy(Z1))
    z2 = _l2_normalize(_to_numpy(Z2))
    dist = np.linalg.norm(z1 - z2, axis=1)
    return float(np.mean(dist**alpha))


def uniformity(Z: ArrayLike, t: float = 2.0) -> float:  # noqa: N803
    """Uniformity loss (Wang & Isola 2020) of an embedding on the sphere.

    Rows are L2-normalized, then log(mean(exp(-t * ||zi - zj||^2))) is
    computed over pairs i != j. Strongly negative is good (points spread
    over the sphere); a value near 0 indicates collapse (all points
    coincide). For speed, pairs are subsampled when N > 2000.

    Args:
        Z: Embedding matrix, shape (N, D).
        t: Kernel bandwidth for the Gaussian potential.

    Returns:
        Log mean pairwise Gaussian potential. Always <= 0.
    """
    z = _l2_normalize(_to_numpy(Z))
    n = z.shape[0]

    if n > _UNIFORMITY_SUBSAMPLE_N:
        rng = np.random.default_rng(0)
        idx_i = rng.integers(0, n, size=_UNIFORMITY_MAX_PAIRS)
        idx_j = rng.integers(0, n, size=_UNIFORMITY_MAX_PAIRS)
        keep = idx_i != idx_j
        idx_i, idx_j = idx_i[keep], idx_j[keep]
    else:
        idx_i, idx_j = np.triu_indices(n, k=1)

    sq_dist = np.sum((z[idx_i] - z[idx_j]) ** 2, axis=1)
    return float(np.log(np.mean(np.exp(-t * sq_dist))))


def invariance_ratio(Z1: ArrayLike, Z2: ArrayLike) -> dict[str, float]:  # noqa: N803
    """Invariance ratio (IR) between two paired embeddings.

    IR = 1 - E||f(v1) - f(v2)||^2 / (2 * tr(Cov(f))), where f is the pool
    of Z1 and Z2. This is collapse-resistant on its own terms (numerator
    and denominator both -> 0 together under collapse), which is why the
    denominator (tr(Cov)) is returned alongside so callers can flag
    near-zero-variance degeneracy explicitly.

    Args:
        Z1: First view's embeddings, shape (N, D).
        Z2: Second view's embeddings, shape (N, D), row-paired with Z1.

    Returns:
        Dict with keys:
            ir: float, the invariance ratio. Good representations score
                roughly 0.7 to 0.9 with a healthy denom.
            denom: float, tr(Cov(pooled)). Near 0 signals collapse; the
                ir value alone should not be trusted when denom is tiny.
    """
    z1 = _to_numpy(Z1)
    z2 = _to_numpy(Z2)

    mean_sq_dist = float(np.mean(np.sum((z1 - z2) ** 2, axis=1)))
    pooled = np.concatenate([z1, z2], axis=0)
    denom = float(np.trace(_cov(pooled)))

    ir = 1.0 - mean_sq_dist / (2.0 * (denom + 1e-12))
    return {"ir": ir, "denom": denom}


def nuisance_probe(
    Z: ArrayLike,  # noqa: N803
    nuisance_labels: ArrayLike,
    n_splits: int = 5,
) -> float:
    """Cross-validated linear probe accuracy for a nuisance variable.

    Fits a linear logistic regression to predict the nuisance label (e.g.
    noise seed, channel subset) from Z, under stratified k-fold
    cross-validation. A good representation should be uninformative about
    nuisances, so accuracy near chance (1 / n_classes) is desired; well
    above chance indicates the nuisance leaked into the embedding.

    Args:
        Z: Embedding matrix, shape (N, D).
        nuisance_labels: Array-like of length N, nuisance class per sample.
        n_splits: Requested number of cross-validation folds. Reduced
            automatically if the smallest class has fewer members.

    Returns:
        Mean held-out classification accuracy across folds, as a float.
    """
    z = _to_numpy(Z)
    labels = np.asarray(nuisance_labels)

    _, class_counts = np.unique(labels, return_counts=True)
    n_splits_eff = max(2, min(n_splits, int(class_counts.min())))

    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=0)
    accuracies = []
    for train_idx, test_idx in skf.split(z, labels):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(z[train_idx], labels[train_idx])
        preds = clf.predict(z[test_idx])
        accuracies.append(float(np.mean(preds == labels[test_idx])))

    return float(np.mean(accuracies))


def collapse_report(
    Z: ArrayLike,  # noqa: N803
    views: list[ArrayLike] | None = None,
    Z_pair: tuple[ArrayLike, ArrayLike] | None = None,  # noqa: N803
    nuisance_labels: ArrayLike | None = None,
) -> dict[str, Any]:
    """Compute the full metric panel and apply the gaming-resistant reading rule.

    No single metric in this panel can be maxed out by a degenerate
    representation without another metric catching it. This function
    gathers whichever metrics the provided inputs allow and flags
    'collapsed' when the pattern is: apparent invariance (high
    invariance_ratio or low alignment) that is actually near-total
    representational collapse (effective_rank <= ~1.5), or when uniformity
    is ~0 (points coincide on the sphere).

    Args:
        Z: Embedding matrix, shape (N, D). Used for effective_rank,
            participation_ratio, and uniformity.
        views: Optional list of (N, D) arrays, repeated embeddings of the
            same samples under different views/augmentations. Used for
            lidar, and as a fallback source for Z_pair (first two views)
            if Z_pair is not given.
        Z_pair: Optional (Z1, Z2) tuple of paired (N, D) embeddings, used
            for alignment and invariance_ratio.
        nuisance_labels: Optional array-like of length N, nuisance class
            per sample in Z. Used for nuisance_probe.

    Returns:
        Dict with the individual metric values that could be computed
        (effective_rank, participation_ratio, uniformity, and any of
        lidar, alignment, invariance_ratio/denom, nuisance_probe_accuracy
        that the inputs support), plus:
            collapsed: bool, True if the reading rule detects collapse.
            reasons: list[str], human-readable triggers for the flag.
    """
    report: dict[str, Any] = {
        "effective_rank": effective_rank(Z),
        "participation_ratio": participation_ratio(Z),
        "uniformity": uniformity(Z),
    }

    pair = Z_pair
    if pair is None and views is not None and len(views) >= 2:
        pair = (views[0], views[1])

    if pair is not None:
        z1, z2 = pair
        report["alignment"] = alignment(z1, z2)
        ir_result = invariance_ratio(z1, z2)
        report["invariance_ratio"] = ir_result["ir"]
        report["invariance_denom"] = ir_result["denom"]

    if views is not None:
        report["lidar"] = lidar(views)

    if nuisance_labels is not None:
        report["nuisance_probe_accuracy"] = nuisance_probe(Z, nuisance_labels)

    reasons = []
    erank = report["effective_rank"]
    near_collapsed_rank = erank <= _ERANK_COLLAPSE_THRESHOLD

    apparent_invariance = False
    if "invariance_ratio" in report and report["invariance_ratio"] >= _IR_HIGH_THRESHOLD:
        apparent_invariance = True
    if "alignment" in report and report["alignment"] <= _ALIGNMENT_LOW_THRESHOLD:
        apparent_invariance = True

    if apparent_invariance and near_collapsed_rank:
        reasons.append(
            "high invariance / low alignment with effective_rank <= "
            f"{_ERANK_COLLAPSE_THRESHOLD} (collapse masquerading as invariance)"
        )

    if report["uniformity"] >= _UNIFORMITY_COLLAPSE_THRESHOLD:
        reasons.append(f"uniformity >= {_UNIFORMITY_COLLAPSE_THRESHOLD} (embeddings coincide)")

    report["collapsed"] = len(reasons) > 0
    report["reasons"] = reasons
    return report
