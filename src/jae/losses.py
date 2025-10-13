"""
Loss functions for JAE models.

This module implements various loss functions for training joint autoencoders:
- Original JAE loss (MSE-based)
- Modernized JAE2 loss (Huber + VICReg)
"""

import torch
import torch.nn.functional as F


def jae1_loss_fn(x1_hat, x2_hat, z1, z2, x1_target, x2_target, latent_weight=1.0):
    """
    Original JAE loss function from Altan et al. (2021).

    Combines reconstruction MSE for both views with latent space alignment MSE.
    The loss encourages both accurate reconstruction and similar latent representations
    across the two autoencoders.

    Args:
        x1_hat (torch.Tensor): Reconstructed first view, shape (batch, channels, time).
        x2_hat (torch.Tensor): Reconstructed second view, shape (batch, channels, time).
        z1 (torch.Tensor): Latent from first autoencoder, shape (batch*time, latent_dim).
        z2 (torch.Tensor): Latent from second autoencoder, shape (batch*time, latent_dim).
        x1_target (torch.Tensor): Target first view, shape (batch, channels, time).
        x2_target (torch.Tensor): Target second view, shape (batch, channels, time).
        latent_weight (float, optional): Weight for latent alignment loss. Default: 1.0.

    Returns:
        torch.Tensor: Scalar loss value.

    Example:
        >>> x1_hat = torch.randn(16, 48, 128)
        >>> x2_hat = torch.randn(16, 48, 128)
        >>> z1 = torch.randn(2048, 12)
        >>> z2 = torch.randn(2048, 12)
        >>> loss = jae1_loss_fn(x1_hat, x2_hat, z1, z2, x1_hat, x2_hat)
        >>> print(loss.shape)  # torch.Size([])
    """
    # Reconstruction loss (calculated on (B, C, T) tensors)
    recon_loss = F.mse_loss(x1_hat, x1_target) + F.mse_loss(x2_hat, x2_target)

    # Latent loss (calculated on (B*T, Latent_Dim) tensors)
    latent_loss = F.mse_loss(z1, z2)

    return recon_loss + latent_weight * latent_loss


def huber_reconstruction_loss(reconstructions, targets, delta=1.0):
    """
    Huber (Smooth L1) reconstruction loss for multiple views.

    Huber loss is less sensitive to outliers than MSE, making it more robust for
    neural data with occasional high-amplitude artifacts. It behaves like MSE for
    small errors and like MAE for large errors.

    Args:
        reconstructions (list of torch.Tensor): List of reconstructed views, each with
            shape (batch, channels, time).
        targets (list of torch.Tensor): List of target views, each with shape
            (batch, channels, time).
        delta (float, optional): Threshold where the loss transitions from quadratic
            to linear. Default: 1.0.

    Returns:
        torch.Tensor: Scalar loss value (average across all views).

    Example:
        >>> recons = [torch.randn(8, 48, 64) for _ in range(5)]
        >>> targets = [torch.randn(8, 48, 64) for _ in range(5)]
        >>> loss = huber_reconstruction_loss(recons, targets)
        >>> print(loss.shape)  # torch.Size([])
    """
    loss = 0
    N = len(reconstructions)
    if N == 0:
        return torch.tensor(0.0)

    for recon, target in zip(reconstructions, targets):
        # F.smooth_l1_loss is equivalent to Huber loss when beta=delta
        loss += F.smooth_l1_loss(recon, target, beta=delta, reduction='mean')

    return loss / N


def vicreg_loss(latents, lambda_inv=25.0, mu_var=25.0, nu_cov=1.0, epsilon=1e-4):
    """
    VICReg (Variance-Invariance-Covariance Regularization) loss.

    VICReg aligns latent representations across multiple views while maintaining
    informative representations. Unlike MSE, VICReg is invariant to linear
    transformations (rotation, scaling) in the latent space.

    The loss has three components:
    1. Invariance: Encourages similar representations across views
    2. Variance: Prevents collapse by maintaining variance in each dimension
    3. Covariance: Decorrelates dimensions to maximize information content

    Reference: Bardes, A., Ponce, J., & LeCun, Y. (2022). "VICReg: 
    Variance-Invariance-Covariance Regularization for Self-Supervised Learning." ICLR.

    Args:
        latents (list of torch.Tensor): List of latent representations, each with
            shape (batch, latent_dim).
        lambda_inv (float, optional): Weight for invariance loss. Default: 25.0.
        mu_var (float, optional): Weight for variance loss. Default: 25.0.
        nu_cov (float, optional): Weight for covariance loss. Default: 1.0.
        epsilon (float, optional): Small constant for numerical stability. Default: 1e-4.

    Returns:
        torch.Tensor: Scalar loss value.

    Example:
        >>> latents = [torch.randn(32, 12) for _ in range(5)]
        >>> loss = vicreg_loss(latents)
        >>> print(loss.shape)  # torch.Size([])
    """
    num_views = len(latents)
    if num_views < 2:
        return torch.tensor(0.0, device=latents[0].device if len(latents) > 0 else 'cpu')

    # 1. Invariance Loss (MSE between all pairs of views)
    invariance_loss = 0
    count = 0
    for i in range(num_views):
        for j in range(i + 1, num_views):
            invariance_loss += F.mse_loss(latents[i], latents[j])
            count += 1
    if count > 0:
        invariance_loss /= count

    # 2. Variance and Covariance Loss
    variance_loss = 0
    covariance_loss = 0
    for z in latents:
        B, D = z.shape
        if B <= 1:
            # Cannot calculate covariance/variance with batch size 1
            continue

        # Center the representation
        z = z - z.mean(dim=0)
        cov_z = (z.T @ z) / (B - 1)

        # Variance Loss (Hinge loss on the standard deviation)
        # Encourages std(z_d) >= 1 for each dimension d
        std_z = torch.sqrt(cov_z.diag() + epsilon)
        variance_loss += torch.mean(F.relu(1.0 - std_z))

        # Covariance Loss (L2 norm of the off-diagonal elements)
        # Encourages decorrelation between different dimensions
        if D > 1:
            mask = (~torch.eye(D, dtype=torch.bool, device=z.device))
            covariance_loss += cov_z[mask].pow(2).sum() / D

    if num_views > 0:
        variance_loss /= num_views
        covariance_loss /= num_views

    # Total Loss (Weighted Sum)
    return (lambda_inv * invariance_loss + mu_var * variance_loss + nu_cov * covariance_loss)


def temporal_smoothness_loss(denoised_output):
    """
    Temporal smoothness regularization to reduce high-frequency artifacts.
    
    Penalizes large differences between consecutive time steps, encouraging
    smoother outputs. This helps prevent jagged/oscillatory reconstructions
    common in U-Net architectures.
    
    Args:
        denoised_output (torch.Tensor): Denoised signal of shape (B, C, T).
    
    Returns:
        torch.Tensor: Scalar smoothness loss (mean of temporal differences).
    
    Example:
        >>> x = torch.randn(8, 96, 128)
        >>> loss = temporal_smoothness_loss(x)
        >>> print(loss.shape)  # torch.Size([])
    """
    # Compute first-order temporal differences
    diff = denoised_output[:, :, 1:] - denoised_output[:, :, :-1]
    # L2 norm of differences
    return torch.mean(diff ** 2)


def jae2_loss_fn(
    reconstructions,
    latents,
    targets,
    denoised_output=None,
    recon_weight=1.0,
    vicreg_weight=0.1,
    smoothness_weight=0.0,
    huber_delta=1.0,
    lambda_inv=25.0,
    mu_var=25.0,
    nu_cov=1.0
):
    """
    Combined loss function for JAE2.

    Combines robust Huber reconstruction loss with VICReg latent alignment.
    The reconstruction term ensures accurate denoising, while the VICReg term
    encourages consistent, informative latent representations across views.

    Args:
        reconstructions (list of torch.Tensor): List of reconstructed views from
            each network.
        latents (list of torch.Tensor): List of latent representations from each network.
        targets (list of torch.Tensor): List of target (input) views for each network.
        denoised_output (torch.Tensor, optional): Final denoised output (B, C, T) for
            smoothness regularization. If None, smoothness is not applied.
        recon_weight (float, optional): Weight for reconstruction loss. Default: 1.0.
        vicreg_weight (float, optional): Weight for VICReg loss. Default: 0.1.
        smoothness_weight (float, optional): Weight for temporal smoothness. Default: 0.0 (disabled).
        huber_delta (float, optional): Delta parameter for Huber loss. Default: 1.0.
        lambda_inv (float, optional): VICReg invariance weight. Default: 25.0.
        mu_var (float, optional): VICReg variance weight. Default: 25.0.
        nu_cov (float, optional): VICReg covariance weight. Default: 1.0.

    Returns:
        torch.Tensor: Scalar loss value.

    Example:
        >>> recons = [torch.randn(8, 48, 64) for _ in range(5)]
        >>> latents = [torch.randn(8, 12) for _ in range(5)]
        >>> targets = [torch.randn(8, 48, 64) for _ in range(5)]
        >>> loss = jae2_loss_fn(recons, latents, targets)
        >>> print(loss.shape)  # torch.Size([])
    """
    recon_loss = huber_reconstruction_loss(reconstructions, targets, delta=huber_delta)
    v_loss = vicreg_loss(latents, lambda_inv=lambda_inv, mu_var=mu_var, nu_cov=nu_cov)
    
    total_loss = recon_weight * recon_loss + vicreg_weight * v_loss
    
    # Add temporal smoothness if enabled and denoised output is provided
    if smoothness_weight > 0 and denoised_output is not None:
        smooth_loss = temporal_smoothness_loss(denoised_output)
        total_loss = total_loss + smoothness_weight * smooth_loss

    return total_loss

