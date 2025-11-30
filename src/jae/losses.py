"""
Loss functions for JAE models.
"""

import torch
import torch.nn.functional as F


def jae1_loss_fn(x1_hat, x2_hat, z1, z2, x1_target, x2_target, latent_weight=1.0):
    """
    Original JAE loss function from Altan et al. (2021), Eq 3.

    C = MSE(X1, X̂1) + MSE(X2, X̂2) + ||Z1 - Z2||²

    Parameters
    ----------
    x1_hat : Tensor, shape (batch, channels, time)
        Reconstructed first partition.
    x2_hat : Tensor, shape (batch, channels, time)
        Reconstructed second partition.
    z1 : Tensor, shape (batch*time, latent_dim)
        Latent from first autoencoder.
    z2 : Tensor, shape (batch*time, latent_dim)
        Latent from second autoencoder.
    x1_target : Tensor
        Target first partition.
    x2_target : Tensor
        Target second partition.
    latent_weight : float, default=1.0
        Weight for latent alignment loss.

    Returns
    -------
    loss : Tensor
        Scalar loss value.
    """
    recon_loss = F.mse_loss(x1_hat, x1_target) + F.mse_loss(x2_hat, x2_target)
    latent_loss = F.mse_loss(z1, z2)
    return recon_loss + latent_weight * latent_loss


def huber_reconstruction_loss(reconstructions, targets, delta=1.0):
    """
    Huber (Smooth L1) reconstruction loss for multiple views.

    Parameters
    ----------
    reconstructions : list of Tensor
        Reconstructed views.
    targets : list of Tensor
        Target views.
    delta : float, default=1.0
        Huber threshold.

    Returns
    -------
    loss : Tensor
        Average loss across views.
    """
    if len(reconstructions) == 0:
        return torch.tensor(0.0)

    loss = sum(
        F.smooth_l1_loss(recon, target, beta=delta, reduction="mean")
        for recon, target in zip(reconstructions, targets)
    )
    return loss / len(reconstructions)


def vicreg_loss(latents, lambda_inv=25.0, mu_var=25.0, nu_cov=1.0, epsilon=1e-4):
    """
    VICReg (Variance-Invariance-Covariance) loss for latent alignment.

    Reference: Bardes et al. (2022), ICLR.

    Parameters
    ----------
    latents : list of Tensor, each shape (batch, latent_dim)
        Latent representations from each network.
    lambda_inv : float
        Weight for invariance (alignment) loss.
    mu_var : float
        Weight for variance loss.
    nu_cov : float
        Weight for covariance loss.
    epsilon : float
        Numerical stability constant.

    Returns
    -------
    loss : Tensor
        Scalar loss value.
    """
    num_views = len(latents)
    if num_views < 2:
        return torch.tensor(0.0, device=latents[0].device if latents else "cpu")

    # Invariance loss (MSE between pairs)
    invariance_loss = 0
    count = 0
    for i in range(num_views):
        for j in range(i + 1, num_views):
            invariance_loss += F.mse_loss(latents[i], latents[j])
            count += 1
    if count > 0:
        invariance_loss /= count

    # Variance and covariance loss
    variance_loss = 0
    covariance_loss = 0

    for z in latents:
        B, D = z.shape
        if B <= 1:
            continue

        z_centered = z - z.mean(dim=0)
        cov_z = (z_centered.T @ z_centered) / (B - 1)

        # Variance: encourage std >= 1
        std_z = torch.sqrt(cov_z.diag() + epsilon)
        variance_loss += torch.mean(F.relu(1.0 - std_z))

        # Covariance: decorrelate dimensions
        if D > 1:
            mask = ~torch.eye(D, dtype=torch.bool, device=z.device)
            covariance_loss += cov_z[mask].pow(2).sum() / D

    if num_views > 0:
        variance_loss /= num_views
        covariance_loss /= num_views

    return lambda_inv * invariance_loss + mu_var * variance_loss + nu_cov * covariance_loss


def mse_latent_alignment(latents):
    """
    MSE-based latent alignment loss (original paper approach).

    Parameters
    ----------
    latents : list of Tensor
        Latent representations from each network.

    Returns
    -------
    loss : Tensor
        Average pairwise MSE.
    """
    num_views = len(latents)
    if num_views < 2:
        return torch.tensor(0.0, device=latents[0].device if latents else "cpu")

    loss = 0
    count = 0
    for i in range(num_views):
        for j in range(i + 1, num_views):
            loss += F.mse_loss(latents[i], latents[j])
            count += 1

    return loss / count if count > 0 else torch.tensor(0.0)


def jae2_loss_fn(
    reconstructions,
    latents,
    targets,
    use_vicreg=False,
    recon_weight=1.0,
    latent_weight=1.0,
    huber_delta=1.0,
):
    """
    Loss function for enhanced JAE (multiple networks).

    Parameters
    ----------
    reconstructions : list of Tensor
        Reconstructed views from each network.
    latents : list of Tensor
        Latent representations from each network.
    targets : list of Tensor
        Target views for each network.
    use_vicreg : bool, default=False
        Use VICReg instead of MSE for latent alignment.
    recon_weight : float, default=1.0
        Weight for reconstruction loss.
    latent_weight : float, default=1.0
        Weight for latent alignment loss.
    huber_delta : float, default=1.0
        Delta for Huber loss.

    Returns
    -------
    loss : Tensor
        Scalar loss value.
    """
    recon_loss = huber_reconstruction_loss(reconstructions, targets, delta=huber_delta)

    if use_vicreg:
        latent_loss = vicreg_loss(latents)
        latent_weight = 0.1  # VICReg needs lower weight
    else:
        latent_loss = mse_latent_alignment(latents)

    return recon_weight * recon_loss + latent_weight * latent_loss
