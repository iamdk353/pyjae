#!/usr/bin/env python
"""
Final JAE validation script with progress output.
"""

import sys
sys.path.insert(0, 'src')

import torch
from jae import JAE, simulate_neural_data, calculate_vaf, run_pca_baseline, set_seed

print("=" * 60)
print("JAE VALIDATION")
print("=" * 60)
print("Setting seed...", flush=True)
set_seed(42)

# Generate test data
print("\n[1] Generating simulated neural data...", flush=True)
clean, noisy, info = simulate_neural_data(
    n_samples=300,
    n_channels=96,
    n_timepoints=128,
    latent_dim=6,
    snr_db=5.0,
    nonlinear=True,
    alpha=2.0,
    seed=42
)
print(f"    Shape: {noisy.shape}", flush=True)
print(f"    SNR: {info['snr_db']:.1f} dB", flush=True)

# PCA baseline
print("\n[2] Computing PCA baseline...", flush=True)
pca_denoised = run_pca_baseline(noisy, latent_dim=8)
vaf_pca = calculate_vaf(clean, pca_denoised)
print(f"    PCA VAF: {vaf_pca:.3f}", flush=True)

# JAE
print("\n[3] Training JAE (300 epochs)...", flush=True)
model = JAE(latent_dim=8, verbose=False)

# Custom training with progress
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from jae.losses import jae1_loss_fn

X = torch.tensor(noisy.numpy(), dtype=torch.float32)
model._initialize_model(X.shape[1])
model.optimizer = torch.optim.AdamW(model.model.parameters(), lr=0.001, weight_decay=1e-5)

train_loader = DataLoader(TensorDataset(X, X), batch_size=32, shuffle=True, drop_last=True)
model.model.train()

for epoch in range(300):
    epoch_loss = 0
    n_batches = 0
    for batch_data, _ in train_loader:
        batch_data = batch_data.to(model.device)
        model.optimizer.zero_grad()
        x_denoised, z1, z2, x1_target, x2_target = model.model(batch_data)
        half_dim = model.model.half_dim
        x1_hat = x_denoised[:, :half_dim, :]
        x2_hat = x_denoised[:, half_dim:, :]
        loss = jae1_loss_fn(x1_hat, x2_hat, z1, z2, x1_target, x2_target)
        loss.backward()
        model.optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    
    if (epoch + 1) % 50 == 0:
        print(f"    Epoch {epoch+1}/300, loss={epoch_loss/n_batches:.4f}", flush=True)

model.is_fitted = True

print("    Denoising...", flush=True)
denoised = model.denoise(noisy.numpy())
vaf_jae = calculate_vaf(clean, torch.from_numpy(denoised))
print(f"    JAE VAF: {vaf_jae:.3f}", flush=True)

# Results
print("\n" + "=" * 60, flush=True)
print("RESULTS", flush=True)
print("=" * 60, flush=True)
vaf_noisy = calculate_vaf(clean, noisy)
print(f"VAF Noisy:  {vaf_noisy:.3f}", flush=True)
print(f"VAF PCA:    {vaf_pca:.3f}", flush=True)
print(f"VAF JAE:    {vaf_jae:.3f}", flush=True)

if vaf_jae > vaf_pca:
    print(f"\n✓ SUCCESS: JAE beats PCA by {vaf_jae - vaf_pca:.3f}", flush=True)
else:
    print(f"\n✗ PCA wins by {vaf_pca - vaf_jae:.3f}", flush=True)

# Quick SNR test
print("\n" + "=" * 60, flush=True)
print("SNR SWEEP (quick)", flush=True)
print("=" * 60, flush=True)

for snr in [3, 5, 10]:
    print(f"\nTesting SNR={snr}dB...", flush=True)
    clean, noisy, _ = simulate_neural_data(n_samples=200, snr_db=snr, nonlinear=True, seed=42)
    
    pca_denoised = run_pca_baseline(noisy, latent_dim=8)
    vaf_pca = calculate_vaf(clean, pca_denoised)
    
    model = JAE(latent_dim=8, verbose=False)
    model.fit(noisy.numpy(), epochs=200, batch_size=32, verbose=False)
    denoised = model.denoise(noisy.numpy())
    vaf_jae = calculate_vaf(clean, torch.from_numpy(denoised))
    
    winner = "JAE ✓" if vaf_jae > vaf_pca else "PCA"
    print(f"  PCA={vaf_pca:.3f}, JAE={vaf_jae:.3f} -> {winner}", flush=True)

print("\nDone!", flush=True)
