#!/usr/bin/env python
"""
Final JAE validation script.
Tests that JAE beats PCA on simulated neural data.
"""

import sys
sys.path.insert(0, 'src')

import torch
from jae import JAE, simulate_neural_data, calculate_vaf, run_pca_baseline, set_seed

set_seed(42)

print("=" * 60)
print("JAE VALIDATION")
print("=" * 60)

# Generate test data
print("\n[1] Generating simulated neural data...")
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

print(f"    Shape: {noisy.shape}")
print(f"    SNR: {info['snr_db']:.1f} dB")
print(f"    Clean range: [{clean.min():.2f}, {clean.max():.2f}]")

# PCA baseline
print("\n[2] PCA baseline...")
pca_denoised = run_pca_baseline(noisy, latent_dim=8)
vaf_pca = calculate_vaf(clean, pca_denoised)
print(f"    VAF: {vaf_pca:.3f}")

# JAE
print("\n[3] Training JAE...")
model = JAE(latent_dim=8, verbose=False)
model.fit(noisy.numpy(), epochs=300, batch_size=32, verbose=False)

denoised = model.denoise(noisy.numpy())
vaf_jae = calculate_vaf(clean, torch.from_numpy(denoised))
print(f"    VAF: {vaf_jae:.3f}")

# Results
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
vaf_noisy = calculate_vaf(clean, noisy)
print(f"VAF Noisy:  {vaf_noisy:.3f}")
print(f"VAF PCA:    {vaf_pca:.3f}")
print(f"VAF JAE:    {vaf_jae:.3f}")
print(f"\nJAE non-negative: {(denoised >= 0).all()}")

if vaf_jae > vaf_pca:
    print(f"\n✓ SUCCESS: JAE beats PCA by {vaf_jae - vaf_pca:.3f}")
else:
    print(f"\n✗ PCA wins by {vaf_pca - vaf_jae:.3f}")


# SNR sweep
print("\n" + "=" * 60)
print("SNR SWEEP")
print("=" * 60)

results = []
for snr in [0, 3, 5, 10]:
    clean, noisy, _ = simulate_neural_data(
        n_samples=300, snr_db=snr, nonlinear=True, seed=42
    )
    
    pca_denoised = run_pca_baseline(noisy, latent_dim=8)
    vaf_pca = calculate_vaf(clean, pca_denoised)
    
    model = JAE(latent_dim=8, verbose=False)
    model.fit(noisy.numpy(), epochs=300, batch_size=32, verbose=False)
    denoised = model.denoise(noisy.numpy())
    vaf_jae = calculate_vaf(clean, torch.from_numpy(denoised))
    
    winner = "JAE" if vaf_jae > vaf_pca else "PCA"
    print(f"SNR={snr:>2}dB: PCA={vaf_pca:.3f}, JAE={vaf_jae:.3f} -> {winner}")
    results.append((snr, vaf_pca, vaf_jae))


# Summary
jae_wins = sum(1 for _, pca, jae in results if jae > pca)
print(f"\nJAE wins: {jae_wins}/{len(results)} SNR levels")
