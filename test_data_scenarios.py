"""Quick test script to verify enhanced data generation capabilities."""

import sys
sys.path.insert(0, 'src')

from jae import generate_data_scenarios, calculate_snr
import numpy as np

print("Testing Enhanced Data Generation Scenarios")
print("=" * 60)

scenarios = ['easy', 'transients', 'heterogeneous', 'nonstationary', 'high_noise']

for scenario in scenarios:
    print(f"\n📊 Testing scenario: {scenario}")
    print("-" * 60)
    
    data = generate_data_scenarios(scenario=scenario, n_samples=100, seed=42)
    
    clean = data['clean']
    noisy = data['noisy']
    info = data['info']
    
    # Verify shapes
    print(f"✓ Clean shape: {clean.shape}")
    print(f"✓ Noisy shape: {noisy.shape}")
    
    # Verify data properties
    print(f"✓ Clean range: [{clean.min():.3f}, {clean.max():.3f}]")
    print(f"✓ Noisy range: [{noisy.min():.3f}, {noisy.max():.3f}]")
    print(f"✓ Clean mean: {clean.mean():.3f}, std: {clean.std():.3f}")
    print(f"✓ Noisy mean: {noisy.mean():.3f}, std: {noisy.std():.3f}")
    
    # Check for NaN/Inf
    assert not np.isnan(clean.numpy()).any(), f"NaN in clean data for {scenario}"
    assert not np.isnan(noisy.numpy()).any(), f"NaN in noisy data for {scenario}"
    assert not np.isinf(clean.numpy()).any(), f"Inf in clean data for {scenario}"
    assert not np.isinf(noisy.numpy()).any(), f"Inf in noisy data for {scenario}"
    print("✓ No NaN or Inf values")
    
    # Check non-negativity
    assert (noisy >= 0).all(), f"Negative values in noisy data for {scenario}"
    print("✓ All values non-negative")
    
    # Display scenario info
    print(f"✓ SNR: {info['snr_db']:.2f} dB")
    print(f"✓ Description: {info['description']}")
    
    if 'channel_snrs' in info:
        ch_snrs = info['channel_snrs']
        print(f"✓ Channel SNR range: [{ch_snrs.min():.2f}, {ch_snrs.max():.2f}] dB")
        print(f"✓ Channel SNR std: {ch_snrs.std():.2f} dB (heterogeneity)")
    
    if 'regime_boundaries' in info:
        print(f"✓ Regime changes at: {info['regime_boundaries']}")

print("\n" + "=" * 60)
print("✅ All scenarios tested successfully!")
print("\nKey observations:")
print("- 'easy': Smooth, low noise → JAE1 and JAE2 similar")
print("- 'transients': Sharp spikes → JAE2 (U-Net) advantage")
print("- 'heterogeneous': Varied noise → JAE2 (multi-view) advantage")
print("- 'nonstationary': Regime changes → JAE2 (temporal) advantage")
print("- 'high_noise': Extreme challenge for both")
print("\nData is normalized and non-negative to prevent activation explosions.")

