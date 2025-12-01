# JAE: Joint Autoencoder for Neural Signal Denoising

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch implementation of the Joint Autoencoder (JAE) for denoising high-dimensional neural signals, based on [Altan et al. (2021)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008591).

## Installation

```bash
git clone https://github.com/yourusername/jae.git
cd jae
pip install -e .
```

## Quick Start

```python
from jae import JAE, simulate_neural_data, calculate_vaf, run_pca_baseline

# Generate synthetic neural data
clean, noisy, info = simulate_neural_data(
    n_samples=300,
    n_channels=96,
    latent_dim=6,
    snr_db=5.0,
    nonlinear=True
)

# Train JAE
model = JAE(latent_dim=8)
model.fit(noisy.numpy(), epochs=300)

# Denoise and evaluate
denoised = model.denoise(noisy.numpy())

# Compare to PCA baseline
pca_denoised = run_pca_baseline(noisy, latent_dim=8)

vaf_pca = calculate_vaf(clean, pca_denoised)
vaf_jae = calculate_vaf(clean, denoised)

print(f"PCA VAF: {vaf_pca:.3f}")
print(f"JAE VAF: {vaf_jae:.3f}")
```

## How It Works

JAE splits neural recordings into two random partitions and trains parallel autoencoders. Both partitions share the same underlying signal but have independent noise. By forcing the latent representations to match, JAE learns to extract signal and reject noise.

Loss function (Eq 3 from paper):
```
C = MSE(X1, X̂1) + MSE(X2, X̂2) + λ||Z1 - Z2||²
```

## Configuration

```python
model = JAE(
    latent_dim=8,        # Latent space dimensionality
    dropout_p=0.05,      # Input dropout (paper default)
    learning_rate=0.001, # ADAM learning rate
    use_gpu=True         # Auto-detects CUDA
)
```

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- NumPy ≥ 1.21.0
- scikit-learn ≥ 1.0.0
- scipy

## Citation

```bibtex
@article{altan2021jae,
  title={Estimating the dimensionality of the manifold underlying multi-electrode neural recordings},
  author={Altan, Ege and Solla, Sara A. and Miller, Lee E. and Perreault, Eric J.},
  journal={PLOS Computational Biology},
  year={2021},
  volume={17},
  number={11},
  pages={e1008591},
  doi={10.1371/journal.pcbi.1008591}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
