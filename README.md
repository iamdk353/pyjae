<div align="center">

# `pyjae`

**Joint Autoencoders (JAE) for Neural Signal Denoising**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

`pyjae` (pronounced "pie-jay") is the official implementation of joint autoencoder
models for denoising high-dimensional neural population recordings and recovering
the low-dimensional manifold underneath, based on
[Altan et al. (2021)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008591).

## Installation

```bash
pip install pyjae
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add pyjae
```

For development from source:

```bash
git clone https://github.com/egealtan/pyjae.git
cd pyjae
uv sync            # installs the package with the dev extras
```

## Quick start

```python
from pyjae import JAE, simulate_neural_data

# Simulate a 6D nonlinear manifold observed on 64 noisy channels
clean, noisy, info = simulate_neural_data(
    n_samples=400, n_channels=64, n_timepoints=96,
    latent_dim=6, snr_db=10.0, nonlinear=True, alpha=3.0, seed=0,
)

# Train the channel-split model and denoise
model = JAE(latent_dim=6)          # backend="jae1" by default
model.fit(noisy, epochs=200)
denoised = model.denoise(noisy)
print("VAF:", model.score(clean, denoised))
```

Use the JEPA backend for representation learning plus a denoising readout:

```python
model = JAE(latent_dim=32, backend="jepa", patch_len=8, d_model=64)
model.fit(noisy, epochs=150)
denoised = model.denoise(noisy)
```

## Package layout

| Module | Purpose |
| --- | --- |
| `pyjae.api` | `JAE` facade (fit / denoise / score / save / load) over both backends |
| `pyjae.models` | `JAE1`, `JAE2`, and shared encoder building blocks |
| `pyjae.views` | Modular channel-split and JEPA-mask strategies |
| `pyjae.data` | Simulator (Altan et al. generative model) and evaluation controls |
| `pyjae.metrics` | Per-channel VAF plus a collapse-resistant latent-quality panel |
| `pyjae.baselines` | PCA, Factor Analysis, denoising autoencoder, Wiener oracle |
| `pyjae.eval` | Benchmark and evaluation harness |

## Requirements

- Python >= 3.10
- PyTorch >= 2.0, NumPy, scikit-learn, SciPy

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

MIT License. See [LICENSE](LICENSE).
