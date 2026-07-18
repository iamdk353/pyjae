# JAE: Joint Autoencoders for Neural Signal Denoising and Manifold Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`pyjae` (pronounced "pie-jay") provides two joint-autoencoder models for
denoising high-dimensional neural population recordings and recovering the
low-dimensional manifold underneath them:

- **JAE1** is a corrected reimplementation of the channel-split Joint Autoencoder
  from [Altan et al. (2021)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008591).
  It splits channels into disjoint partitions that share the signal but carry
  independent noise, and denoises by forcing their latents to agree.
- **JAE2** is a JEPA (Joint-Embedding Predictive Architecture): instead of a
  fixed channel split, it masks part of the input and predicts the *embedding*
  of the masked region from the visible context, learning the manifold with a
  VICReg objective that prevents latent collapse.

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

## How it works

**JAE1 (channel split).** The recorded channels are partitioned into two disjoint
sets. Both observe the same underlying low-dimensional signal, but the per-channel
noise is independent across the partitions. Each partition is encoded and decoded
by its own autoencoder, and the loss adds a term that pulls the two partitions'
latents together:

```
C = MSE(X1, X1_hat) + MSE(X2, X2_hat) + ||Z1 - Z2||^2
```

Matching independent noise realizations is not a low-cost solution, so the model
keeps only the shared, denoised structure. This is the Noise2Noise principle
applied across channel subsets, and it is why a plain autoencoder without the
split has no comparable pressure to reject channel-specific noise.

**JAE2 (JEPA).** A single shared encoder embeds both a masked "context" view and
the full input. A small predictor guesses the embeddings of the masked region
from the context, and the loss is computed in latent space (Smooth-L1), not
signal space, so the model can discard unpredictable noise instead of
reconstructing it. A VICReg variance/covariance term keeps the representation
from collapsing. A lightweight decoder head turns the encoding into a denoised
signal.

## Evaluation

Denoising quality is measured as per-channel Variance Accounted For (VAF / R^2)
against the clean ground truth, on a held-out test split, with matched latent
dimensions across methods. The results are regime-specific by design:

- On **linear** data, PCA and Factor Analysis are near-optimal and beat JAE1
  (a JAE win here would be a red flag).
- On **nonlinear** data, JAE1 beats PCA and Factor Analysis, and the margin
  grows with the strength of the nonlinearity, reproducing the paper's central
  finding.

The harness also runs negative controls (phase-shuffled and pure-noise data) to
confirm the model does not invent structure that is not there. Run it with:

```bash
uv run python scripts/benchmark.py --quick     # small, fast sweep
uv run python scripts/benchmark.py             # fuller sweep
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
| `pyjae.eval` | Non-gameable benchmark harness |

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
