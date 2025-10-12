# JAE: Joint Autoencoder for Neural Signal Denoising

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modern PyTorch implementation of Joint Autoencoders (JAE) for denoising high-dimensional neural signals, based on [Altan et al. (2021)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008722) with state-of-the-art enhancements.

## Features

- **Original JAE (JAE1)**: Faithful implementation of the 2021 paper with dual autoencoders and joint latent space alignment
- **Modernized JAE2**: Enhanced version featuring:
  - N-way parallel networks with stochastic subsampling
  - 1D U-Net architecture for temporal context awareness
  - VICReg loss for rotation/scale-invariant latent alignment
  - Robust Huber loss for reconstruction
- **Scikit-learn style API**: Simple `.fit()` and `.denoise()` methods
- **Automatic GPU detection**: Seamlessly uses CUDA when available
- **Sensible defaults**: Auto-configures parameters with helpful warnings
- **Comprehensive documentation**: Full Sphinx docs with examples

## Quick Start

```python
import numpy as np
from jae import JAEDenoiser

# Your noisy neural data: shape (n_samples, n_channels, n_timepoints)
noisy_data = np.random.randn(100, 96, 128)

# Create and train denoiser (uses JAE2 by default)
denoiser = JAEDenoiser(latent_dim=12, use_gpu=True)
denoiser.fit(noisy_data, epochs=100)

# Denoise your signals
clean_data = denoiser.denoise(noisy_data)
```

## Installation

### From PyPI (coming soon)

```bash
pip install jae
```

### From Source

```bash
git clone https://github.com/yourusername/jae.git
cd jae
pip install -e .
```

### With GPU Support

JAE automatically detects and uses CUDA-capable GPUs. For NVIDIA GPUs, ensure you have:

```bash
# Check PyTorch CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

Visit [PyTorch installation guide](https://pytorch.org/get-started/locally/) for GPU-specific setup.

## Usage Examples

### Basic Denoising

```python
from jae import JAEDenoiser

# Initialize with automatic parameter detection
denoiser = JAEDenoiser()  # Will auto-detect latent_dim from input

# Fit and denoise in one step
clean_data = denoiser.fit_denoise(noisy_data, epochs=150)
```

### Comparing JAE1 vs JAE2

```python
# Original JAE (2021)
jae1 = JAEDenoiser(model_type='jae1', latent_dim=12)
jae1.fit(noisy_data, epochs=100)
clean_jae1 = jae1.denoise(noisy_data)

# Modernized JAE2 (default)
jae2 = JAEDenoiser(model_type='jae2', latent_dim=12, num_networks=5)
jae2.fit(noisy_data, epochs=100)
clean_jae2 = jae2.denoise(noisy_data)
```

### Advanced Configuration

```python
denoiser = JAEDenoiser(
    model_type='jae2',
    latent_dim=16,
    num_networks=8,           # More parallel networks
    subsample_fraction=0.7,   # 70% neuron subsampling
    unet_channels=[64, 128],  # Deeper U-Net
    use_gpu=True,
    device='cuda:0'           # Specific GPU
)

denoiser.fit(
    noisy_data,
    epochs=200,
    batch_size=64,
    learning_rate=0.0005,
    verbose=True
)
```

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- NumPy ≥ 1.21.0
- scikit-learn ≥ 1.0.0

## Architecture Overview

### JAE1 (Original)
- Splits input channels into two fixed groups (50/50)
- Two parallel fully-connected autoencoders
- Loss: MSE(reconstruction) + MSE(latent_1, latent_2)

### JAE2 (Modernized)
- N parallel networks with random neuron subsampling
- 1D U-Net encoders with skip connections
- Loss: Huber(reconstruction) + VICReg(latents)
- Invariant to linear transformations in latent space

## Citation

If you use this package in your research, please cite the original paper:

```bibtex
@article{altan2021large,
  title={Large-scale neural recordings with single neuron resolution using Neuropixels probes},
  author={Altan, Evren and others},
  journal={PLOS Computational Biology},
  year={2021}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original JAE paper: Altan et al. (2021)
- VICReg: Bardes et al. (2022)
- U-Net architecture: Ronneberger et al. (2015)

## Support

- **Documentation**: [https://jae.readthedocs.io](https://jae.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/yourusername/jae/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/jae/discussions)

