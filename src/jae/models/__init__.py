"""Neural network models for the JAE package.

Exposes the two Joint Autoencoder variants and their shared building blocks:

- :class:`JAE1`: the modular, collapse-resistant channel-split autoencoder
  (a corrected reimplementation of Altan et al. 2021).
- :class:`JAE2`: a JEPA-style joint-embedding predictive model that learns the
  underlying manifold by predicting masked-region embeddings in latent space.
"""

from jae.models.encoders import ConvPatchEncoder, MLPDecoder, MLPEncoder
from jae.models.jae1 import JAE1, JAE1Output
from jae.models.jae2 import JAE2, JAE2Output

__all__ = [
    "JAE1",
    "JAE1Output",
    "JAE2",
    "JAE2Output",
    "MLPEncoder",
    "MLPDecoder",
    "ConvPatchEncoder",
]
