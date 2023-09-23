from latent_geometry.mapping.abstract import (
    DerivativeMapping,
    EuclideanMatrixMapping,
    Mapping,
    MatrixMapping,
)
from latent_geometry.mapping.identity import IdentityMapping
from latent_geometry.mapping.torch import BaseTorchModelMapping, TorchModelMapping
from latent_geometry.mapping.toy.northern_hemisphere import (
    create_northern_hemisphere_mapping,
)
from latent_geometry.mapping.toy.sphere_immersion import create_sphere_immersion
