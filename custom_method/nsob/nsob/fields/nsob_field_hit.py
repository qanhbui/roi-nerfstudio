from typing import Dict, Optional, Tuple, Type, Literal
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    RGBFieldHead,
)
# from nsob.fields.nsob_field_heads import HitFieldHead
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field


class NsobHitField(Field):
    """NeRF Field

    Args:
        position_encoding: Position encoder.
        direction_encoding: Direction encoder.
        base_mlp_num_layers: Number of layers for base MLP.
        base_mlp_layer_width: Width of base MLP layers.
        head_mlp_num_layers: Number of layer for output head MLP.
        head_mlp_layer_width: Width of output head MLP layers.
        skip_connections: Where to add skip connection in base MLP.
        use_integrated_encoding: Used integrated samples as encoding input.
        spatial_distortion: Spatial distortion.
    """
    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_layers: int = 8,
        layer_width: int = 256,
        spatial_distortion: Optional[SpatialDistortion] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.encoding = Identity(in_dim=3)
        self.spatial_distortion = spatial_distortion

        # self.mlp_base = MLP(
        #     in_dim=self.encoding.get_out_dim(),
        #     num_layers=num_layers,
        #     layer_width=layer_width,
        #     out_activation=nn.ReLU(),
        #     implementation=implementation,
        # )
        self.network = MLP(
            in_dim=self.encoding.get_out_dim(),
            num_layers=num_layers,
            layer_width=layer_width,
            out_dim=1,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

        self.mlp = torch.nn.Sequential(self.encoding, self.network)

        # self.field_output_density = HitFieldHead(in_dim=self.network.get_out_dim())

    def get_hit(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        density = self.network(positions_flat).view(*ray_samples.frustums.shape, -1)
        # density_before_activation = (
        #     self.mlp(positions_flat).view(*ray_samples.frustums.shape, -1).to(positions)
        # )
        return density, None
    
    # def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
    #     """Computes and returns the densities."""
    #     if self.spatial_distortion is not None:
    #         positions = ray_samples.frustums.get_positions()
    #         positions = self.spatial_distortion(positions)
    #         positions = (positions + 2.0) / 4.0
    #     else:
    #         positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
    #     # Make sure the tcnn gets inputs between 0 and 1.
    #     selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
    #     positions = positions * selector[..., None]
    #     self._sample_locations = positions
    #     if not self._sample_locations.requires_grad:
    #         self._sample_locations.requires_grad = True
    #     positions_flat = positions.view(-1, 3)
    #     h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
    #     density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
    #     self._density_before_activation = density_before_activation

    #     # Rectifying the density with an exponential is much more stable than a ReLU or
    #     # softplus, because it enables high post-activation (float32) density outputs
    #     # from smaller internal (float16) parameters.
    #     density = trunc_exp(density_before_activation.to(positions))
    #     density = density * selector[..., None]
    #     return density, base_mlp_out

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> dict:
        return {}
    
    def forward(self, ray_samples: RaySamples, compute_normals: bool = False) -> Dict:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        hit, base_mlp_out = self.get_hit(ray_samples)
        field_outputs = self.get_outputs(ray_samples, density_embedding=base_mlp_out)
        field_outputs["hit_object"] = hit  # type: ignore

        return field_outputs