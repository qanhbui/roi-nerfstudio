import torch
from jaxtyping import Float, Bool
from torch import Tensor

from typing import Callable, Dict, Literal, Optional, Tuple, Union, overload

from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples

class NsobFrustums(Frustums):
    
    def get_centers_on_ray(self) -> Float[Tensor, "*batch 1"]:
        return (self.starts + self.ends) / 2


class NsobRaySamples(RaySamples):
    """Samples along a ray for nsob"""

    frustums: NsobFrustums
    # # No need init
    # def __init__(self, num_rays_per_batch: int, keep_full_image: bool = False, **kwargs) -> None:
    #     super().__init__(num_rays_per_batch, keep_full_image, **kwargs)

    def get_weights(self, densities: Float[Tensor, "*batch num_samples 1"], occlusion_mask: Optional[Bool[Tensor, "*batch num_samples 1"]] = None) -> Float[Tensor, "*batch num_samples 1"]:
        """Return weights based on predicted densities

        Args:
            densities: Predicted densities for samples along ray

        Returns:
            Weights for each sample
        """

        delta_density = self.deltas * densities
        alphas = 1 - torch.exp(-delta_density)

        transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
        transmittance = torch.cat(
            [torch.zeros((*transmittance.shape[:1], 1, 1), device=densities.device), transmittance], dim=-2
        )
        transmittance = torch.exp(-transmittance)  # [..., "num_samples"]
        if occlusion_mask is not None:
            alphas[occlusion_mask] = 0
        weights = alphas * transmittance  # [..., "num_samples"]
        weights = torch.nan_to_num(weights)

        return weights


class NsobRayBundle(RayBundle):
    """A bundle of ray parameters for nsob."""

    def get_ray_samples(
        self,
        bin_starts: Float[Tensor, "*bs num_samples 1"],
        bin_ends: Float[Tensor, "*bs num_samples 1"],
        spacing_starts: Optional[Float[Tensor, "*bs num_samples 1"]] = None,
        spacing_ends: Optional[Float[Tensor, "*bs num_samples 1"]] = None,
        spacing_to_euclidean_fn: Optional[Callable] = None,
    ) -> NsobRaySamples:
        """Produces samples for each ray by projection points along the ray direction. Currently samples uniformly.

        Args:
            bin_starts: Distance from origin to start of bin.
            bin_ends: Distance from origin to end of bin.

        Returns:
            Samples projected along ray.
        """
        deltas = bin_ends - bin_starts
        if self.camera_indices is not None:
            camera_indices = self.camera_indices[..., None]
        else:
            camera_indices = None

        shaped_raybundle_fields = self[..., None]

        frustums = NsobFrustums(
            origins=shaped_raybundle_fields.origins,  # [..., 1, 3]
            directions=shaped_raybundle_fields.directions,  # [..., 1, 3]
            starts=bin_starts,  # [..., num_samples, 1]
            ends=bin_ends,  # [..., num_samples, 1]
            pixel_area=shaped_raybundle_fields.pixel_area,  # [..., 1, 1]
        )

        ray_samples = NsobRaySamples(
            frustums=frustums,
            camera_indices=camera_indices,  # [..., 1, 1]
            deltas=deltas,  # [..., num_samples, 1]
            spacing_starts=spacing_starts,  # [..., num_samples, 1]
            spacing_ends=spacing_ends,  # [..., num_samples, 1]
            spacing_to_euclidean_fn=spacing_to_euclidean_fn,
            metadata=shaped_raybundle_fields.metadata,
            times=None if self.times is None else self.times[..., None],  # [..., 1, 1]
        )

        return ray_samples