# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Some ray datastructures.
"""
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Literal, Optional, Tuple, Union, overload

import torch
from jaxtyping import Float, Int, Shaped
from torch import Tensor

from nerfstudio.utils.math import Gaussians, conical_frustum_to_gaussian
from nerfstudio.utils.tensor_dataclass import TensorDataclass

TORCH_DEVICE = Union[str, torch.device]


@dataclass
class Frustums(TensorDataclass):
    """Describes region of space as a frustum."""

    origins: Float[Tensor, "*bs 3"]
    """xyz coordinate for ray origin."""
    directions: Float[Tensor, "*bs 3"]
    """Direction of ray."""
    starts: Float[Tensor, "*bs 1"]
    """Where the frustum starts along a ray."""
    ends: Float[Tensor, "*bs 1"]
    """Where the frustum ends along a ray."""
    pixel_area: Float[Tensor, "*bs 1"]
    """Projected area of pixel a distance 1 away from origin."""
    offsets: Optional[Float[Tensor, "*bs 3"]] = None
    """Offsets for each sample position"""

    def get_positions(self) -> Float[Tensor, "*batch 3"]:
        """Calculates "center" position of frustum. Not weighted by mass.

        Returns:
            xyz positions.
        """
        pos = self.origins + self.directions * (self.starts + self.ends) / 2
        if self.offsets is not None:
            pos = pos + self.offsets
        return pos

    def get_start_positions(self) -> Float[Tensor, "*batch 3"]:
        """Calculates "start" position of frustum.

        Returns:
            xyz positions.
        """
        return self.origins + self.directions * self.starts

    def set_offsets(self, offsets):
        """Sets offsets for this frustum for computing positions"""
        self.offsets = offsets

    def get_gaussian_blob(self) -> Gaussians:
        """Calculates guassian approximation of conical frustum.

        Returns:
            Conical frustums approximated by gaussian distribution.
        """
        # Cone radius is set such that the square pixel_area matches the cone area.
        cone_radius = torch.sqrt(self.pixel_area) / 1.7724538509055159  # r = sqrt(pixel_area / pi)
        if self.offsets is not None:
            raise NotImplementedError()
        return conical_frustum_to_gaussian(
            origins=self.origins,
            directions=self.directions,
            starts=self.starts,
            ends=self.ends,
            radius=cone_radius,
        )

    @classmethod
    def get_mock_frustum(cls, device: Optional[TORCH_DEVICE] = "cpu") -> "Frustums":
        """Helper function to generate a placeholder frustum.

        Returns:
            A size 1 frustum with meaningless values.
        """
        return Frustums(
            origins=torch.ones((1, 3)).to(device),
            directions=torch.ones((1, 3)).to(device),
            starts=torch.ones((1, 1)).to(device),
            ends=torch.ones((1, 1)).to(device),
            pixel_area=torch.ones((1, 1)).to(device),
        )


@dataclass
class RaySamples(TensorDataclass):
    """Samples along a ray"""

    frustums: Frustums
    """Frustums along ray."""
    camera_indices: Optional[Int[Tensor, "*bs 1"]] = None
    """Camera index."""
    deltas: Optional[Float[Tensor, "*bs 1"]] = None
    """"width" of each sample."""
    spacing_starts: Optional[Float[Tensor, "*bs num_samples 1"]] = None
    """Start of normalized bin edges along ray [0,1], before warping is applied, ie. linear in disparity sampling."""
    spacing_ends: Optional[Float[Tensor, "*bs num_samples 1"]] = None
    """Start of normalized bin edges along ray [0,1], before warping is applied, ie. linear in disparity sampling."""
    spacing_to_euclidean_fn: Optional[Callable] = None
    """Function to convert bins to euclidean distance."""
    metadata: Optional[Dict[str, Shaped[Tensor, "*bs latent_dims"]]] = None
    """additional information relevant to generating ray samples"""

    times: Optional[Float[Tensor, "*batch 1"]] = None
    """Times at which rays are sampled"""

    def get_weights(self, densities: Float[Tensor, "*batch num_samples 1"]) -> Float[Tensor, "*batch num_samples 1"]:
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

        weights = alphas * transmittance  # [..., "num_samples"]
        weights = torch.nan_to_num(weights)

        return weights

    @overload
    @staticmethod
    def get_weights_and_transmittance_from_alphas(
        alphas: Float[Tensor, "*batch num_samples 1"], weights_only: Literal[True]
    ) -> Float[Tensor, "*batch num_samples 1"]:
        ...

    @overload
    @staticmethod
    def get_weights_and_transmittance_from_alphas(
        alphas: Float[Tensor, "*batch num_samples 1"], weights_only: Literal[False] = False
    ) -> Tuple[Float[Tensor, "*batch num_samples 1"], Float[Tensor, "*batch num_samples 1"]]:
        ...

    @staticmethod
    def get_weights_and_transmittance_from_alphas(
        alphas: Float[Tensor, "*batch num_samples 1"], weights_only: bool = False
    ) -> Union[
        Float[Tensor, "*batch num_samples 1"],
        Tuple[Float[Tensor, "*batch num_samples 1"], Float[Tensor, "*batch num_samples 1"]],
    ]:
        """Return weights based on predicted alphas
        Args:
            alphas: Predicted alphas (maybe from sdf) for samples along ray
            weights_only: If function should return only weights
        Returns:
            Tuple of weights and transmittance for each sample
        """

        transmittance = torch.cumprod(
            torch.cat([torch.ones((*alphas.shape[:1], 1, 1), device=alphas.device), 1.0 - alphas + 1e-7], 1), 1
        )

        weights = alphas * transmittance[:, :-1, :]
        if weights_only:
            return weights
        return weights, transmittance
    
def match_ray_samples_with_selector(
        ray_samples_1: RaySamples, 
        ray_samples_2: RaySamples, 
        selector: Tensor):
        """Match ray samples 2 to the reference ray samples 1 with indices from selectors
        Args:
            ray_samples_1 : ray_samples reference
            ray_samples_2 : ray_samples need to expand to mactch reference
        """
        
        starts = ray_samples_1.spacing_starts[..., 0] # num, 48
        starts[selector] = ray_samples_2.spacing_starts[..., 0] 

        # bins
        ends = ray_samples_1.spacing_ends[..., -1:, 0]
        ends[selector] = ray_samples_2.spacing_ends[..., -1:, 0]
        bins = torch.cat([starts, ends], dim=-1)
        # Stop gradients
        bins = bins.detach()

        euclidean_bins = ray_samples_2.spacing_to_euclidean_fn(bins)
        
        bin_starts = euclidean_bins[..., :-1, None]
        bin_ends = euclidean_bins[..., 1:, None]
        spacing_starts = bins[..., :-1, None]
        spacing_ends = bins[..., 1:, None]
        spacing_to_euclidean_fn = ray_samples_2.spacing_to_euclidean_fn

        deltas = bin_ends - bin_starts
        camera_indices = ray_samples_1.camera_indices
        camera_indices[selector] = ray_samples_2.camera_indices

        origins = ray_samples_1.frustums.origins
        origins[selector] = ray_samples_2.frustums.origins
        directions = ray_samples_1.frustums.directions
        directions[selector] = ray_samples_2.frustums.directions
        pixel_area = ray_samples_1.frustums.pixel_area
        pixel_area[selector] = ray_samples_2.frustums.pixel_area
        
        metadata = ray_samples_1.metadata.copy()
        metadata_2 = ray_samples_2.metadata
        for key in metadata.keys():
            metadata[key][selector] = metadata_2[key]

        if ray_samples_1.times and ray_samples_2.times:
            times = ray_samples_1.times
            times[selector] = ray_samples_2.times
        else:
            times = None

        frustums = Frustums(
            origins=origins,
            directions=directions,
            starts=bin_starts,
            ends=bin_ends,
            pixel_area=pixel_area,
        )

        matched_ray_samples = RaySamples(
            frustums=frustums,
            camera_indices=camera_indices,
            deltas=deltas,
            spacing_starts=spacing_starts,
            spacing_ends=spacing_ends,
            spacing_to_euclidean_fn=spacing_to_euclidean_fn,
            metadata=metadata,
            times=times
        )

        return matched_ray_samples
    
def merge_different_ray_samples_with_indices(
        ray_samples_1: RaySamples, 
        ray_samples_2: RaySamples, 
        inside_selector_1: Tensor, 
        inside_selector_2: Tensor,
        inside_ray_indices: Tensor,
        scale_transform: Float):
        """Merge two set of different ray samples with indices from selectors, apply transform and return sorted index which can be used to merge field outputs
        Args:
            ray_samples_1 : ray_samples to merge
            ray_samples_2 : ray_samples to merge
        """
        inside_ray_samples_1 = ray_samples_1[inside_ray_indices]
        inside_ray_samples_2 = ray_samples_2[inside_ray_indices]

        inside_rays_selector_1 = inside_selector_1[inside_ray_indices]
        inside_rays_selector_2 = inside_selector_2[inside_ray_indices]

        assert inside_ray_samples_1.spacing_starts is not None and inside_ray_samples_2.spacing_starts is not None
        assert inside_ray_samples_1.spacing_ends is not None and inside_ray_samples_2.spacing_ends is not None
        assert inside_ray_samples_1.spacing_to_euclidean_fn is not None
        
        starts_1 = inside_ray_samples_1.spacing_starts[..., 0] # num, 48
        starts_2 = inside_ray_samples_2.spacing_starts[..., 0] * scale_transform

        min_starts_1 = starts_1[..., 0].unsqueeze(-1).expand(-1, starts_1.size(-1))

        # bins
        starts_1[inside_rays_selector_1] = min_starts_1[inside_rays_selector_1]
        starts_2[~inside_rays_selector_2] = min_starts_1[~inside_rays_selector_2]
        bins, sorted_index = torch.sort(torch.cat([starts_1, starts_2], -1), -1)

        ends = inside_ray_samples_1.spacing_ends[..., -1:, 0]
        bins = torch.cat([bins, ends], dim=-1)
        # Stop gradients
        bins = bins.detach()

        euclidean_bins = inside_ray_samples_1.spacing_to_euclidean_fn(bins)
        
        bin_starts = euclidean_bins[..., :-1, None]
        bin_ends = euclidean_bins[..., 1:, None]
        spacing_starts = bins[..., :-1, None]
        spacing_ends = bins[..., 1:, None]
        spacing_to_euclidean_fn = inside_ray_samples_1.spacing_to_euclidean_fn

        deltas = bin_ends - bin_starts
        camera_indices = inside_ray_samples_1.camera_indices[..., 0, None, :].repeat(1, spacing_starts.size(1), 1) # num, 48, 1 -> num, 1, 1 -> num, 48*2, 1 

        origins = inside_ray_samples_1.frustums.origins[..., 0, None, :].repeat(1, spacing_starts.size(1), 1)
        directions = inside_ray_samples_1.frustums.directions[..., 0, None, :].repeat(1, spacing_starts.size(1), 1)
        pixel_area = inside_ray_samples_1.frustums.pixel_area[..., 0, None, :].repeat(1, spacing_starts.size(1), 1)
        
        metadata = inside_ray_samples_1.metadata.copy()
        for key in metadata.keys():
            metadata[key] = metadata[key][..., 0, None, :].repeat(1, spacing_starts.size(1), 1)

        times = None if inside_ray_samples_1.times is None else inside_ray_samples_1.times[..., 0, None, :].repeat(1, spacing_starts.size(1), 1)

        frustums = Frustums(
            origins=origins,
            directions=directions,
            starts=bin_starts,
            ends=bin_ends,
            pixel_area=pixel_area,
        )

        merged_ray_samples = RaySamples(
            frustums=frustums,
            camera_indices=camera_indices,
            deltas=deltas,
            spacing_starts=spacing_starts,
            spacing_ends=spacing_ends,
            spacing_to_euclidean_fn=spacing_to_euclidean_fn,
            metadata=metadata,
            times=times
        )

        return merged_ray_samples, sorted_index

def merge_spacing_bins(  # bins spacing
        ray_samples_1: RaySamples, 
        ray_samples_2: RaySamples, 
        merged_spacing_bins: Optional[Tensor], 
        selector_1: Tensor, 
        selector_2: Tensor,
        merged_selector: Optional[Tensor], 
        scale_transform: Float):
        """Merge two set of different ray samples with indices from selectors, apply transform and return sorted index which can be used to merge field outputs
        Args:
            ray_samples_1 : ray_samples to merge
            ray_samples_2 : ray_samples to merge
            merged_spacing_bins : ofray samples [0,1]
        """
        assert ray_samples_1.spacing_starts is not None and ray_samples_2.spacing_starts is not None
        assert ray_samples_1.spacing_ends is not None and ray_samples_2.spacing_ends is not None
        assert ray_samples_1.spacing_to_euclidean_fn is not None

        if torch.is_tensor(merged_spacing_bins) and torch.is_tensor(merged_selector):
            selector = merged_selector
            starts_1 = merged_spacing_bins[..., :-1] # num, 48
            ends = merged_spacing_bins[..., -1:] # num, 1
        else:
            selector = selector_1
            starts_1 = ray_samples_1.spacing_starts[..., 0] # num, 48, 1 -> num, 48
            ends = ray_samples_1.spacing_ends[..., -1:, 0]
        
        starts_2 = ray_samples_2.spacing_starts[..., 0] * scale_transform

        min_starts = starts_1[..., 0]
        min_starts_1 = min_starts.unsqueeze(-1).expand_as(starts_1)
        min_starts_2 = min_starts.unsqueeze(-1).expand_as(starts_2)

        # starts_1[~selector] = min_starts_1[~selector]
        # starts_2[~selector_2] = min_starts_2[~selector_2]
        starts_1 = (starts_1 * selector) + ((~selector) * min_starts_1)
        starts_2 = (starts_2 * selector_2) + ((~selector_2) * min_starts_2)

        bins, sorted_index = torch.sort(torch.cat([starts_1, starts_2], -1), -1)

        bins = torch.cat([bins, ends], dim=-1) # num, 48+1
        return bins, sorted_index

def get_rays_from_spacing_bins(
        ray_samples: RaySamples,
        spacing_bins: Tensor):
        """Set new ray samples from bins.
        Args:
            ray_samples : ray_samples reference
            spacing_bins : spacing_bins [0,1] to build new ray_samples
        """
        # Stop gradients
        spacing_bins = spacing_bins.detach()

        euclidean_bins = ray_samples.spacing_to_euclidean_fn(spacing_bins)
        
        bin_starts = euclidean_bins[..., :-1, None]
        bin_ends = euclidean_bins[..., 1:, None]
        spacing_starts = spacing_bins[..., :-1, None] # num, 48 -> num, 48, 1 
        spacing_ends = spacing_bins[..., 1:, None]
        spacing_to_euclidean_fn = ray_samples.spacing_to_euclidean_fn

        deltas = bin_ends - bin_starts
        num_samples = spacing_starts.size(1)
        camera_indices = ray_samples.camera_indices[..., 0, None, :].repeat(1,num_samples, 1)

        origins = ray_samples.frustums.origins[..., 0, None, :].repeat(1, num_samples, 1)
        directions = ray_samples.frustums.directions[..., 0, None, :].repeat(1, num_samples, 1)
        pixel_area = ray_samples.frustums.pixel_area[..., 0, None, :].repeat(1, num_samples, 1)
        
        metadata = ray_samples.metadata.copy()
        for key in metadata.keys():
            metadata[key] = metadata[key][..., 0, None, :].repeat(1, num_samples, 1)

        times = None if ray_samples.times is None else ray_samples.times[..., 0, None, :].repeat(1, num_samples, 1)

        frustums = Frustums(
            origins=origins,
            directions=directions,
            starts=bin_starts,
            ends=bin_ends,
            pixel_area=pixel_area,
        )

        final_ray_samples = RaySamples(
            frustums=frustums,
            camera_indices=camera_indices,
            deltas=deltas,
            spacing_starts=spacing_starts,
            spacing_ends=spacing_ends,
            spacing_to_euclidean_fn=spacing_to_euclidean_fn,
            metadata=metadata,
            times=times
        )

        return final_ray_samples

def get_rays_samples_with_indices(ray_samples: RaySamples, ray_indices: Int[Tensor, "*num_indices"]):
    """Get rays samples using indices
    Args:
        ray_samples : ray_samples
        indices : ray indices 
    """
    indiced_frustum = ray_samples.frustums[ray_indices]
    indiced_camera_indices = ray_samples.camera_indices[ray_indices]
    indiced_deltas = ray_samples.deltas[ray_indices]
    indiced_spacing_starts = ray_samples.spacing_starts[ray_indices]
    indiced_spacing_ends = ray_samples.spacing_ends[ray_indices]
    indiced_spacing_to_euclidean_fn = ray_samples.spacing_to_euclidean_fn
    indiced_metadata = ray_samples.metadata.copy()
    for key in indiced_metadata.keys():
        indiced_metadata[key] = indiced_metadata[key][ray_indices]

    indiced_times = None if ray_samples.times is None else ray_samples.times[ray_indices]

    indiced_ray_samples = RaySamples(
        frustums=indiced_frustum,
        camera_indices=indiced_camera_indices,
        deltas=indiced_deltas,
        spacing_starts=indiced_spacing_starts,
        spacing_ends=indiced_spacing_ends,
        spacing_to_euclidean_fn=indiced_spacing_to_euclidean_fn,
        metadata=indiced_metadata,
        times=indiced_times,
    )
    return indiced_ray_samples

def get_samples_inside_box(ray_samples: RaySamples, aabb: Float[Tensor, "2 3"]):
    """Get points inside object box
    Args:
        ray_samples : ray_samples
        aabb : object aabb that contain min and max 3D points
    """

    positions = ray_samples.frustums.get_positions()
    min_point = aabb[0]
    max_point = aabb[1]
    inside_selector = ((positions[..., 0] >= min_point[0].item()) & (positions[..., 0] <= max_point[0].item())
                       & (positions[..., 1] >= min_point[1].item()) & (positions[..., 1] <= max_point[1].item())
                       & (positions[..., 2] >= min_point[2].item()) & (positions[..., 2] <= max_point[2].item()))
    return inside_selector

def get_rays_inside_box(ray_depths: Float[Tensor, "*num_rays 1"], aabb: Float[Tensor, "2 3"]):
    """Get points inside object box
    Args:
        ray_samples : ray_samples
        aabb : object aabb that contain min and max 3D points
    """

    positions = ray_depths
    min_point = aabb[0]
    max_point = aabb[1]
    inside_selector = ((positions[..., 0] >= min_point[0].item()) & (positions[..., 0] <= max_point[0].item())
                       & (positions[..., 1] >= min_point[1].item()) & (positions[..., 1] <= max_point[1].item())
                       & (positions[..., 2] >= min_point[2].item()) & (positions[..., 2] <= max_point[2].item()))
    return inside_selector

def get_samples_selector(ray_samples: RaySamples, t_min: Float[Tensor, "*num_rays"], t_max: Float[Tensor, "*num_rays"], inside: bool = True):
    """Get points inside object box
    Args:
        ray_samples : ray_samples
        aabb : object aabb that contain min and max 3D points
    """
    assert ray_samples.spacing_starts is not None 
    assert ray_samples.spacing_ends is not None

    # starts = ray_samples.spacing_starts[..., 0] # num, 48, 1 -> num, 48 : [0,1]48
    starts = ray_samples.frustums.starts[..., 0] # num, 48, 1 -> num, 48 : [0,20]48
    # ends = ray_samples.spacing_ends[..., -1:, 0]
    # bins = torch.cat([starts, ends], dim=-1)

    # ends = ray_samples.spacing_ends[..., 0]
    ends = ray_samples.frustums.ends[..., 0]
    centers = (starts + ends) / 2

    t_min = t_min.unsqueeze(-1).expand_as(centers) # num -> num, 48
    t_max = t_max.unsqueeze(-1).expand_as(centers)

    # selector = (bins >= t_min) & (bins <= t_max)
    selector = (centers >= t_min) & (centers <= t_max) # num, 48
    if inside:
        return selector
    else:
        return ~selector
    
def get_bins_selector(spacing_bins: Tensor, ray_samples: RaySamples, t_min: Float[Tensor, "*num_rays"], t_max: Float[Tensor, "*num_rays"], inside: bool = True):
    """Get bins inside object box
    Args:
        spacing_bins : spacing_bins
        ray_samples : ray_samples reference
    """
    # starts = ray_samples.spacing_starts[..., 0]
    
    # Transform spacing_bins to bins of frustum before compare
    bins = ray_samples.spacing_to_euclidean_fn(spacing_bins)

    starts = bins[..., :-1]

    # ends = ray_samples.spacing_ends[..., 0]
    ends = bins[..., 1:]
    centers = (starts + ends) / 2

    t_min = t_min.unsqueeze(-1).expand_as(centers)
    t_max = t_max.unsqueeze(-1).expand_as(centers)

    # selector = (bins >= t_min) & (bins <= t_max)
    selector = (centers >= t_min) & (centers <= t_max) # num, 48
    if inside:
        return selector
    else:
        return ~selector

def get_4D_points(ray_samples: RaySamples):
    """Get 4D points from 3D points of ray samples
    Args:
        ray_samples : ray_samples
    """
    
    positions_flat = ray_samples.frustums.get_positions().view(-1, 3) # 1572864, 3
    output_positions = torch.cat(
        (
            positions_flat, # n, 3 >< n, 3, 4
            torch.tensor([[1]], dtype=positions_flat.dtype, device=positions_flat.device).repeat_interleave(len(positions_flat), 0),
        ),
        1,
    ) # n, 4

    return output_positions

def transform_ray_samples(ray_samples: RaySamples, scene_scale: Float, inv_scene_transform: Float[Tensor, "4 4"], roi_scale: Float, roi_transform: Float[Tensor, "4 4"]):
    """Transform 3D points of ray_samples from A to B
    Args:
        ray_samples : ray_samples
    """
    
    positions_flat = ray_samples.frustums.get_positions().view(-1, 3) # 1572864, 3
    output_positions = torch.cat(
        (
            positions_flat, # n, 3 >< n, 3, 4
            torch.tensor([[1]], dtype=positions_flat.dtype, device=positions_flat.device).repeat_interleave(len(positions_flat), 0),
        ),
        1,
    ) # n, 4

    output_positions[..., :3] /= scene_scale # n, 4
    output_positions = inv_scene_transform @ (output_positions.T) # 4, 4 @ 4, n = 4, n
    output_positions = roi_transform @ output_positions # 4, 4 @ 4, n = 4, n
    output_positions = output_positions.T # n, 4
    output_positions[..., :3] *= roi_scale
    output_positions = output_positions[:, :3]

    return output_positions

def transform_points(positions: Float[Tensor, "*batch 3"], scene_scale: Float, inv_scene_transform: Float[Tensor, "4 4"], roi_scale: Float, roi_transform: Float[Tensor, "4 4"]):
    """Transform 3D points of ray_samples from A to B
    Args:
        ray_samples : ray_samples
    """
    output_positions = torch.cat(
        (
            positions, # n, 3 >< n, 3, 4
            torch.tensor([[1]], dtype=positions.dtype, device=positions.device).repeat_interleave(len(positions), 0),
        ),
        1,
    ) # n, 4

    output_positions[..., :3] /= scene_scale # n, 4
    output_positions = inv_scene_transform @ (output_positions.T) # 4, 4 @ 4, n = 4, n
    output_positions = roi_transform @ output_positions # 4, 4 @ 4, n = 4, n
    output_positions = output_positions.T # n, 4
    output_positions[..., :3] *= roi_scale
    output_positions = output_positions[:, :3]

    return output_positions

def transform_single_point(position: Float[Tensor, "3"], scene_scale: Float, inv_scene_transform: Float[Tensor, "4 4"], roi_scale: Float, roi_transform: Float[Tensor, "4 4"]):
    """Transform 3D points from A to B
    Args:
    
    """
    
    position = position.unsqueeze(0) # 1572864, 3
    output_positions = torch.cat(
        (
            position, # n, 3 >< n, 3, 4
            torch.tensor([[1]], dtype=position.dtype, device=position.device),
        ),
        1,
    ) # n, 4

    output_positions[..., :3] /= scene_scale # n, 4
    output_positions = inv_scene_transform @ (output_positions.T) # 4, 4 @ 4, n = 4, n
    output_positions = roi_transform @ output_positions # 4, 4 @ 4, n = 4, n
    output_positions = output_positions.T # n, 4
    output_positions[..., :3] *= roi_scale
    output_positions = output_positions[:, :3]

    return output_positions

@dataclass
class RayBundle(TensorDataclass):
    """A bundle of ray parameters."""

    # TODO(ethan): make sure the sizes with ... are correct
    origins: Float[Tensor, "*batch 3"]
    """Ray origins (XYZ)"""
    directions: Float[Tensor, "*batch 3"]
    """Unit ray direction vector"""
    pixel_area: Float[Tensor, "*batch 1"]
    """Projected area of pixel a distance 1 away from origin"""
    camera_indices: Optional[Int[Tensor, "*batch 1"]] = None
    """Camera indices"""
    nears: Optional[Float[Tensor, "*batch 1"]] = None
    """Distance along ray to start sampling"""
    fars: Optional[Float[Tensor, "*batch 1"]] = None
    """Rays Distance along ray to stop sampling"""
    metadata: Dict[str, Shaped[Tensor, "num_rays latent_dims"]] = field(default_factory=dict)
    """Additional metadata or data needed for interpolation, will mimic shape of rays"""
    times: Optional[Float[Tensor, "*batch 1"]] = None
    """Times at which rays are sampled"""
    height: Optional[Int] = None
    """Original image height"""
    width: Optional[Int] = None
    """Original image width"""

    def set_camera_indices(self, camera_index: int) -> None:
        """Sets all the camera indices to a specific camera index.

        Args:
            camera_index: Camera index.
        """
        self.camera_indices = torch.ones_like(self.origins[..., 0:1]).long() * camera_index

    def __len__(self) -> int:
        num_rays = torch.numel(self.origins) // self.origins.shape[-1]
        return num_rays

    def sample(self, num_rays: int) -> "RayBundle":
        """Returns a RayBundle as a subset of rays.

        Args:
            num_rays: Number of rays in output RayBundle

        Returns:
            RayBundle with subset of rays.
        """
        assert num_rays <= len(self)
        indices = random.sample(range(len(self)), k=num_rays)
        return self[indices]

    def get_row_major_sliced_ray_bundle(self, start_idx: int, end_idx: int) -> "RayBundle":
        """Flattens RayBundle and extracts chunk given start and end indices.

        Args:
            start_idx: Start index of RayBundle chunk.
            end_idx: End index of RayBundle chunk.

        Returns:
            Flattened RayBundle with end_idx-start_idx rays.

        """
        return self.flatten()[start_idx:end_idx]

    def get_ray_samples(
        self,
        bin_starts: Float[Tensor, "*bs num_samples 1"],
        bin_ends: Float[Tensor, "*bs num_samples 1"],
        spacing_starts: Optional[Float[Tensor, "*bs num_samples 1"]] = None,
        spacing_ends: Optional[Float[Tensor, "*bs num_samples 1"]] = None,
        spacing_to_euclidean_fn: Optional[Callable] = None,
    ) -> RaySamples:
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

        frustums = Frustums(
            origins=shaped_raybundle_fields.origins,  # [..., 1, 3]
            directions=shaped_raybundle_fields.directions,  # [..., 1, 3]
            starts=bin_starts,  # [..., num_samples, 1]
            ends=bin_ends,  # [..., num_samples, 1]
            pixel_area=shaped_raybundle_fields.pixel_area,  # [..., 1, 1]
        )

        ray_samples = RaySamples(
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
