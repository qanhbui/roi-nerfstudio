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
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type, Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import time
import math

from nerfstudio.cameras.rays import (
    RayBundle, RaySamples, Frustums, 
    get_samples_inside_box, 
    get_4D_points, 
    transform_ray_samples, 
    transform_points, 
    transform_single_point, 
    get_rays_inside_box,
    get_samples_selector,
    merge_different_ray_samples_with_indices,
    match_ray_samples_with_selector,
    get_bins_selector,
    merge_spacing_bins,
    get_rays_from_spacing_bins
)
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler, NeuSSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider, AABBBoxCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.rich_utils import CONSOLE
import nerfstudio.utils.math
from nerfstudio.data.scene_box import SceneBox

@dataclass
class NerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    """Whether to randomize the background color."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 16
    """Resolution of the base grid for the hashgrid."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 2
    """How many hashgrid features per level"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""
    appearance_embed_dim: int = 32
    """Dimension of the appearance embedding."""
    use_AABB_collider: bool = False
    """Whether to use AABBBoxCollider or else NearFarCollider."""
    define_N_max: bool = True
    """define N_max for NeRF train."""


class NerfactoModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
            self.scene_box = self.cam_box
            # extend_factor = 1.0
            # contraction_box = self.scene_box.aabb.clone()
            # center_box = (contraction_box[0] + contraction_box[1]) / 2.0
            # half_dims = (contraction_box[1] - contraction_box[0]) / 2.0
            # extend_half_dims = half_dims * extend_factor
            # contraction_box[0] = center_box - extend_half_dims
            # contraction_box[1] = center_box + extend_half_dims
            # self.scene_box = SceneBox(aabb=contraction_box)
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        if self.N_max is not None and self.config.define_N_max:
            self.config.max_res = self.N_max

        # Fields
        self.field = NerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_embed_dim,
            implementation=self.config.implementation,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def scene_samples_selector( # Get selector for scene samples
            self, 
            scene_ray_samples: RaySamples,
            dataparser_outputs: DataparserOutputs,
            object_ray_samples_list: List[RaySamples] = None, # small object ray samples
            object_models_list: List[Model] = None,
            object_dataparser_outputs_list: List[DataparserOutputs] = None,
            scene_rois_aabb_tmin_list: List[Tensor] = None,  # full
            scene_rois_aabb_tmax_list: List[Tensor] = None, 
            rois_selectors_list: List[Tensor] = None # full
    ):
        # init list
        compo_roi_indices = []
        multi_full_roi_scene_selector = torch.zeros((scene_ray_samples.shape), dtype=torch.bool, device=self.device)
        full_roi_scene_selector_list = []
        full_roi_scene_ray_selector_list = []
        roi_scene_samples_list = []

        for index, roi_ray_selector in enumerate(rois_selectors_list):

            full_roi_scene_selector = torch.zeros((scene_ray_samples.shape), dtype=torch.bool, device=self.device)
            roi_scene_ray_samples = scene_ray_samples[roi_ray_selector] # roi aabb size
            roi_scene_selector = get_samples_selector(roi_scene_ray_samples, scene_rois_aabb_tmin_list[index][roi_ray_selector], scene_rois_aabb_tmax_list[index][roi_ray_selector])


            if roi_scene_selector.sum().item() > 0:
                compo_roi_indices.append(index)
                full_roi_scene_selector[roi_ray_selector] = roi_scene_selector
                full_roi_scene_ray_selector = full_roi_scene_selector.any(dim=1)

                multi_full_roi_scene_selector |= full_roi_scene_selector # full size
                roi_scene_samples = roi_scene_ray_samples[roi_scene_selector]

                full_roi_scene_selector_list.append(full_roi_scene_selector)
                full_roi_scene_ray_selector_list.append(full_roi_scene_ray_selector)
                roi_scene_samples_list.append(roi_scene_samples)

        ####NOTE : Compo with intersection attention
        """
        Put these 2 blocks before return in comment to have a compo overlap
        """
        # Check intersection region and output ordered rois
        intersection_rays_selectors_list = []
        intersection_samples_selectors_list = []
        if len(compo_roi_indices) > 1:
            object_indices_order = []
            relative_object_indices_order = []
            for i in range(len(compo_roi_indices)):
                for j in range(i + 1, len(compo_roi_indices)):
                    intersection_samples_selector = full_roi_scene_selector_list[i] & full_roi_scene_selector_list[j] # full size
                    intersection_rays_selector = intersection_samples_selector.any(dim=1)
                    if intersection_rays_selector.sum().item() > 0:
                        # i
                        t_min_i = scene_rois_aabb_tmin_list[compo_roi_indices[i]][intersection_rays_selector] # t wasn't cropped by intersection aabb
                        t_max_i = scene_rois_aabb_tmax_list[compo_roi_indices[i]][intersection_rays_selector]  
                        t_center_i = (t_min_i + t_max_i) / 2
                        # j
                        t_min_j = scene_rois_aabb_tmin_list[compo_roi_indices[j]][intersection_rays_selector] # t wasn't cropped by intersection aabb
                        t_max_j = scene_rois_aabb_tmax_list[compo_roi_indices[j]][intersection_rays_selector] 
                        t_center_j = (t_min_j + t_max_j) / 2
                        
                        if torch.mean(t_center_i).item() < torch.mean(t_center_j).item() :
                            object_indices_order.append([compo_roi_indices[i], compo_roi_indices[j]])
                            relative_object_indices_order.append([i, j])
                        else:
                            object_indices_order.append([compo_roi_indices[j], compo_roi_indices[i]])
                            relative_object_indices_order.append([j, i])
                        intersection_rays_selectors_list.append(intersection_rays_selector) # full size
                        intersection_samples_selectors_list.append(intersection_samples_selector)

        # NOTE: intersection process through samples:
        if intersection_samples_selectors_list:
            # iterate through intersection regions: 
            for index, intersection_samples_selector in enumerate(intersection_samples_selectors_list): # full size
                # NOTE: Exclude intersection of second rois intersect
                full_roi_scene_selector_list[relative_object_indices_order[index][1]][intersection_samples_selector] = False
                full_roi_scene_ray_selector_list[relative_object_indices_order[index][1]] = full_roi_scene_selector_list[relative_object_indices_order[index][1]].any(dim=1)

                temp_roi_scene_samples = scene_ray_samples[full_roi_scene_selector_list[relative_object_indices_order[index][1]]]
                roi_scene_samples_list[relative_object_indices_order[index][1]] = temp_roi_scene_samples

        return compo_roi_indices, roi_scene_samples_list, full_roi_scene_selector_list, full_roi_scene_ray_selector_list, multi_full_roi_scene_selector
            
    def compo_v0( # field_outputs of rois using object model from roi_selector on scene ray samples, no object samples
            self, 
            scene_ray_samples: RaySamples,
            dataparser_outputs: DataparserOutputs,
            object_ray_samples_list: List[RaySamples] = None, # small object rays samples
            object_models_list: List[Model] = None,
            object_dataparser_outputs_list: List[DataparserOutputs] = None,
            scene_rois_aabb_tmin_list: List[Tensor] = None, # full
            scene_rois_aabb_tmax_list: List[Tensor] = None, 
            rois_selectors_list: List[Tensor] = None
    ):
        # Prepare and define variables
        scene_scale = dataparser_outputs.dataparser_scale
        scene_transform = dataparser_outputs.dataparser_transform # 3, 4

        inv_scene_transform = torch.linalg.inv(
            torch.cat(
                (
                    scene_transform,
                    torch.tensor([[0, 0, 0, 1]], dtype=scene_transform.dtype),
                ),
                0,
            )
        ).to(self.device)

        # NOTE: Get infos from objects: indices, rays and samples selector
        compo_roi_indices, roi_scene_samples_list, full_roi_scene_selector_list, full_roi_scene_ray_selector_list, multi_full_roi_scene_selector = self.scene_samples_selector(
            scene_ray_samples,
            dataparser_outputs,
            object_ray_samples_list, # small object rays samples
            object_models_list,
            object_dataparser_outputs_list,
            scene_rois_aabb_tmin_list, # full, # num: 32768
            scene_rois_aabb_tmax_list,
            rois_selectors_list
        )

        compo_selector = torch.zeros((scene_ray_samples.shape[0], 1), device=self.device, dtype=torch.bool)
        compo_scene_rgb = torch.zeros((scene_ray_samples.shape[0], 3), device=self.device)

        if compo_roi_indices:
            # NOTE: Get infos from scene: rays and samples selector
            scene_sample_selector = ~multi_full_roi_scene_selector
            scene_ray_selector = scene_sample_selector.any(dim=1)

            full_field_outputs = {}
            full_field_outputs[FieldHeadNames.DENSITY] = torch.zeros((*scene_ray_samples.shape, 1), device=self.device) # num, 48, 1 # scene compo size
            full_field_outputs[FieldHeadNames.RGB] = torch.zeros((*scene_ray_samples.shape, 3), device=self.device) # num, 48, 3

            # NOTE: Get field_outputs from scene model for just scene samples, others will be 0, wait for info from object mode:
            # if scene_ray_selector.sum().item() > 0:
            scene_rays = scene_ray_samples[scene_ray_selector] # full size flat
            scene_samples = scene_ray_samples[scene_sample_selector] # full size flat
            
            scene_samples_positions = scene_samples.frustums.get_positions()#.view(-1, 3) # num, 48, 3
            scene_samples_directions = scene_samples.frustums.directions#.reshape(-1, 3) # num, 48, 3
            scene_samples_camera_indices = scene_samples.camera_indices#.reshape(-1, 1) # num, 48, 1

            scene_selected = {}
            scene_selected["positions_flat"] = scene_samples_positions # n, 3
            scene_selected["directions_flat"] = scene_samples_directions
            scene_selected["camera_indices_flat"] = scene_samples_camera_indices

            field_outputs = self.field.forward(scene_rays, compute_normals=self.config.predict_normals, selected_infos=scene_selected)

            full_field_outputs[FieldHeadNames.DENSITY][scene_sample_selector] = field_outputs[FieldHeadNames.DENSITY]
            full_field_outputs[FieldHeadNames.RGB][scene_sample_selector] = field_outputs[FieldHeadNames.RGB]

            # NOTE: Get infos from rois : rays and samples selector
            for index, roi_indice in enumerate(compo_roi_indices):
                # NOTE: get object selectors full and check if still exist object rays after intersections
                full_roi_scene_ray_selector = full_roi_scene_ray_selector_list[index] # full size
                if full_roi_scene_ray_selector.sum().item() > 0:

                    # get roi selectors
                    full_roi_scene_selector = full_roi_scene_selector_list[index]

                    # Get infos
                    object_model = object_models_list[roi_indice]

                    roi_scale = object_dataparser_outputs_list[roi_indice].dataparser_scale
                    roi_transform = object_dataparser_outputs_list[roi_indice].dataparser_transform # 3, 4
                    roi_transform = torch.cat(
                        (
                            roi_transform,
                            torch.tensor([[0, 0, 0, 1]], dtype=roi_transform.dtype),
                        ),
                        0,
                    ).to(self.device)

                    # NOTE: Get field_outputs from scene model for just scene samples, others will be 0, wait for info from object mode:
                    roi_rays = scene_ray_samples[full_roi_scene_ray_selector] # full size flat
                    roi_samples = scene_ray_samples[full_roi_scene_selector] # full size flat
                    
                    roi_samples_positions = roi_samples.frustums.get_positions()#.view(-1, 3) # num, 48, 3
                    roi_samples_directions = roi_samples.frustums.directions#.reshape(-1, 3) # num, 48, 3
                    roi_samples_camera_indices = roi_samples.camera_indices#.reshape(-1, 1) # num, 48, 1

                    transformed_roi_positions = transform_ray_samples(roi_samples, scene_scale, inv_scene_transform, roi_scale, roi_transform)

                    roi_selected = {}
                    roi_selected["positions_flat"] = transformed_roi_positions # n, 3
                    roi_selected["directions_flat"] = roi_samples_directions
                    roi_selected["camera_indices_flat"] = roi_samples_camera_indices

                    field_outputs = object_model.field.forward(roi_rays, compute_normals=self.config.predict_normals, selected_infos=roi_selected)

                    full_field_outputs[FieldHeadNames.DENSITY][full_roi_scene_selector] = field_outputs[FieldHeadNames.DENSITY]
                    full_field_outputs[FieldHeadNames.RGB][full_roi_scene_selector] = field_outputs[FieldHeadNames.RGB]

            weights = scene_ray_samples.get_weights(full_field_outputs[FieldHeadNames.DENSITY])

            rgb = self.renderer_rgb(rgb=full_field_outputs[FieldHeadNames.RGB], weights=weights)
            depth = self.renderer_depth(weights=weights, ray_samples=scene_ray_samples)
            accumulation = self.renderer_accumulation(weights=weights)

        else:
            ray_samples = scene_ray_samples
            field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
            field_outputs[FieldHeadNames.DENSITY] = field_outputs[FieldHeadNames.DENSITY].view(*ray_samples.shape, -1) 
            field_outputs[FieldHeadNames.RGB] = field_outputs[FieldHeadNames.RGB].view(*ray_samples.shape, -1)

            if self.config.use_gradient_scaling:
                field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

            weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

            rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
            accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "compo_selector": compo_selector,
            "compo_scene_rgb": compo_scene_rgb,
        }
        return outputs

    def roi_samples_selector( # Processing roi for small size, compov1
            self, 
            scene_ray_samples: RaySamples,
            dataparser_outputs: DataparserOutputs,
            object_ray_samples_list: List[RaySamples] = None, # small object ray samples
            object_model_list: List[Model] = None,
            object_dataparser_outputs_list: List[DataparserOutputs] = None,
            scene_rois_aabb_tmin_list: List[Tensor] = None,  # full
            scene_rois_aabb_tmax_list: List[Tensor] = None, 
            rois_selectors_list: List[Tensor] = None # full
    ):
        # Roi NeRF model
        # init list
        scene_scale = dataparser_outputs.dataparser_scale
        compo_roi_indices = []

        roi_samples_selectors_list = []
        roi_rays_selectors_list = [] 
        full_roi_rays_selector_list = [] 
        
        multi_full_roi_rays_selector = torch.zeros((scene_ray_samples.shape[0]), dtype=torch.bool, device=self.device)
        # iter objects
        for index, object_ray_samples in enumerate(object_ray_samples_list): # box size
            
            object_scale = object_dataparser_outputs_list[index].dataparser_scale
            roi_selector = rois_selectors_list[index] # full size

            object_t_min = scene_rois_aabb_tmin_list[index][roi_selector] * (object_scale/scene_scale) # roi box size
            object_t_max = scene_rois_aabb_tmax_list[index][roi_selector] * (object_scale/scene_scale)
            
            roi_samples_selector = get_samples_selector(object_ray_samples, object_t_min, object_t_max) # get samples inside aabb, inside size
            roi_rays_selector = roi_samples_selector.any(dim=1)

            if roi_rays_selector.sum().item() > 0:
                compo_roi_indices.append(index)
                
                roi_samples_selectors_list.append(roi_samples_selector) # inside size, old_name: object_samples_selectors_list
                roi_rays_selectors_list.append(roi_rays_selector) # inside size , old_name: object_depth_rays_selectors_list

                full_roi_rays_selector = roi_selector.clone()
                full_roi_rays_selector[roi_selector] = roi_rays_selector # rays inside aabb and depth insdide aabb, full size
                full_roi_rays_selector_list.append(full_roi_rays_selector) # old_name: full_object_depth_rays_selectors_list
                
                # compo multiples object regions to full size image 
                multi_full_roi_rays_selector |= full_roi_rays_selector # full size ray, old name: compo_rays_selector

        # Scene NeRF model
        multi_full_roi_scene_selector = torch.zeros((scene_ray_samples.shape), dtype=torch.bool, device=self.device)
        # iter objects
        for index, roi_indice in enumerate(compo_roi_indices): # box size
            roi_ray_selector = full_roi_rays_selector_list[index] # full size
            roi_scene_ray_samples = scene_ray_samples[roi_ray_selector] # inside  size

            scene_object_tmin = scene_rois_aabb_tmin_list[roi_indice][roi_ray_selector] # inside size
            scene_object_tmax = scene_rois_aabb_tmax_list[roi_indice][roi_ray_selector]
            roi_scene_selector = get_samples_selector(roi_scene_ray_samples, scene_object_tmin, scene_object_tmax)
            
            multi_full_roi_scene_selector[roi_ray_selector] |= roi_scene_selector # full size

        return compo_roi_indices, roi_samples_selectors_list, roi_rays_selectors_list, full_roi_rays_selector_list, multi_full_roi_rays_selector, multi_full_roi_scene_selector
        
    def compo_v1( # field_outputs of rois using object model from roi_selector on rois ray samples.
            # Work only on 1 single ROI composition, because the complex without ray depth filtering 
            # Just work on rays inside box, merge rays, put other bins to min bin, merge field_outputs tensor, put other (densitys, rgbs ) = 0.
            self, 
            scene_ray_samples: RaySamples,
            dataparser_outputs: DataparserOutputs,
            object_ray_samples_list: List[RaySamples] = None, # small object rays samples
            object_model_list: List[Model] = None,
            object_dataparser_outputs_list: List[DataparserOutputs] = None,
            scene_objects_tmin_list: List[Tensor] = None, # full
            scene_objects_tmax_list: List[Tensor] = None, 
            objects_selectors_list: List[Tensor] = None
    ):

        # NOTE: Get infos from objects: indices, rays and samples selector
        compo_roi_indices, roi_samples_selectors_list, roi_rays_selectors_list, full_roi_rays_selector_list, multi_full_roi_rays_selector, multi_full_roi_scene_selector = self.roi_samples_selector(
            scene_ray_samples,
            dataparser_outputs,
            object_ray_samples_list, # small object rays samples
            object_model_list,
            object_dataparser_outputs_list,
            scene_objects_tmin_list, # full, # num: 32768
            scene_objects_tmax_list,
            objects_selectors_list
        )
        compo_selector = torch.zeros((scene_ray_samples.shape[0], 1), device=self.device, dtype=torch.bool)
        compo_scene_rgb = torch.zeros((scene_ray_samples.shape[0], 3), device=self.device)

        # check if object need to compo
        if compo_roi_indices:
            #NOTE: initialize pixel infos tensor of image
            rgb = torch.zeros((scene_ray_samples.shape[0], 3), device=self.device)
            depth = torch.zeros((scene_ray_samples.shape[0], 1), device=self.device)
            accumulation = torch.zeros((scene_ray_samples.shape[0], 1), device=self.device)

            scene_scale = dataparser_outputs.dataparser_scale

            # NOTE: Get field_outputs from scene model for just compo rays, others will be 0, just in compo_rays_selector:
            scene_compo_ray_samples = scene_ray_samples[multi_full_roi_rays_selector] # compo size/ inside size
            scene_compo_samples_selector = ~(multi_full_roi_scene_selector[multi_full_roi_rays_selector]) # compo size
            
            scene_compo_samples = scene_compo_ray_samples[scene_compo_samples_selector]

            scene_compo_positions = scene_compo_samples.frustums.get_positions()#.view(-1, 3) # num, 48, 3
            scene_compo_directions = scene_compo_samples.frustums.directions#.reshape(-1, 3) # num, 48, 3
            scene_compo_camera_indices = scene_compo_samples.camera_indices#.reshape(-1, 1) # num, 48, 1

            scene_selected = {}
            scene_selected["positions_flat"] = scene_compo_positions # n, 3
            scene_selected["directions_flat"] = scene_compo_directions
            scene_selected["camera_indices_flat"] = scene_compo_camera_indices

            scene_field_outputs = {}
            scene_field_outputs[FieldHeadNames.DENSITY] = torch.zeros((*scene_compo_ray_samples.shape, 1), device=self.device) # num, 48, 1 # scene compo size
            scene_field_outputs[FieldHeadNames.RGB] = torch.zeros((*scene_compo_ray_samples.shape, 3), device=self.device) # num, 48, 3

            field_outputs = self.field.forward(scene_compo_ray_samples, compute_normals=self.config.predict_normals, selected_infos=scene_selected)

            scene_field_outputs[FieldHeadNames.DENSITY][scene_compo_samples_selector] = field_outputs[FieldHeadNames.DENSITY]
            scene_field_outputs[FieldHeadNames.RGB][scene_compo_samples_selector] = field_outputs[FieldHeadNames.RGB]

            # NOTE: Get field_outputs from objects models note intersect
            for index, object_indice in enumerate(compo_roi_indices):
                # NOTE: get object selectors full and check if still exist object rays after intersections
                full_roi_rays_selector = full_roi_rays_selector_list[index] # full size

                # get object selectors at multiple size of rays and samples
                roi_rays_selector = roi_rays_selectors_list[index] # box size -> to crop to compo size 
                roi_samples_selector = roi_samples_selectors_list[index] # NOTE: samples selectors at compo size, already exclusive interection if exist

                # Get infos
                object_scale = object_dataparser_outputs_list[object_indice].dataparser_scale
                object_model = object_model_list[object_indice]
                object_ray_samples = object_ray_samples_list[object_indice]  # box size

                # NOTE: Get scene ray samples infos 
                # Get scene samples and infos within current roi
                obcompo_scene_ray_samples = scene_ray_samples[full_roi_rays_selector]  # Crop scene rays full to compo/inside size
                obcompo_scene_samples_selector = ~(multi_full_roi_scene_selector[full_roi_rays_selector]) # obcompo size

                obcompo_scene_field_outputs = {}
                obcompo_scene_field_outputs[FieldHeadNames.DENSITY] = scene_field_outputs[FieldHeadNames.DENSITY][full_roi_rays_selector[multi_full_roi_rays_selector]] # obcompo size
                obcompo_scene_field_outputs[FieldHeadNames.RGB] = scene_field_outputs[FieldHeadNames.RGB][full_roi_rays_selector[multi_full_roi_rays_selector]]

                # NOTE: Get object ray samples infos 
                # Get object samples and infos within current roi
                compo_object_ray_samples = object_ray_samples[roi_rays_selector]  # Crop object rays with inside filter of box size, get compo/inside size
                compo_roi_samples_selector = roi_samples_selector[roi_rays_selector]  # Crop object rays with inside filter of box size, get compo/inside size

                # NOTE: merge rays here
                # Do merged bin of scene and object here
                merged_spacing_bins, sorted_index = merge_spacing_bins(
                    obcompo_scene_ray_samples, 
                    compo_object_ray_samples, 
                    None, 
                    obcompo_scene_samples_selector, 
                    compo_roi_samples_selector, 
                    None,
                    (scene_scale/object_scale)
                )

                merged_ray_samples = get_rays_from_spacing_bins(obcompo_scene_ray_samples, merged_spacing_bins)

                object_samples = compo_object_ray_samples[compo_roi_samples_selector]
                object_samples_positions = object_samples.frustums.get_positions() # n, 3
                object_samples_directions = object_samples.frustums.directions # n, 3
                object_samples_camera_indices = object_samples.camera_indices # n, 1

                object_selected = {}
                object_selected["positions_flat"] = object_samples_positions # n, 3
                object_selected["directions_flat"] = object_samples_directions
                object_selected["camera_indices_flat"] = object_samples_camera_indices

                compo_object_field_outputs = object_model.field.forward(compo_object_ray_samples, compute_normals=object_model.config.predict_normals, selected_infos=object_selected)

                object_field_outputs = {}
                object_field_outputs[FieldHeadNames.DENSITY] = torch.zeros((*compo_object_ray_samples.shape, 1), device=self.device) # num, 48, 1 # object compo size
                object_field_outputs[FieldHeadNames.RGB] = torch.zeros((*compo_object_ray_samples.shape, 3), device=self.device) # num, 48, 3
                
                object_field_outputs[FieldHeadNames.DENSITY][compo_roi_samples_selector] = compo_object_field_outputs[FieldHeadNames.DENSITY]
                object_field_outputs[FieldHeadNames.RGB][compo_roi_samples_selector] = compo_object_field_outputs[FieldHeadNames.RGB]

                # # NOTE: merge field_outputs here
                merged_field_outputs = {}
                merged_field_outputs[FieldHeadNames.DENSITY] = torch.cat([obcompo_scene_field_outputs[FieldHeadNames.DENSITY], object_field_outputs[FieldHeadNames.DENSITY]], dim=1)
                merged_field_outputs[FieldHeadNames.RGB] = torch.cat([obcompo_scene_field_outputs[FieldHeadNames.RGB], object_field_outputs[FieldHeadNames.RGB]], dim=1)

                expanded_density_sorted_index = sorted_index.unsqueeze(-1)
                expanded_rgb_sorted_index = sorted_index.unsqueeze(-1).expand(-1, -1, 3)

                merged_field_outputs[FieldHeadNames.DENSITY] = torch.gather(merged_field_outputs[FieldHeadNames.DENSITY], 1, expanded_density_sorted_index)
                merged_field_outputs[FieldHeadNames.RGB] = torch.gather(merged_field_outputs[FieldHeadNames.RGB], 1, expanded_rgb_sorted_index)

                # NOTE: render pixels of image and store to tensors
                if self.config.use_gradient_scaling:
                    merged_field_outputs = scale_gradients_by_distance_squared(merged_field_outputs, merged_ray_samples)

                merged_weights = merged_ray_samples.get_weights(merged_field_outputs[FieldHeadNames.DENSITY])

                merged_rgb = self.renderer_rgb(rgb=merged_field_outputs[FieldHeadNames.RGB], weights=merged_weights)
                merged_depth = self.renderer_depth(weights=merged_weights, ray_samples=merged_ray_samples)
                merged_accumulation = self.renderer_accumulation(weights=merged_weights)

                rgb[full_roi_rays_selector] = merged_rgb
                depth[full_roi_rays_selector] = merged_depth
                accumulation[full_roi_rays_selector] = merged_accumulation

            # NOTE: Render outside scence rays part 
            # NOTE: Get field_outputs from scene model for just outside rays

            if (~multi_full_roi_rays_selector).sum().item() > 0:
                scene_outside_ray_samples = scene_ray_samples[~multi_full_roi_rays_selector] # outside size
                field_outputs = self.field.forward(scene_outside_ray_samples, compute_normals=self.config.predict_normals)

                field_outputs[FieldHeadNames.DENSITY] = field_outputs[FieldHeadNames.DENSITY].view(*scene_outside_ray_samples.shape, -1)
                field_outputs[FieldHeadNames.RGB] = field_outputs[FieldHeadNames.RGB].view(*scene_outside_ray_samples.shape, -1)

                #NOTE: render pixels of image and stack to tensors
                if self.config.use_gradient_scaling:
                    field_outputs = scale_gradients_by_distance_squared(field_outputs, scene_outside_ray_samples)

                outside_weights = scene_outside_ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

                outside_rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=outside_weights)
                outside_depth = self.renderer_depth(weights=outside_weights, ray_samples=scene_outside_ray_samples)
                outside_accumulation = self.renderer_accumulation(weights=outside_weights)

                rgb[~multi_full_roi_rays_selector] = outside_rgb
                depth[~multi_full_roi_rays_selector] = outside_depth
                accumulation[~multi_full_roi_rays_selector] = outside_accumulation
   
        else:
            ray_samples = scene_ray_samples
            field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
            field_outputs[FieldHeadNames.DENSITY] = field_outputs[FieldHeadNames.DENSITY].view(*ray_samples.shape, -1) 
            field_outputs[FieldHeadNames.RGB] = field_outputs[FieldHeadNames.RGB].view(*ray_samples.shape, -1)

            if self.config.use_gradient_scaling:
                field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

            weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

            rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
            accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "compo_selector": compo_selector,
            "compo_scene_rgb": compo_scene_rgb,
        }
        return outputs

    def roi_processing_v2( # Get selector for scene samples, depth filter, compov2
            self, 
            scene_ray_samples: RaySamples,
            dataparser_outputs: DataparserOutputs,
            object_ray_samples_list: List[RaySamples] = None, # small object ray samples
            object_models_list: List[Model] = None,
            object_dataparser_outputs_list: List[DataparserOutputs] = None,
            scene_rois_aabb_tmin_list: List[Tensor] = None,  # full
            scene_rois_aabb_tmax_list: List[Tensor] = None, 
            rois_selectors_list: List[Tensor] = None # full
    ):
        # init list
        compo_roi_indices = []
        multi_full_roi_scene_selector = torch.zeros((scene_ray_samples.shape), dtype=torch.bool, device=self.device)
        full_roi_scene_selector_list = []
        full_roi_scene_ray_selector_list = []
        roi_scene_samples_list = []

        for index, roi_ray_selector in enumerate(rois_selectors_list):
            full_roi_scene_selector = torch.zeros((scene_ray_samples.shape), dtype=torch.bool, device=self.device)
            roi_scene_ray_samples = scene_ray_samples[roi_ray_selector] # roi aabb size
            
            #NOTE: depth, density calculation of inside box rays with scene model
            roi_density_flat, roi_density_embedding_flat = self.field.get_density(roi_scene_ray_samples) # roi aabb/box size

            roi_density = roi_density_flat.view(*roi_scene_ray_samples.shape, -1) # num_rays, 48, 1  # small size
            roi_density_embedding = roi_density_embedding_flat.view(*roi_scene_ray_samples.shape, -1) # num_rays, 48, 15
            
            #NOTE: compute depth pixels/rays
            roi_weights = roi_scene_ray_samples.get_weights(roi_density)
            roi_rays_depth = self.renderer_depth(weights=roi_weights, ray_samples=roi_scene_ray_samples) 
            
            roi_rays_t_depth = roi_rays_depth.squeeze()
            roi_t_min = scene_rois_aabb_tmin_list[index][roi_ray_selector] # roi aabb/box size
            roi_t_max = scene_rois_aabb_tmax_list[index][roi_ray_selector]
            roi_depth_rays_selector = (roi_rays_t_depth >= roi_t_min) & (roi_rays_t_depth <= roi_t_max) # rays box size

            roi_scene_ray_samples = roi_scene_ray_samples[roi_depth_rays_selector]
            
            temp = roi_ray_selector.clone()
            temp[roi_ray_selector] = roi_depth_rays_selector
            roi_ray_selector = temp
            
            roi_scene_selector = get_samples_selector(roi_scene_ray_samples, roi_t_min[roi_depth_rays_selector], roi_t_max[roi_depth_rays_selector])

            if roi_scene_selector.sum().item() > 0:
                compo_roi_indices.append(index)
                full_roi_scene_selector[roi_ray_selector] = roi_scene_selector
                full_roi_scene_ray_selector = full_roi_scene_selector.any(dim=1)

                multi_full_roi_scene_selector |= full_roi_scene_selector # full size
                roi_scene_samples = roi_scene_ray_samples[roi_scene_selector]

                full_roi_scene_selector_list.append(full_roi_scene_selector)
                full_roi_scene_ray_selector_list.append(full_roi_scene_ray_selector)
                roi_scene_samples_list.append(roi_scene_samples)

        ####NOTE : Compo with intersection attention
        """
        Put these 2 blocks before return in comment to have a compo overlap
        """
        # Check intersection region and output ordered rois
        intersection_rays_selectors_list = []
        intersection_samples_selectors_list = []
        if len(compo_roi_indices) > 1:
            object_indices_order = []
            relative_object_indices_order = []
            for i in range(len(compo_roi_indices)):
                for j in range(i + 1, len(compo_roi_indices)):
                    intersection_samples_selector = full_roi_scene_selector_list[i] & full_roi_scene_selector_list[j] # full size
                    intersection_rays_selector = intersection_samples_selector.any(dim=1)
                    if intersection_rays_selector.sum().item() > 0:
                        # i
                        t_min_i = scene_rois_aabb_tmin_list[compo_roi_indices[i]][intersection_rays_selector] # t wasn't cropped by intersection aabb
                        t_max_i = scene_rois_aabb_tmax_list[compo_roi_indices[i]][intersection_rays_selector]  
                        t_center_i = (t_min_i + t_max_i) / 2
                        # j
                        t_min_j = scene_rois_aabb_tmin_list[compo_roi_indices[j]][intersection_rays_selector] # t wasn't cropped by intersection aabb
                        t_max_j = scene_rois_aabb_tmax_list[compo_roi_indices[j]][intersection_rays_selector] 
                        t_center_j = (t_min_j + t_max_j) / 2
                        
                        # TODO: compare and get object order
                        if torch.mean(t_center_i).item() < torch.mean(t_center_j).item() :
                            object_indices_order.append([compo_roi_indices[i], compo_roi_indices[j]])
                            relative_object_indices_order.append([i, j])
                        else:
                            object_indices_order.append([compo_roi_indices[j], compo_roi_indices[i]])
                            relative_object_indices_order.append([j, i])
                        intersection_rays_selectors_list.append(intersection_rays_selector) # full size
                        intersection_samples_selectors_list.append(intersection_samples_selector)

        # NOTE: intersection process through samples:
        if intersection_samples_selectors_list:
            # iterate through intersection regions: 
            for index, intersection_samples_selector in enumerate(intersection_samples_selectors_list): # full size
                # NOTE: Exclude intersection of second rois intersect
                full_roi_scene_selector_list[relative_object_indices_order[index][1]][intersection_samples_selector] = False
                full_roi_scene_ray_selector_list[relative_object_indices_order[index][1]] = full_roi_scene_selector_list[relative_object_indices_order[index][1]].any(dim=1)

                temp_roi_scene_samples = scene_ray_samples[full_roi_scene_selector_list[relative_object_indices_order[index][1]]]
                roi_scene_samples_list[relative_object_indices_order[index][1]] = temp_roi_scene_samples

        return compo_roi_indices, roi_scene_samples_list, full_roi_scene_selector_list, full_roi_scene_ray_selector_list, multi_full_roi_scene_selector
            
    def compo_v2( # field_outputs of rois using object model from roi_selector on scene ray samples, no object samples
            self, 
            scene_ray_samples: RaySamples,
            dataparser_outputs: DataparserOutputs,
            object_ray_samples_list: List[RaySamples] = None, # small object rays samples
            object_models_list: List[Model] = None,
            object_dataparser_outputs_list: List[DataparserOutputs] = None,
            scene_rois_aabb_tmin_list: List[Tensor] = None, # full
            scene_rois_aabb_tmax_list: List[Tensor] = None, 
            rois_selectors_list: List[Tensor] = None
    ):
        # Prepare and define variables
        scene_scale = dataparser_outputs.dataparser_scale
        scene_transform = dataparser_outputs.dataparser_transform # 3, 4

        inv_scene_transform = torch.linalg.inv(
            torch.cat(
                (
                    scene_transform,
                    torch.tensor([[0, 0, 0, 1]], dtype=scene_transform.dtype),
                ),
                0,
            )
        ).to(self.device)

        # NOTE: Get infos from objects: indices, rays and samples selector
        compo_roi_indices, roi_scene_samples_list, full_roi_scene_selector_list, full_roi_scene_ray_selector_list, multi_full_roi_scene_selector = self.roi_processing_v2(
            scene_ray_samples,
            dataparser_outputs,
            object_ray_samples_list, # small object rays samples
            object_models_list,
            object_dataparser_outputs_list,
            scene_rois_aabb_tmin_list, # full, # num: 32768
            scene_rois_aabb_tmax_list,
            rois_selectors_list
        )

        compo_selector = torch.zeros((scene_ray_samples.shape[0], 1), device=self.device, dtype=torch.bool)
        compo_scene_rgb = torch.zeros((scene_ray_samples.shape[0], 3), device=self.device)

        if compo_roi_indices:
            # NOTE: Get infos from scene: rays and samples selector
            scene_sample_selector = ~multi_full_roi_scene_selector
            scene_ray_selector = scene_sample_selector.any(dim=1)

            full_field_outputs = {}
            full_field_outputs[FieldHeadNames.DENSITY] = torch.zeros((*scene_ray_samples.shape, 1), device=self.device) # num, 48, 1 # scene compo size
            full_field_outputs[FieldHeadNames.RGB] = torch.zeros((*scene_ray_samples.shape, 3), device=self.device) # num, 48, 3

            if scene_ray_selector.sum().item() > 0:
                # NOTE: Get field_outputs from scene model for just scene samples, others will be 0, wait for info from object mode:
                scene_rays = scene_ray_samples[scene_ray_selector] # full size flat
                scene_samples = scene_ray_samples[scene_sample_selector] # full size flat
                
                scene_samples_positions = scene_samples.frustums.get_positions()#.view(-1, 3) # num, 48, 3
                scene_samples_directions = scene_samples.frustums.directions#.reshape(-1, 3) # num, 48, 3
                scene_samples_camera_indices = scene_samples.camera_indices#.reshape(-1, 1) # num, 48, 1

                scene_selected = {}
                scene_selected["positions_flat"] = scene_samples_positions # n, 3
                scene_selected["directions_flat"] = scene_samples_directions
                scene_selected["camera_indices_flat"] = scene_samples_camera_indices

                field_outputs = self.field.forward(scene_rays, compute_normals=self.config.predict_normals, selected_infos=scene_selected)

                full_field_outputs[FieldHeadNames.DENSITY][scene_sample_selector] = field_outputs[FieldHeadNames.DENSITY]
                full_field_outputs[FieldHeadNames.RGB][scene_sample_selector] = field_outputs[FieldHeadNames.RGB]

            # NOTE: Get infos from rois : rays and samples selector
            for index, roi_indice in enumerate(compo_roi_indices):
                # NOTE: get object selectors full and check if still exist object rays after intersections
                full_roi_scene_ray_selector = full_roi_scene_ray_selector_list[index] # full size
                if full_roi_scene_ray_selector.sum().item() > 0:

                    # get roi selectors
                    full_roi_scene_selector = full_roi_scene_selector_list[index]

                    # Get infos
                    object_model = object_models_list[roi_indice]

                    roi_scale = object_dataparser_outputs_list[roi_indice].dataparser_scale
                    roi_transform = object_dataparser_outputs_list[roi_indice].dataparser_transform # 3, 4
                    roi_transform = torch.cat(
                        (
                            roi_transform,
                            torch.tensor([[0, 0, 0, 1]], dtype=roi_transform.dtype),
                        ),
                        0,
                    ).to(self.device)

                    # NOTE: Get field_outputs from scene model for just scene samples, others will be 0, wait for info from object mode:
                    roi_rays = scene_ray_samples[full_roi_scene_ray_selector] # full size flat
                    roi_samples = scene_ray_samples[full_roi_scene_selector] # full size flat
                    
                    roi_samples_positions = roi_samples.frustums.get_positions()#.view(-1, 3) # num, 48, 3
                    roi_samples_directions = roi_samples.frustums.directions#.reshape(-1, 3) # num, 48, 3
                    roi_samples_camera_indices = roi_samples.camera_indices#.reshape(-1, 1) # num, 48, 1

                    transformed_roi_positions = transform_ray_samples(roi_samples, scene_scale, inv_scene_transform, roi_scale, roi_transform)

                    roi_selected = {}
                    roi_selected["positions_flat"] = transformed_roi_positions # n, 3
                    roi_selected["directions_flat"] = roi_samples_directions
                    roi_selected["camera_indices_flat"] = roi_samples_camera_indices

                    field_outputs = object_model.field.forward(roi_rays, compute_normals=self.config.predict_normals, selected_infos=roi_selected)

                    full_field_outputs[FieldHeadNames.DENSITY][full_roi_scene_selector] = field_outputs[FieldHeadNames.DENSITY]
                    full_field_outputs[FieldHeadNames.RGB][full_roi_scene_selector] = field_outputs[FieldHeadNames.RGB]

            weights = scene_ray_samples.get_weights(full_field_outputs[FieldHeadNames.DENSITY])

            rgb = self.renderer_rgb(rgb=full_field_outputs[FieldHeadNames.RGB], weights=weights)
            depth = self.renderer_depth(weights=weights, ray_samples=scene_ray_samples)
            accumulation = self.renderer_accumulation(weights=weights)

        else:
            ray_samples = scene_ray_samples
            field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
            field_outputs[FieldHeadNames.DENSITY] = field_outputs[FieldHeadNames.DENSITY].view(*ray_samples.shape, -1) 
            field_outputs[FieldHeadNames.RGB] = field_outputs[FieldHeadNames.RGB].view(*ray_samples.shape, -1)

            if self.config.use_gradient_scaling:
                field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

            weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

            rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
            accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "compo_selector": compo_selector,
            "compo_scene_rgb": compo_scene_rgb,
        }
        return outputs

    def objects_processing_small( # Processing objects for small size, multiobv6
            # TODO: order
            self, 
            scene_ray_samples: RaySamples,
            dataparser_outputs: DataparserOutputs,
            object_ray_samples_list: List[RaySamples] = None, # small object ray samples
            object_model_list: List[Model] = None,
            object_dataparser_outputs_list: List[DataparserOutputs] = None,
            scene_objects_tmin_list: List[Tensor] = None,  # full
            scene_objects_tmax_list: List[Tensor] = None, 
            objects_selectors_list: List[Tensor] = None # full
    ):
        # init list
        scene_scale = dataparser_outputs.dataparser_scale
        compo_object_indices = []
        object_density_list = []
        object_density_embedding_list = []
        object_depth_rays_selectors_list = [] 
        full_object_depth_rays_selectors_list = [] 
        compo_rays_selector = torch.zeros((scene_ray_samples.shape[0]), dtype=torch.bool, device=self.device)
        object_samples_selectors_list = []

        # iter objects
        for index, object_ray_samples in enumerate(object_ray_samples_list):
            object_scale = object_dataparser_outputs_list[index].dataparser_scale
            object_model = object_model_list[index]
            object_selector = objects_selectors_list[index]

            object_t_min = scene_objects_tmin_list[index][object_selector] * (object_scale/scene_scale) # small size
            object_t_max = scene_objects_tmax_list[index][object_selector] * (object_scale/scene_scale)
        
            #NOTE: depth, density calculation of intersected rays with object model
            object_density_flat, object_density_embedding_flat = object_model.field.get_density(object_ray_samples) # small size

            object_density = object_density_flat.view(*object_ray_samples.shape, -1) # num_rays, 48, 1  # small size
            object_density_embedding = object_density_embedding_flat.view(*object_ray_samples.shape, -1) # num_rays, 48, 15
            
            #NOTE: compute depth pixels/rays
            object_weights = object_ray_samples.get_weights(object_density) # TODO: chect to store ?
            object_rays_depth = object_model.renderer_depth(weights=object_weights, ray_samples=object_ray_samples) 
            
            object_rays_t_depth = object_rays_depth.squeeze()
            object_depth_rays_selector = (object_rays_t_depth >= object_t_min) & (object_rays_t_depth <= object_t_max) # rays small size
            
            # NOTE: Ray Compo process
            if object_depth_rays_selector.sum().item() > 0:
                compo_object_indices.append(index) # 2nd objects filter in view, 1st: object exist in view, 2nd: object visible in view  

                object_density_list.append(object_density) # small
                object_density_embedding_list.append(object_density_embedding) # small
                
                full_object_depth_rays_selector = object_selector.clone() # full object aabb exist in view
                full_object_depth_rays_selector[object_selector] = object_depth_rays_selector # rays inside aabb and depth insdide aabb, full size
                object_depth_rays_selectors_list.append(object_depth_rays_selector) # object small size
                full_object_depth_rays_selectors_list.append(full_object_depth_rays_selector) # full size
                # Accumulate compo pixels across objects
                compo_rays_selector |= full_object_depth_rays_selector # full size

                # start_time = time.time()
                filtered_object_ray_samples = object_ray_samples[object_depth_rays_selector] # object compo size
                filtered_object_t_min = object_t_min[object_depth_rays_selector] # crop t tensor with depth inside filter 
                filtered_object_t_max = object_t_max[object_depth_rays_selector]
                small_object_samples_selector = get_samples_selector(filtered_object_ray_samples, filtered_object_t_min, filtered_object_t_max) # get samples inside aabb, object compo size
                # small_object_samples_selector = get_samples_inside_box(filtered_object_ray_samples, aabb=object_model.scene_box.aabb)
                
                # object_samples_selector_flat = object_samples_selector.view(-1)
                object_samples_selectors_list.append(small_object_samples_selector) # compo size

        return compo_object_indices, object_density_list, object_density_embedding_list, object_depth_rays_selectors_list, full_object_depth_rays_selectors_list, compo_rays_selector, object_samples_selectors_list

    def compo_full( # field_outputs from object and scene rays with object_selector of scene model
            self, 
            scene_ray_samples: RaySamples,
            dataparser_outputs: DataparserOutputs,
            object_ray_samples_list: List[RaySamples] = None, # small object rays samples
            object_models_list: List[Model] = None,
            object_dataparser_outputs_list: List[DataparserOutputs] = None,
            scene_rois_aabb_tmin_list: List[Tensor] = None, # full
            scene_rois_aabb_tmax_list: List[Tensor] = None, 
            rois_selectors_list: List[Tensor] = None
    ):
        # NOTE: Get infos from objects: indices, rays and samples selector
        compo_object_indices, object_density_list, object_density_embedding_list, object_depth_rays_selectors_list, full_object_depth_rays_selectors_list, compo_rays_selector, object_samples_selectors_list = self.objects_processing_small(
            scene_ray_samples,
            dataparser_outputs,
            object_ray_samples_list, # small object rays samples
            object_models_list,
            object_dataparser_outputs_list,
            scene_rois_aabb_tmin_list, # full, # num: 32768
            scene_rois_aabb_tmax_list,
            rois_selectors_list
        )

        compo_selector = torch.zeros((scene_ray_samples.shape[0], 1), device=self.device, dtype=torch.bool)
        compo_scene_rgb = torch.zeros((scene_ray_samples.shape[0], 3), device=self.device)

        # check if object need to compo
        if compo_object_indices:
            #NOTE: initialize pixel infos tensor of image
            rgb = torch.zeros((scene_ray_samples.shape[0], 3), device=self.device)
            depth = torch.zeros((scene_ray_samples.shape[0], 1), device=self.device)
            accumulation = torch.zeros((scene_ray_samples.shape[0], 1), device=self.device)

            scene_scale = dataparser_outputs.dataparser_scale

            # NOTE: Compute order of objects
            intersection_rays_selectors_list = []
            if len(compo_object_indices) > 1:
                object_indices_order = []
                relative_object_indices_order = []
                for i in range(len(compo_object_indices)):
                    for j in range(i + 1, len(compo_object_indices)):
                        intersection_rays_selector = full_object_depth_rays_selectors_list[i] & full_object_depth_rays_selectors_list[j] # full size
                        if intersection_rays_selector.sum().item() > 0:
                            # i
                            t_min_i = scene_rois_aabb_tmin_list[compo_object_indices[i]][intersection_rays_selector] # t wasn't cropped by intersection aabb
                            t_max_i = scene_rois_aabb_tmax_list[compo_object_indices[i]][intersection_rays_selector]  
                            t_center_i = (t_min_i + t_max_i) / 2
                            # j
                            t_min_j = scene_rois_aabb_tmin_list[compo_object_indices[j]][intersection_rays_selector] # t wasn't cropped by intersection aabb
                            t_max_j = scene_rois_aabb_tmax_list[compo_object_indices[j]][intersection_rays_selector] 
                            t_center_j = (t_min_j + t_max_j) / 2
                            
                            # TODO: compare and get object order
                            if torch.mean(t_center_i).item() < torch.mean(t_center_j).item() :
                                object_indices_order.append([compo_object_indices[i], compo_object_indices[j]])
                                relative_object_indices_order.append([i, j])
                            else:
                                object_indices_order.append([compo_object_indices[j], compo_object_indices[i]])
                                relative_object_indices_order.append([j, i])
                            intersection_rays_selectors_list.append(intersection_rays_selector) # full size

            # NOTE: Get infos from scene: rays and samples selector
            scene_samples_inside_selector = torch.zeros((scene_ray_samples.shape), dtype=torch.bool, device=self.device) # init scene samples selector for compo with full shape
            for index, object_indice in enumerate(compo_object_indices):
                scene_object_t_min = scene_rois_aabb_tmin_list[object_indice] # t wasn't cropped by intersection aabb, full size
                scene_object_t_max = scene_rois_aabb_tmax_list[object_indice] 
                
                inside_filtered_scene_ray_samples = scene_ray_samples[full_object_depth_rays_selectors_list[index]] # Crop inside aabb scene rays with depth inside filter full, 1 separate object compo size
                filtered_scene_t_min = scene_object_t_min[full_object_depth_rays_selectors_list[index]] # crop t tensor with depth inside filter full, separate object compo size
                filtered_scene_t_max = scene_object_t_max[full_object_depth_rays_selectors_list[index]]
                scene_obcompo_samples_inside_selector = get_samples_selector(inside_filtered_scene_ray_samples, filtered_scene_t_min, filtered_scene_t_max) # get scene samples inside aabb, object for compo size
                scene_samples_inside_selector[full_object_depth_rays_selectors_list[index]] |= scene_obcompo_samples_inside_selector # scene samples inside valid multiple masks OR, full size
                                
            # NOTE: Get field_outputs from scene model for just compo rays, others will be 0, just in compo_rays_selector:
            scene_compo_ray_samples = scene_ray_samples[compo_rays_selector] # compo size
            scene_compo_samples_selector = ~(scene_samples_inside_selector[compo_rays_selector]) # compo size
            
            scene_compo_positions = scene_compo_ray_samples.frustums.get_positions()#.view(-1, 3) # num, 48, 3
            scene_compo_directions = scene_compo_ray_samples.frustums.directions#.reshape(-1, 3) # num, 48, 3
            scene_compo_camera_indices = scene_compo_ray_samples.camera_indices#.reshape(-1, 1) # num, 48, 1

            scene_compo_samples_positions = scene_compo_positions[scene_compo_samples_selector] # n, 3
            scene_compo_samples_directions = scene_compo_directions[scene_compo_samples_selector] # n, 3
            scene_compo_samples_camera_indices = scene_compo_camera_indices[scene_compo_samples_selector] # n, 1

            scene_selected = {}
            scene_selected["positions_flat"] = scene_compo_samples_positions # n, 3
            scene_selected["directions_flat"] = scene_compo_samples_directions
            scene_selected["camera_indices_flat"] = scene_compo_samples_camera_indices

            scene_field_outputs = {}
            scene_field_outputs[FieldHeadNames.DENSITY] = torch.zeros((*scene_compo_ray_samples.shape, 1), device=self.device) # num, 48, 1 # scene compo size
            scene_field_outputs[FieldHeadNames.RGB] = torch.zeros((*scene_compo_ray_samples.shape, 3), device=self.device) # num, 48, 3

            field_outputs = self.field.forward(scene_compo_ray_samples, compute_normals=self.config.predict_normals, selected_infos=scene_selected)

            scene_field_outputs[FieldHeadNames.DENSITY][scene_compo_samples_selector] = field_outputs[FieldHeadNames.DENSITY]
            scene_field_outputs[FieldHeadNames.RGB][scene_compo_samples_selector] = field_outputs[FieldHeadNames.RGB]

            # NOTE: compo separte
            # NOTE: intersection first:
            if intersection_rays_selectors_list:
                # Clone selectors list for intersection
                intersection_full_object_depth_rays_selector = full_object_depth_rays_selectors_list.copy() # full size
                intersection_object_depth_rays_selector = object_depth_rays_selectors_list.copy() # small size
                intersection_object_samples_selector = object_samples_selectors_list.copy()

                for index, intersection_rays_selector in enumerate(intersection_rays_selectors_list): # full size
                    # scene intersect
                    intersect_scene_ray_samples = scene_ray_samples[intersection_rays_selector] # inteX size
                    intersect_scene_samples_selector = ~scene_samples_inside_selector[intersection_rays_selector] # inteX size
                    intersect_scene_field_outputs = {}
                    intersect_scene_field_outputs[FieldHeadNames.DENSITY] = scene_field_outputs[FieldHeadNames.DENSITY][intersection_rays_selector[compo_rays_selector]] # inteX size
                    intersect_scene_field_outputs[FieldHeadNames.RGB] = scene_field_outputs[FieldHeadNames.RGB][intersection_rays_selector[compo_rays_selector]]

                    # init bins for merged
                    merged_spacing_bins = None
                    merged_bins_selector = None
                    merged_field_outputs = None

                    for intersect_indice, relative_indice in zip(object_indices_order[index], relative_object_indices_order[index]):
                        object_scale = object_dataparser_outputs_list[intersect_indice].dataparser_scale
                        object_model = object_models_list[intersect_indice]
                        object_ray_samples = object_ray_samples_list[intersect_indice]  # small size
                        object_selector = rois_selectors_list[intersect_indice] # full size -> small

                        # get object selectors at multiple size of rays and samples
                        full_object_depth_rays_selector = intersection_full_object_depth_rays_selector[relative_indice] # full size
                        object_depth_rays_selector = intersection_object_depth_rays_selector[relative_indice] # small size
                        object_samples_selector = intersection_object_samples_selector[relative_indice] # samples selectors compo size

                        # Crop full intersection selector to compo:
                        small_intersection_rays_selector = intersection_rays_selector[object_selector] # small size -> inteX
                        compo_intersection_rays_selector = intersection_rays_selector[full_object_depth_rays_selector] # compo size -> inteX
                        intersection_samples_selector = object_samples_selector[compo_intersection_rays_selector] # intersect size
                        
                        # NOTE: Exclude intersection and re-store in lists
                        full_object_depth_rays_selectors_list[relative_indice] = full_object_depth_rays_selector ^ intersection_rays_selector # full
                        object_depth_rays_selectors_list[relative_indice] = object_depth_rays_selector ^ small_intersection_rays_selector # small
                        object_samples_selectors_list[relative_indice] = object_samples_selector[~compo_intersection_rays_selector] # compo - intersect
                        
                        # NOTE: Get object ray samples infos 
                        # Filter object ray samples with intersect selector
                        intersect_object_ray_samples = object_ray_samples[small_intersection_rays_selector]  # Crop object rays small to intersect size
                        # NOTE: merge rays here
                        if merged_spacing_bins is not None :
                            # Recompute scene samples selector for overlap merged rays
                            intersect_scene_object_t_min = scene_rois_aabb_tmin_list[intersect_indice][intersection_rays_selector]
                            intersect_scene_object_t_max = scene_rois_aabb_tmax_list[intersect_indice][intersection_rays_selector] 
                            merged_bins_selector = get_bins_selector(merged_spacing_bins, scene_ray_samples, intersect_scene_object_t_min, intersect_scene_object_t_max, inside=False) # get samples outside aabb with inside=False

                        if intersect_object_ray_samples.shape != intersection_samples_selector.shape:
                            print("Error here")
                        # Do merged bin of scene and object here
                        merged_spacing_bins, sorted_index = merge_spacing_bins( # bin spacing of ray_samples Good
                            intersect_scene_ray_samples, 
                            intersect_object_ray_samples, 
                            merged_spacing_bins, 
                            intersect_scene_samples_selector, 
                            intersection_samples_selector, 
                            merged_bins_selector,
                            (scene_scale/object_scale)
                        )

                        # NOTE: Get field_outputs from objet model for just interect rays, others will be 0
                        # NOTE: crop density, embedding
                        object_density = object_density_list[relative_indice] # density rays, small size
                        object_density_embedding = object_density_embedding_list[relative_indice]

                        intersect_object_rays_density = object_density[small_intersection_rays_selector] # Crop small to inteX
                        intersect_object_rays_density_embedding = object_density_embedding[small_intersection_rays_selector]

                        intersect_object_samples_density = intersect_object_rays_density[intersection_samples_selector] # filter samples in inteX size
                        intersect_object_samples_density_embedding = intersect_object_rays_density_embedding[intersection_samples_selector]

                        # object samples selector
                        object_positions = intersect_object_ray_samples.frustums.get_positions() # num * 48, 3 , # inteX size
                        object_directions = intersect_object_ray_samples.frustums.directions # num * 48, 3
                        object_camera_indices = intersect_object_ray_samples.camera_indices # num * 48, 1

                        object_samples_positions = object_positions[intersection_samples_selector] # n, 3 , # filter samples in inteX size
                        object_samples_directions = object_directions[intersection_samples_selector] # n, 3
                        object_samples_camera_indices = object_camera_indices[intersection_samples_selector] # n, 1
                        
                        object_selected = {}
                        object_selected["positions_flat"] = object_samples_positions # n, 3
                        object_selected["directions_flat"] = object_samples_directions
                        object_selected["camera_indices_flat"] = object_samples_camera_indices

                        # Object field_outputs
                        intersection_object_field_outputs = object_model.field.get_outputs(intersect_object_ray_samples, density_embedding=intersect_object_samples_density_embedding, selected_infos=object_selected)
                        intersection_object_field_outputs[FieldHeadNames.DENSITY] = intersect_object_samples_density

                        object_field_outputs = {}
                        object_field_outputs[FieldHeadNames.DENSITY] = torch.zeros((*intersect_object_ray_samples.shape, 1), device=self.device) # num, 48, 1 # object compo size
                        object_field_outputs[FieldHeadNames.RGB] = torch.zeros((*intersect_object_ray_samples.shape, 3), device=self.device) # num, 48, 3
                        
                        object_field_outputs[FieldHeadNames.DENSITY][intersection_samples_selector] = intersection_object_field_outputs[FieldHeadNames.DENSITY]
                        object_field_outputs[FieldHeadNames.RGB][intersection_samples_selector] = intersection_object_field_outputs[FieldHeadNames.RGB]
                        
                        # NOTE: merge field_outputs here
                        if merged_field_outputs is not None :
                            base_field_outputs = merged_field_outputs
                        else:
                            base_field_outputs = intersect_scene_field_outputs

                        field_outputs = {}
                        field_outputs[FieldHeadNames.DENSITY] = torch.cat([base_field_outputs[FieldHeadNames.DENSITY], object_field_outputs[FieldHeadNames.DENSITY]], dim=1)
                        field_outputs[FieldHeadNames.RGB] = torch.cat([base_field_outputs[FieldHeadNames.RGB], object_field_outputs[FieldHeadNames.RGB]], dim=1)

                        expanded_density_sorted_index = sorted_index.unsqueeze(-1)
                        expanded_rgb_sorted_index = sorted_index.unsqueeze(-1).expand(-1, -1, 3)

                        field_outputs[FieldHeadNames.DENSITY] = torch.gather(field_outputs[FieldHeadNames.DENSITY], 1, expanded_density_sorted_index)
                        field_outputs[FieldHeadNames.RGB] = torch.gather(field_outputs[FieldHeadNames.RGB], 1, expanded_rgb_sorted_index)

                        merged_field_outputs = field_outputs
                    
                    merged_intersect_ray_samples = get_rays_from_spacing_bins(intersect_scene_ray_samples, merged_spacing_bins)
                    
                    # NOTE: render pixels of image and store to tensors
                    if self.config.use_gradient_scaling:
                        merged_field_outputs = scale_gradients_by_distance_squared(merged_field_outputs, merged_intersect_ray_samples)

                    merged_intersect_weights = merged_intersect_ray_samples.get_weights(merged_field_outputs[FieldHeadNames.DENSITY])

                    merged_intersect_rgb = self.renderer_rgb(rgb=merged_field_outputs[FieldHeadNames.RGB], weights=merged_intersect_weights)
                    merged_intersect_depth = self.renderer_depth(weights=merged_intersect_weights, ray_samples=merged_intersect_ray_samples)
                    merged_intersect_accumulation = self.renderer_accumulation(weights=merged_intersect_weights)

                    rgb[intersection_rays_selector] = merged_intersect_rgb
                    depth[intersection_rays_selector] = merged_intersect_depth
                    accumulation[intersection_rays_selector] = merged_intersect_accumulation

            # NOTE: NO intersection then:
            # NOTE: Get field_outputs from objects models note intersect
            for index, object_indice in enumerate(compo_object_indices):
                # NOTE: get object selectors full and check if still exist object rays after intersections
                full_object_depth_rays_selector = full_object_depth_rays_selectors_list[index] # full size
                if full_object_depth_rays_selector.sum().item() > 0:
                    # get object selectors at multiple size of rays and samples
                    object_depth_rays_selector = object_depth_rays_selectors_list[index] # small size -> to scrop to compo size 
                    object_samples_selector = object_samples_selectors_list[index] # NOTE: samples selectors at compo size, already exclusive interection if exist

                    # Get infos
                    object_scale = object_dataparser_outputs_list[object_indice].dataparser_scale
                    object_model = object_models_list[object_indice]
                    object_ray_samples = object_ray_samples_list[object_indice]  # small size
                    object_selector = rois_selectors_list[object_indice] # full size -> small

                    # NOTE: Get scene ray samples infos 
                    # Filter object ray samples with depth selector
                    obcompo_scene_ray_samples = scene_ray_samples[full_object_depth_rays_selector]  # Crop scene rays full to obcompo size
                    obcompo_scene_samples_selector = ~(scene_samples_inside_selector[full_object_depth_rays_selector]) # obcompo size

                    obcompo_scene_field_outputs = {}
                    obcompo_scene_field_outputs[FieldHeadNames.DENSITY] = scene_field_outputs[FieldHeadNames.DENSITY][full_object_depth_rays_selector[compo_rays_selector]] # obcompo size
                    obcompo_scene_field_outputs[FieldHeadNames.RGB] = scene_field_outputs[FieldHeadNames.RGB][full_object_depth_rays_selector[compo_rays_selector]]

                    # NOTE: Get object ray samples infos 
                    # Filter object ray samples with depth selector
                    compo_object_ray_samples = object_ray_samples[object_depth_rays_selector]  # Crop object rays with depth inside filter small, compo size

                    # NOTE: merge rays here
                    # Do merged bin of scene and object here
                    merged_spacing_bins, sorted_index = merge_spacing_bins(
                        obcompo_scene_ray_samples, 
                        compo_object_ray_samples, 
                        None, 
                        obcompo_scene_samples_selector, 
                        object_samples_selector, 
                        None,
                        (scene_scale/object_scale)
                    )

                    merged_ray_samples = get_rays_from_spacing_bins(obcompo_scene_ray_samples, merged_spacing_bins)

                    # NOTE: crop density, embedding
                    object_density = object_density_list[index] # density rays, small size
                    object_density_embedding = object_density_embedding_list[index]

                    object_rays_density = object_density[object_depth_rays_selector] # Crop small to obcompo
                    object_rays_density_embedding = object_density_embedding[object_depth_rays_selector]

                    object_samples_density = object_rays_density[object_samples_selector] # filter samples in obcompo size
                    object_samples_density_embedding = object_rays_density_embedding[object_samples_selector]

                    # object sample selector
                    object_positions = compo_object_ray_samples.frustums.get_positions() # num * 48, 3
                    object_directions = compo_object_ray_samples.frustums.directions # num * 48, 3
                    object_camera_indices = compo_object_ray_samples.camera_indices # num * 48, 1

                    object_samples_positions = object_positions[object_samples_selector] # n, 3
                    object_samples_directions = object_directions[object_samples_selector] # n, 3
                    object_samples_camera_indices = object_camera_indices[object_samples_selector] # n, 1
                    
                    object_selected = {}
                    object_selected["positions_flat"] = object_samples_positions # n, 3
                    object_selected["directions_flat"] = object_samples_directions
                    object_selected["camera_indices_flat"] = object_samples_camera_indices

                    compo_object_field_outputs = object_model.field.get_outputs(compo_object_ray_samples, density_embedding=object_samples_density_embedding, selected_infos=object_selected)
                    compo_object_field_outputs[FieldHeadNames.DENSITY] = object_samples_density

                    object_field_outputs = {}
                    object_field_outputs[FieldHeadNames.DENSITY] = torch.zeros((*compo_object_ray_samples.shape, 1), device=self.device) # num, 48, 1 # object compo size
                    object_field_outputs[FieldHeadNames.RGB] = torch.zeros((*compo_object_ray_samples.shape, 3), device=self.device) # num, 48, 3
                    
                    object_field_outputs[FieldHeadNames.DENSITY][object_samples_selector] = compo_object_field_outputs[FieldHeadNames.DENSITY]
                    object_field_outputs[FieldHeadNames.RGB][object_samples_selector] = compo_object_field_outputs[FieldHeadNames.RGB]

                    # NOTE: merge field_outputs here
                    merged_field_outputs = {}
                    merged_field_outputs[FieldHeadNames.DENSITY] = torch.cat([obcompo_scene_field_outputs[FieldHeadNames.DENSITY], object_field_outputs[FieldHeadNames.DENSITY]], dim=1)
                    merged_field_outputs[FieldHeadNames.RGB] = torch.cat([obcompo_scene_field_outputs[FieldHeadNames.RGB], object_field_outputs[FieldHeadNames.RGB]], dim=1)

                    expanded_density_sorted_index = sorted_index.unsqueeze(-1)
                    expanded_rgb_sorted_index = sorted_index.unsqueeze(-1).expand(-1, -1, 3)

                    merged_field_outputs[FieldHeadNames.DENSITY] = torch.gather(merged_field_outputs[FieldHeadNames.DENSITY], 1, expanded_density_sorted_index)
                    merged_field_outputs[FieldHeadNames.RGB] = torch.gather(merged_field_outputs[FieldHeadNames.RGB], 1, expanded_rgb_sorted_index)

                    # NOTE: render pixels of image and store to tensors
                    if self.config.use_gradient_scaling:
                        merged_field_outputs = scale_gradients_by_distance_squared(merged_field_outputs, merged_ray_samples)

                    merged_weights = merged_ray_samples.get_weights(merged_field_outputs[FieldHeadNames.DENSITY])

                    merged_rgb = self.renderer_rgb(rgb=merged_field_outputs[FieldHeadNames.RGB], weights=merged_weights)
                    merged_depth = self.renderer_depth(weights=merged_weights, ray_samples=merged_ray_samples)
                    merged_accumulation = self.renderer_accumulation(weights=merged_weights)

                    rgb[full_object_depth_rays_selector] = merged_rgb
                    depth[full_object_depth_rays_selector] = merged_depth
                    accumulation[full_object_depth_rays_selector] = merged_accumulation

                    # NOTE: Compute scene at compo pixel
                    compo_pixel_scene_ray_samples = scene_ray_samples[full_object_depth_rays_selector]
                    compo_pixel_scene_field_outputs = self.field.forward(compo_pixel_scene_ray_samples, compute_normals=self.config.predict_normals)
                    compo_pixel_scene_field_outputs[FieldHeadNames.DENSITY] = compo_pixel_scene_field_outputs[FieldHeadNames.DENSITY].view(*compo_pixel_scene_ray_samples.shape, -1) 
                    compo_pixel_scene_field_outputs[FieldHeadNames.RGB] = compo_pixel_scene_field_outputs[FieldHeadNames.RGB].view(*compo_pixel_scene_ray_samples.shape, -1)

                    compo_pixel_scene_weights = compo_pixel_scene_ray_samples.get_weights(compo_pixel_scene_field_outputs[FieldHeadNames.DENSITY])
                    compo_pixel_scene_rgb = self.renderer_rgb(rgb=compo_pixel_scene_field_outputs[FieldHeadNames.RGB], weights=compo_pixel_scene_weights) # normalized rgb in [0, 1]

                    compo_scene_rgb[full_object_depth_rays_selector] = compo_pixel_scene_rgb

                    # # NOTE: compositional pixel here
                    # rgb[full_object_depth_rays_selector] = compo_rgb_ct
                    # depth[full_object_depth_rays_selector] = merged_depth
                    # accumulation[full_object_depth_rays_selector] = merged_accumulation
                    compo_selector |= full_object_depth_rays_selector.unsqueeze(-1)

            # NOTE: Render outside scence rays part 
            # NOTE: Get field_outputs from scene model for just outside rays
            # check if outside rays exist
            if (~compo_rays_selector).sum().item() > 0:
                scene_outside_ray_samples = scene_ray_samples[~compo_rays_selector] # outside size
                field_outputs = self.field.forward(scene_outside_ray_samples, compute_normals=self.config.predict_normals)

                field_outputs[FieldHeadNames.DENSITY] = field_outputs[FieldHeadNames.DENSITY].view(*scene_outside_ray_samples.shape, -1)
                field_outputs[FieldHeadNames.RGB] = field_outputs[FieldHeadNames.RGB].view(*scene_outside_ray_samples.shape, -1)

                #NOTE: render pixels of image and stack to tensors
                if self.config.use_gradient_scaling:
                    field_outputs = scale_gradients_by_distance_squared(field_outputs, scene_outside_ray_samples)

                outside_weights = scene_outside_ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

                outside_rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=outside_weights)
                outside_depth = self.renderer_depth(weights=outside_weights, ray_samples=scene_outside_ray_samples)
                outside_accumulation = self.renderer_accumulation(weights=outside_weights)

                rgb[~compo_rays_selector] = outside_rgb
                depth[~compo_rays_selector] = outside_depth
                accumulation[~compo_rays_selector] = outside_accumulation
   
        else:
            ray_samples = scene_ray_samples
            field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
            field_outputs[FieldHeadNames.DENSITY] = field_outputs[FieldHeadNames.DENSITY].view(*ray_samples.shape, -1) 
            field_outputs[FieldHeadNames.RGB] = field_outputs[FieldHeadNames.RGB].view(*ray_samples.shape, -1)

            if self.config.use_gradient_scaling:
                field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

            weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

            rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
            accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "compo_selector": compo_selector,
            "compo_scene_rgb": compo_scene_rgb,
        }
        return outputs

    def get_outputs( # get outputs multiple rois v0
            self, 
            ray_bundle: RayBundle, 
            object_ray_bundles_list: List[RayBundle] = None, 
            object_models_list: List[Model] = None,
            dataparser_outputs: DataparserOutputs = None,
            object_dataparser_outputs_list: List[DataparserOutputs] = None,
            scene_object_boxes_list: Optional[List[SceneBox]] = None,
    ):
        ray_samples: RaySamples
        scene_ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        aabb_selector = torch.zeros((scene_ray_samples.shape[0], 1), device=self.device, dtype=torch.bool)
        
        if not self.training and object_ray_bundles_list:
            rays_o = ray_bundle.origins.contiguous()
            rays_d = ray_bundle.directions.contiguous()
            max_bound : float = 1e10

            inside_roi_indices_list = []
            scene_rois_aabb_tmin_list = []
            scene_rois_aabb_tmax_list = []
            rois_selectors_list = []
            inside_object_ray_samples_list = []

            # iterate objects list, compute rays selector samples inside AABB -> selector
            for index, scene_object_box in enumerate(scene_object_boxes_list):
                with torch.no_grad():
                    tensor_aabb = Parameter(scene_object_box.aabb.flatten(), requires_grad=False)
                    tensor_aabb = tensor_aabb.to(rays_o.device)

                    tx_min = (tensor_aabb[:3] - rays_o) / rays_d
                    tx_max = (tensor_aabb[3:] - rays_o) / rays_d

                    t_min = torch.stack((tx_min, tx_max)).amin(dim=0)
                    t_max = torch.stack((tx_min, tx_max)).amax(dim=0)

                    t_min = t_min.amax(dim=-1)
                    t_max = t_max.amin(dim=-1)

                    t_min = torch.clamp(t_min, min=0, max=max_bound)
                    t_max = torch.clamp(t_max, min=0, max=max_bound)

                    selector = t_max > t_min

                    aabb_selector |= selector.unsqueeze(-1)
                

                if selector.sum().item() > 0:
                    inside_roi_indices_list.append(index)

                    scene_rois_aabb_tmin_list.append(t_min) # full size
                    scene_rois_aabb_tmax_list.append(t_max) # full size
                    rois_selectors_list.append(selector)
                    
                    object_model = object_models_list[index]
                    inside_object_ray_bundle = object_ray_bundles_list[index][selector]
                    inside_object_ray_samples, _, _ = object_model.proposal_sampler(inside_object_ray_bundle, density_fns=object_model.density_fns)
                    inside_object_ray_samples_list.append(inside_object_ray_samples)

            if inside_roi_indices_list:
                inside_object_models = np.array(object_models_list)[inside_roi_indices_list].tolist()
                inside_object_dataparser_outputs = np.array(object_dataparser_outputs_list)[inside_roi_indices_list].tolist()

                # outputs = self.compo_v0( # scene sample 
                # outputs = self.compo_v1( # roi sample (multi non support)
                # outputs = self.compo_v2( # scene sample + scene depth filter
                outputs = self.compo_full( # full: roi sample + roi depth filter
                    scene_ray_samples, 
                    dataparser_outputs, 
                    inside_object_ray_samples_list, inside_object_models, inside_object_dataparser_outputs, 
                    scene_rois_aabb_tmin_list, scene_rois_aabb_tmax_list, # full size
                    rois_selectors_list
                )
                outputs["aabb_selector"] = aabb_selector
                return outputs                
    
        #NOTE: image that not view object box, combine with the condition that there is no object model
        ray_samples = scene_ray_samples
        outputs_shape = ray_samples.shape
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        field_outputs[FieldHeadNames.DENSITY] = field_outputs[FieldHeadNames.DENSITY].view(*outputs_shape, -1) 
        field_outputs[FieldHeadNames.RGB] = field_outputs[FieldHeadNames.RGB].view(*outputs_shape, -1)

        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        compo_selector = torch.zeros((ray_samples.shape[0], 1), device=self.device, dtype=torch.bool)
        compo_scene_rgb = torch.zeros((scene_ray_samples.shape[0], 3), device=self.device)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "compo_selector": compo_selector,
            "compo_scene_rgb": compo_scene_rgb,
            "aabb_selector": aabb_selector,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        if not object_ray_bundles_list:
            for i in range(self.config.num_proposal_iterations):
                outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict

    def get_image_metrics_and_images_compo(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], aabb_only: bool = False, compo_outputs: Dict[str, torch.Tensor] = None
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)
        
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )
        
        if aabb_only:
            black_image = torch.zeros_like(gt_rgb)
            if compo_outputs is not None:
                selector = compo_outputs["aabb_selector"].squeeze(-1)
            else:
                selector = outputs["aabb_selector"].squeeze(-1)
            predicted_rgb[~selector] = black_image[~selector]
            gt_rgb[~selector] = black_image[~selector]

        combined_rgb = torch.cat([predicted_rgb], dim=1)
        # combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        # combined_acc = torch.cat([acc], dim=1)
        # combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        if aabb_only:
            expanded_selector = selector.unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1)
            gt_rgb_small = gt_rgb[expanded_selector]
            predicted_rgb_small = predicted_rgb[expanded_selector]
        else:
            gt_rgb_small = gt_rgb
            predicted_rgb_small = predicted_rgb

        psnr = self.psnr(gt_rgb_small, predicted_rgb_small)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        # images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        images_dict = {"img": combined_rgb}

        # for i in range(self.config.num_proposal_iterations):
        #     key = f"prop_depth_{i}"
        #     prop_depth_i = colormaps.apply_depth_colormap(
        #         outputs[key],
        #         accumulation=outputs["accumulation"],
        #     )
        #     images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
