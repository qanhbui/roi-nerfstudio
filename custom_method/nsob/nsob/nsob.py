"""
NeRF implementation that combines many recent advancements.
"""

# from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type, Union
from collections import defaultdict

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation

from nerfstudio.fields.base_field import Field
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
# from nerfstudio.fields.nerfacto_field import NerfactoField
from nsob.fields.nsob_field import NsobField


from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler

from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps


@dataclass
class NsobModelConfig(ModelConfig):
    """Nsob Model Config"""

    _target: Type = field(default_factory=lambda: NsobModel)
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


class NsobModel(Model):
    """Nsob model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NsobModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = NsobField(
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

        self.field_object = NsobField(
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

        def create_proposal_networks():
            density_fns = []
            num_prop_nets = self.config.num_proposal_iterations
            # Build the proposal network(s)
            proposal_networks = torch.nn.ModuleList()
            if self.config.use_same_proposal_network:
                assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
                prop_net_args = self.config.proposal_net_args_list[0]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                )
                proposal_networks.append(network)
                density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
            else:
                for i in range(num_prop_nets):
                    prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                    network = HashMLPDensityField(
                        self.scene_box.aabb,
                        spatial_distortion=scene_contraction,
                        **prop_net_args,
                        implementation=self.config.implementation,
                    )
                    proposal_networks.append(network)
                density_fns.extend([network.density_fn for network in proposal_networks])
            return proposal_networks, density_fns

        self.proposal_networks, self.density_fns = create_proposal_networks()
        self.proposal_networks_object, self.density_fns_object = create_proposal_networks()

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

        self.proposal_sampler_object = ProposalNetworkSampler(
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
        self.accumulation_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters()) + list(self.proposal_networks_object.parameters())
        param_groups["fields"] = list(self.field.parameters()) + list(self.field_object.parameters())
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
                # self.proposal_sampler_object.set_anneal(anneal) # TODO: not sure to add this, need to verify function, black and white and not converge to color

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
    
    def get_outputs(self, ray_bundle: RayBundle, batch: Dict[str, torch.Tensor] = None):
        if batch:
            self.outputs_scene = self.get_outputs_group(ray_bundle=ray_bundle, batch=batch, proposal_sampler=self.proposal_sampler, density_fns=self.density_fns, field=self.field, group_name="")
            self.outputs_object = self.get_outputs_group(ray_bundle=ray_bundle, batch=batch, proposal_sampler=self.proposal_sampler_object, density_fns=self.density_fns_object, field=self.field_object, group_name="_object")
        else:
            self.outputs_scene = self.get_outputs_group(ray_bundle=ray_bundle, batch=None, proposal_sampler=self.proposal_sampler, density_fns=self.density_fns, field=self.field, group_name="")
            self.outputs_object = self.get_outputs_group(ray_bundle=ray_bundle, batch=None, proposal_sampler=self.proposal_sampler_object, density_fns=self.density_fns_object, field=self.field_object, group_name="_object")
        outputs = {}
        outputs.update(self.outputs_scene)
        outputs.update(self.outputs_object)
        return outputs
    
    def get_outputs_group(self, ray_bundle: RayBundle, batch, proposal_sampler: ProposalNetworkSampler, density_fns, field: Field, group_name: Literal["", "_object"] = ""): # TODO: Rename normals when need to use 
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = proposal_sampler(ray_bundle, density_fns=density_fns)
        field_outputs = field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        occlusion_mask = None
        if self.training:
            if group_name == "_object":
                if batch:
                    pass
        
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY], occlusion_mask)
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            f"rgb{group_name}": rgb,
            f"accumulation{group_name}": accumulation,
            f"depth{group_name}": depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs[f"weights_list{group_name}"] = weights_list
            outputs[f"ray_samples_list{group_name}"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth{group_name}_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        mask = batch["mask"].to(self.device)  # RGB or RGBA image
        gt_rgb_object = ~(mask) + (mask * gt_rgb) # set background color white 1 to match output

        predicted_rgb = outputs["rgb"]
        predicted_rgb_object = outputs["rgb_object"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        metrics_dict["psnr_object"] = self.psnr(predicted_rgb_object, gt_rgb_object)

        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
            metrics_dict["distortion_object"] = distortion_loss(outputs["weights_list_object"], outputs["ray_samples_list_object"])
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

        loss_dict_object = self.get_loss_dict_object(outputs, batch, metrics_dict)
        loss_dict.update(loss_dict_object) # maybe lost gradient track here
        return loss_dict

    def get_loss_dict_object(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        mask = batch["mask"].to(self.device)
        # gt_rgb = ~(mask) + (mask * image) # set background color white 1 to match output
        # gt_rgb = self.renderer_rgb.blend_background(gt_rgb)

        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_object"][mask.flatten()],
            pred_accumulation=outputs["accumulation_object"][mask.flatten()],
            gt_image=image[mask.flatten()],
        )

        # pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
        #     pred_image=outputs["rgb_object"],
        #     pred_accumulation=outputs["accumulation_object"],
        #     gt_image=gt_rgb,
        # )

        loss_dict["rgb_loss_object"] = self.rgb_loss(gt_rgb, pred_rgb)

        pred_accumulation = outputs["accumulation_object"]
        gt_mask = mask.float()
        loss_dict["accumulation_loss_object"] = 10 * self.accumulation_loss(gt_mask, pred_accumulation)

        if self.training:
            loss_dict["interlevel_loss_object"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list_object"], outputs["ray_samples_list_object"]
            )
            assert metrics_dict is not None and "distortion_object" in metrics_dict
            loss_dict["distortion_loss_object"] = self.config.distortion_loss_mult * metrics_dict["distortion_object"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss_object"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss_object"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss_object"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss_object"]
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

        # if "mask" in batch:
        #     metrics_dict_object, images_dict_object = self.get_image_metrics_and_images_object(outputs, batch)
        #     metrics_dict.update(metrics_dict_object)
        #     images_dict.update(images_dict_object)

        return metrics_dict, images_dict


    def get_image_metrics_and_images_object(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
        ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        mask = batch["mask"].to(self.device)
        # expanded_mask = mask.unsqueeze(-1)
        gt_rgb = ~(mask) + (mask * image) # set background color white 1 to match output
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        predicted_rgb = outputs["rgb_object"]  # Blended with background (black if random background)
        acc = colormaps.apply_colormap(outputs["accumulation_object"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth_object"],
            accumulation=outputs["accumulation_object"],
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
        metrics_dict = {"psnr_object": float(psnr.item()), "ssim_object": float(ssim)}  # type: ignore
        
        metrics_dict["lpips_object"] = float(lpips)

        images_dict = {"img_object": combined_rgb, "accumulation_object": combined_acc, "depth_object": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_object_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation_object"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
