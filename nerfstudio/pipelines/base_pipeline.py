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
Abstracts for the Pipeline class.
"""
from __future__ import annotations

import typing
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast

import numpy as np
import numpy.typing as npt

import torch
import torch.distributed as dist
from PIL import Image
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from torch import nn
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.cameras.cameras import Cameras
# from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
    VanillaDataManager,
)
# from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler


def module_wrapper(ddp_or_model: Union[DDP, Model]) -> Model:
    """
    If DDP, then return the .module. Otherwise, return the model.
    """
    if isinstance(ddp_or_model, DDP):
        return cast(Model, ddp_or_model.module)
    return ddp_or_model


class Pipeline(nn.Module):
    """The intent of this class is to provide a higher level interface for the Model
    that will be easy to use for our Trainer class.

    This class will contain high level functions for the model like getting the loss
    dictionaries and visualization code. It should have ways to get the next iterations
    training loss, evaluation loss, and generate whole images for visualization. Each model
    class should be 1:1 with a pipeline that can act as a standardized interface and hide
    differences in how each model takes in and outputs data.

    This class's function is to hide the data manager and model classes from the trainer,
    worrying about:
    1) Fetching data with the data manager
    2) Feeding the model the data and fetching the loss
    Hopefully this provides a higher level interface for the trainer to use, and
    simplifying the model classes, which each may have different forward() methods
    and so on.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'train': loads train/eval datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    datamanager: DataManager
    _model: Model
    world_size: int

    @property
    def model(self):
        """Returns the unwrapped model if in ddp"""
        return module_wrapper(self._model)

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: Optional[bool] = None):
        is_ddp_model_state = True
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("_model."):
                # remove the "_model." prefix from key
                model_state[key[len("_model.") :]] = value
                # make sure that the "module." prefix comes from DDP,
                # rather than an attribute of the model named "module"
                if not key.startswith("_model.module."):
                    is_ddp_model_state = False
        # remove "module." prefix added by DDP
        if is_ddp_model_state:
            model_state = {key[len("module.") :]: value for key, value in model_state.items()}

        pipeline_state = {key: value for key, value in state_dict.items() if not key.startswith("_model.")}

        try:
            self.model.load_state_dict(model_state, strict=True)
        except RuntimeError:
            if not strict:
                self.model.load_state_dict(model_state, strict=False)
            else:
                raise

        super().load_state_dict(pipeline_state, strict=False)

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        if self.world_size > 1 and step:
            assert self.datamanager.train_sampler is not None
            self.datamanager.train_sampler.set_epoch(step)
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle, batch)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        if self.world_size > 1:
            assert self.datamanager.eval_sampler is not None
            self.datamanager.eval_sampler.set_epoch(step)
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle, batch)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @abstractmethod
    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """

    @abstractmethod
    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.
        """

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """

    @abstractmethod
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """


@dataclass
class VanillaPipelineConfig(cfg.InstantiateConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: VanillaPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = DataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = ModelConfig()
    """specifies the model config"""


class VanillaPipeline(Pipeline):
    """The pipeline class for the vanilla nerf setup of multiple cameras for one or a few scenes.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine
        grad_scaler: gradient scaler used in the trainer

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    def __init__(
        self,
        config: VanillaPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            cam_box=self.datamanager.train_dataset.cam_box,
            photogrametry_pc_box=self.datamanager.train_dataset.photogrametry_pc_box,
            N_max=self.datamanager.train_dataset.N_max,
            N_min=self.datamanager.train_dataset.N_min,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        if self.config.datamanager.camera_optimizer is not None:
            camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
            if camera_opt_param_group in self.datamanager.get_param_groups():
                # Report the camera optimization metrics
                metrics_dict["camera_opt_translation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
                )
                metrics_dict["camera_opt_rotation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
                )

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    @profiler.time_function
    def get_eval_loss_dict(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, VanillaDataManager)
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera_ray_bundle, batch in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()
                height, width = camera_ray_bundle.shape
                num_rays = height * width
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)

                if output_path is not None:
                    camera_indices = camera_ray_bundle.camera_indices
                    assert camera_indices is not None
                    for key, val in images_dict.items():
                        Image.fromarray((val * 255).byte().cpu().numpy()).save(
                            output_path / "{0:06d}-{1}.jpg".format(int(camera_indices[0, 0, 0]), key)
                        )
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )
        self.train()
        return metrics_dict

    @profiler.time_function
    def get_average_eval_image_metrics_compo(
        self, 
        step: Optional[int] = None, 
        output_path: Optional[Path] = None, 
        get_std: bool = False,
        cameras: Optional[Cameras] = None,
        image_paths: Optional[List[Path]] = None,
        object_pipelines_list: List[Pipeline] = None,
        object_cameras_list: List[Cameras] = None,
        aabb_only: bool = False,
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, VanillaDataManager)
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        
        if cameras:
            cameras = cameras.to(self.device)
            num_images = cameras.size
        
        if object_pipelines_list:
            dataparser_outputs = self.datamanager.dataparser.get_dataparser_outputs()
            scene_scale = dataparser_outputs.dataparser_scale
            scene_transform = dataparser_outputs.dataparser_transform # 3, 4
            
            object_dataparser_outputs_list = []
            object_models_list = []
            scene_object_boxes_list = []
            
            for object_pipeline in object_pipelines_list:
                object_dataparser_outputs = object_pipeline.datamanager.dataparser.get_dataparser_outputs()
                
                object_dataparser_outputs_list.append(object_dataparser_outputs)
                object_model = object_pipeline.model
                object_models_list.append(object_model)

                object_photogrametry_pc_box = object_model.photogrametry_pc_box
                point_min = torch.cat((object_photogrametry_pc_box.aabb[0], torch.tensor([1]))).unsqueeze(-1)
                point_max = torch.cat((object_photogrametry_pc_box.aabb[1], torch.tensor([1]))).unsqueeze(-1)
                transformed_point_min = scene_transform @ point_min
                transformed_point_max = scene_transform @ point_max
                scene_object_aabb = torch.cat((transformed_point_min.T, transformed_point_max.T), dim = 0)
                scene_object_aabb *= scene_scale
                scene_object_box = SceneBox(aabb = scene_object_aabb)
                scene_object_boxes_list.append(scene_object_box)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            
            # for camera_idx in progress.track(range(cameras.size), description=""):  
            for camera_idx in range(cameras.size):  
                camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx, keep_shape=True)

                image_path = image_paths[camera_idx]
                image_name = image_path.name
                pil_image = Image.open(image_path)
                image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
                if len(image.shape) == 2:
                    image = image[:, :, None].repeat(3, axis=2)
                assert len(image.shape) == 3
                assert image.dtype == np.uint8
                assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."

                image = torch.from_numpy(image.astype("float32") / 255.0)
                batch = {"image_idx": camera_idx, "image": image}

                if object_cameras_list:
                    object_camera_ray_bundles_list = []
                    for object_cameras in object_cameras_list:
                        object_cameras = object_cameras.to(self.device)
                        object_camera_ray_bundle = object_cameras.generate_rays(camera_indices=camera_idx, keep_shape=True)
                        object_camera_ray_bundles_list.append(object_camera_ray_bundle)

                inner_start = time()
                height, width = camera_ray_bundle.shape
                num_rays = height * width

                if object_pipelines_list:
                    outputs = self.model.get_outputs_for_camera_ray_bundle(
                        camera_ray_bundle, 
                        object_camera_ray_bundles_list=object_camera_ray_bundles_list,
                        object_models_list=object_models_list,
                        dataparser_outputs=dataparser_outputs, 
                        object_dataparser_outputs_list=object_dataparser_outputs_list,
                        scene_object_boxes_list=scene_object_boxes_list,
                    )
                else:
                    outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

                metrics_dict, images_dict = self.model.get_image_metrics_and_images_compo(outputs, batch, aabb_only=aabb_only)

                if output_path is not None:
                    camera_indices = camera_ray_bundle.camera_indices
                    assert camera_indices is not None
                    for key, val in images_dict.items():
                        Image.fromarray((val * 255).byte().cpu().numpy()).save(
                            # output_path / "{0:06d}-{1}.jpg".format(int(camera_indices[0, 0, 0]), key)
                            output_path / image_name
                        )
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )
        self.train()
        return metrics_dict

    @profiler.time_function
    def get_average_eval_image_metrics_compo_aabb(
        self, 
        step: Optional[int] = None, 
        output_path: Optional[Path] = None, 
        get_std: bool = False,
        cameras: Optional[Cameras] = None,
        image_paths: Optional[List[Path]] = None,
        object_pipelines_list: List[Pipeline] = None,
        object_cameras_list: List[Cameras] = None,
        aabb_only: bool = False,
        compo_outputs_list: List[Dict[str, torch.Tensor]] = None,
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, VanillaDataManager)
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        
        if cameras:
            cameras = cameras.to(self.device)
            num_images = cameras.size
        
        if object_pipelines_list:
            dataparser_outputs = self.datamanager.dataparser.get_dataparser_outputs()
            scene_scale = dataparser_outputs.dataparser_scale
            scene_transform = dataparser_outputs.dataparser_transform # 3, 4
            
            object_dataparser_outputs_list = []
            object_models_list = []
            scene_object_boxes_list = []
            
            for object_pipeline in object_pipelines_list:
                object_dataparser_outputs = object_pipeline.datamanager.dataparser.get_dataparser_outputs()
                
                object_dataparser_outputs_list.append(object_dataparser_outputs)
                object_model = object_pipeline.model
                object_models_list.append(object_model)

                object_photogrametry_pc_box = object_model.photogrametry_pc_box
                point_min = torch.cat((object_photogrametry_pc_box.aabb[0], torch.tensor([1]))).unsqueeze(-1)
                point_max = torch.cat((object_photogrametry_pc_box.aabb[1], torch.tensor([1]))).unsqueeze(-1)
                transformed_point_min = scene_transform @ point_min
                transformed_point_max = scene_transform @ point_max
                scene_object_aabb = torch.cat((transformed_point_min.T, transformed_point_max.T), dim = 0)
                scene_object_aabb *= scene_scale
                scene_object_box = SceneBox(aabb = scene_object_aabb)
                scene_object_boxes_list.append(scene_object_box)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            
            # for camera_idx in progress.track(range(cameras.size), description=""):  
            outputs_list =[]
            for camera_idx in range(cameras.size):  
                camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx, keep_shape=True)

                image_path = image_paths[camera_idx]
                image_name = image_path.name
                pil_image = Image.open(image_path)
                image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
                if len(image.shape) == 2:
                    image = image[:, :, None].repeat(3, axis=2)
                assert len(image.shape) == 3
                assert image.dtype == np.uint8
                assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."

                image = torch.from_numpy(image.astype("float32") / 255.0)
                batch = {"image_idx": camera_idx, "image": image}

                if object_cameras_list:
                    object_camera_ray_bundles_list = []
                    for object_cameras in object_cameras_list:
                        object_cameras = object_cameras.to(self.device)
                        object_camera_ray_bundle = object_cameras.generate_rays(camera_indices=camera_idx, keep_shape=True)
                        object_camera_ray_bundles_list.append(object_camera_ray_bundle)

                inner_start = time()
                height, width = camera_ray_bundle.shape
                num_rays = height * width

                if object_pipelines_list:
                    outputs = self.model.get_outputs_for_camera_ray_bundle(
                        camera_ray_bundle, 
                        object_camera_ray_bundles_list=object_camera_ray_bundles_list,
                        object_models_list=object_models_list,
                        dataparser_outputs=dataparser_outputs, 
                        object_dataparser_outputs_list=object_dataparser_outputs_list,
                        scene_object_boxes_list=scene_object_boxes_list,
                    )
                else:
                    outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

                outputs_list.append(outputs)
                if compo_outputs_list is not None:
                    metrics_dict, images_dict = self.model.get_image_metrics_and_images_compo(outputs, batch, aabb_only=aabb_only, compo_outputs=compo_outputs_list[camera_idx])
                else:
                    metrics_dict, images_dict = self.model.get_image_metrics_and_images_compo(outputs, batch, aabb_only=aabb_only)

                if output_path is not None:
                    camera_indices = camera_ray_bundle.camera_indices
                    assert camera_indices is not None
                    for key, val in images_dict.items():
                        Image.fromarray((val * 255).byte().cpu().numpy()).save(
                            # output_path / "{0:06d}-{1}.jpg".format(int(camera_indices[0, 0, 0]), key)
                            output_path / image_name
                        )
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )
        self.train()
        return metrics_dict, outputs_list

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)
        self.load_state_dict(state)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups()
        # TODO(ethan): assert that key names don't overlap
        return {**datamanager_params, **model_params}
