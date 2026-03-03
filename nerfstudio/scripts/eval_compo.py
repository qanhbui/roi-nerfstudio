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

#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List, Literal
from typing_extensions import Annotated

import tyro

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.cameras.camera_paths import (
    get_camera_from_json,
    get_eval_from_json
)

import hydra
from hydra.core.global_hydra import GlobalHydra

@dataclass
class ComputePSNR:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Optional[Path] = None 
    # Name of the output file.
    output_path: Path = Path("output.json")
    # Optional path to save rendered outputs to.
    render_output_path: Optional[Path] = None
    # Load configurations .yaml file using Hydra.
    load_hydra_configs: bool = False

    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)
        assert self.output_path.suffix == ".json"
        if self.render_output_path is not None:
            self.render_output_path.mkdir(parents=True)
        metrics_dict = pipeline.get_average_eval_image_metrics(output_path=self.render_output_path, get_std=True)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f"Saved results to: {self.output_path}")

@dataclass
class ComputePSNR_compo:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Optional[Path] = None 
    # Path to config YAML file of objects.
    load_objects_configs: Optional[List[str]] = None
    # Filename of the camera path to render.
    camera_path_filenames: Optional[List[str]] = None
    # Name of the output file.
    output_path: Path = Path("output.json")
    # Optional path to save rendered outputs to.
    render_output_path: Optional[Path] = None
    # Load configurations .yaml file using Hydra.
    load_hydra_configs: bool = False

    def main(self) -> None:
        """Main function."""

        pipelines_list = []

        config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)

        # iter through all test sets
        if self.camera_path_filenames:
            camera_meta_list = []
            for camera_path_filename in self.camera_path_filenames:
                with open(camera_path_filename, "r", encoding="utf-8") as f:
                    camera_path_meta = json.load(f)
                camera_meta_list.append(camera_path_meta)
                
            camera_filename = Path(camera_path_filename)
            data_dir = camera_filename.parent

            # transform of scene NeRF
            dataparser_outputs =  pipeline.datamanager.dataparser.get_dataparser_outputs()
            applied_transform = dataparser_outputs.dataparser_transform
            applied_scale = dataparser_outputs.dataparser_scale

            camera_eval_list = []
            image_paths_list = []
            for camera_meta in camera_meta_list:
                camera_eval, image_paths = get_eval_from_json(camera_meta, data_dir, applied_transform, applied_scale)
                camera_eval_list.append(camera_eval)
                image_paths_list.append(image_paths)

            if self.load_objects_configs:
                object_pipelines_list = []
                for object_config in self.load_objects_configs:
                    object_config_path = Path(object_config)
                    _, object_pipeline, _, _ = eval_setup(
                        object_config_path
                    )
                    object_pipelines_list.append(object_pipeline)

            
                object_camera_list = [] # list of items for each test set
                # object_image_path_list = []
                for camera_meta in camera_meta_list: # iter test set
                    roi_camera_eval_list = [] # list of camera of roi object
                    # roi_image_paths_list = []
                    for object_pipeline in object_pipelines_list: # iter multiple roi for each test set
                        object_dataparser_outputs =  object_pipeline.datamanager.dataparser.get_dataparser_outputs()
                        object_applied_transform = object_dataparser_outputs.dataparser_transform
                        object_applied_scale = object_dataparser_outputs.dataparser_scale

                        roi_camera_eval, _ = get_eval_from_json(camera_meta, data_dir, object_applied_transform, object_applied_scale)
                        roi_camera_eval_list.append(roi_camera_eval)
                        # roi_image_paths_list.append(roi_image_paths)
                    
                    object_camera_list.append(roi_camera_eval_list)
                    # object_image_path_list.append(roi_image_paths_list)
                    # object_image_path_list.append(roi_image_paths)
        

        assert self.output_path.suffix == ".json"
        if self.render_output_path is not None:
            self.render_output_path.mkdir(parents=True, exist_ok=True)
        
        if self.camera_path_filenames and self.load_objects_configs:
            metrics_dict = pipeline.get_average_eval_image_metrics_compo(
                output_path=self.render_output_path, 
                get_std=True,
                cameras=camera_eval,
                image_paths=image_paths,
                object_pipelines_list=object_pipelines_list,
                object_cameras_list=roi_camera_eval_list,
            )
        elif self.camera_path_filenames :
            metrics_dict = pipeline.get_average_eval_image_metrics_compo(
                output_path=self.render_output_path, 
                get_std=True,
                cameras=camera_eval,
                image_paths=image_paths,
            )
        else:
            metrics_dict = pipeline.get_average_eval_image_metrics(output_path=self.render_output_path, get_std=True)
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f"Saved results to: {self.output_path}")

@dataclass
class ComputePSNR_compo_full:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Optional[Path] = None 
    # Path to config YAML file of objects.
    load_objects_configs: Optional[List[str]] = None
    # Filename of the camera path to render.
    camera_path_filenames: Optional[List[str]] = None
    # Name of the output file.
    output_path: Optional[Path] = None 
    # Optional path to save rendered outputs of multiple models.
    render_output_path: Optional[Path] = None
    # Optional path to save rendered outputs of multiple models.
    eval_mode: Literal["base", "single", "full_rois", "single_aabb", "full_aabb", "all"] = "base"
    # Load configurations .yaml file using Hydra.
    load_hydra_configs: bool = False
    

    def main(self) -> None:
        """Main function."""
        # List of Scene models
        fb_str = "full_big"
        fh_str = "full_huge"
        sb_str = "scene_big"
        sh_str = "scene_huge"
        baselines = [fb_str, fh_str, sb_str, sh_str]
        scene_pipelines_dict = {}
        for scene_config_path in self.load_config:
            scene_config_path = Path(scene_config_path)
            _, scene_pipeline, _, _ = eval_setup(scene_config_path)
            for baseline_str in baselines:
                if baseline_str in str(scene_config_path):
                    scene_pipelines_dict[baseline_str] = scene_pipeline
                    break  

        object_pipelines_dict = {}
        for roi_index, object_config in enumerate(self.load_objects_configs):
            object_config_path = Path(object_config)
            _, object_pipeline, _, _ = eval_setup(object_config_path)

            # object_pipelines_dict[f"roi_{roi_index+1}"] = object_pipeline
            object_pipelines_dict[f"roi_{roi_index+3}"] = object_pipeline

        test_camera_dict = {}
        for test_index, camera_path_filename in enumerate(self.camera_path_filenames):
            with open(camera_path_filename, "r", encoding="utf-8") as f:
                camera_path_meta = json.load(f)
            
            # test_camera_dict[f"test_{test_index+1}"] = camera_path_meta
            test_camera_dict[f"test_{test_index+3}"] = camera_path_meta
        
        # Get dir of image data 
        camera_filename = Path(camera_path_filename)
        image_data_dir = camera_filename.parent
        # Check and prepare output directories
        if self.output_path is not None and not self.output_path.exists():
            self.output_path.mkdir(parents=True, exist_ok=True)

        if self.render_output_path is not None and not self.render_output_path.exists():
            self.render_output_path.mkdir(parents=True, exist_ok=True)

        # A: NO compo
        if self.eval_mode in ["base", "all"]:
            for indice in range(1, len(self.camera_path_filenames) + 1):
                # Test cams
                test_camera_key = f"test_{indice}"
                camera_meta = test_camera_dict[test_camera_key]
                # ROI
                roi_key = f"roi_{indice}"
                object_pipeline = object_pipelines_dict[roi_key]
                # transform of ROI NeRF
                object_dataparser_outputs =  object_pipeline.datamanager.dataparser.get_dataparser_outputs()
                object_applied_transform = object_dataparser_outputs.dataparser_transform
                object_applied_scale = object_dataparser_outputs.dataparser_scale
                roi_camera_eval, image_paths = get_eval_from_json(camera_meta, image_data_dir, object_applied_transform, object_applied_scale)
                
                object_render_output_path = self.render_output_path / f'test{indice}_roi'
                if not object_render_output_path.exists():
                    object_render_output_path.mkdir(parents=True, exist_ok=True)

                object_metrics_dict = object_pipeline.get_average_eval_image_metrics_compo(
                    output_path=object_render_output_path, 
                    get_std=True,
                    cameras=roi_camera_eval,
                    image_paths=image_paths,
                )
                benchmark_info = {"results": object_metrics_dict}
                # Save output to output file
                object_output_path = self.output_path / f'test{indice}_roi.json'
                object_output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
                CONSOLE.print(f"Saved results to: {object_output_path}")
                # Scene
                for baseline_key in scene_pipelines_dict:
                    pipeline = scene_pipelines_dict[baseline_key]
                    # transform of scene NeRF
                    dataparser_outputs =  pipeline.datamanager.dataparser.get_dataparser_outputs()
                    applied_transform = dataparser_outputs.dataparser_transform
                    applied_scale = dataparser_outputs.dataparser_scale

                    # eval infos for Scene
                    scene_camera_eval, _ = get_eval_from_json(camera_meta, image_data_dir, applied_transform, applied_scale)
                    scene_render_output_path = self.render_output_path / f'test{indice}_{baseline_key}'
                    if not scene_render_output_path.exists():
                        scene_render_output_path.mkdir(parents=True, exist_ok=True)

                    scene_metrics_dict = pipeline.get_average_eval_image_metrics_compo(
                        output_path=scene_render_output_path, 
                        get_std=True,
                        cameras=scene_camera_eval,
                        image_paths=image_paths,
                    )
                    benchmark_info = {"results": scene_metrics_dict}
                    # Save output to output file
                    scene_output_path = self.output_path / f'test{indice}_{baseline_key}.json'
                    scene_output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
                    CONSOLE.print(f"Saved results to: {scene_output_path}")

        # B: Compo single
        if self.eval_mode in ["single", "all"]:
            for indice in range(1, len(self.camera_path_filenames) + 1):
                # Test cams
                indice += 2
                test_camera_key = f"test_{indice}"
                camera_meta = test_camera_dict[test_camera_key]

                object_pipelines_list = []
                roi_camera_eval_list = []
                # ROI
                roi_key = f"roi_{indice}"
                object_pipeline = object_pipelines_dict[roi_key]
                object_pipelines_list.append(object_pipeline)
                # transform of ROI NeRF
                object_dataparser_outputs =  object_pipeline.datamanager.dataparser.get_dataparser_outputs()
                object_applied_transform = object_dataparser_outputs.dataparser_transform
                object_applied_scale = object_dataparser_outputs.dataparser_scale

                roi_camera_eval, image_paths = get_eval_from_json(camera_meta, image_data_dir, object_applied_transform, object_applied_scale)
                roi_camera_eval_list.append(roi_camera_eval)
                
                # Scene
                for baseline_key in scene_pipelines_dict:
                    pipeline = scene_pipelines_dict[baseline_key]
                    # transform of scene NeRF
                    dataparser_outputs =  pipeline.datamanager.dataparser.get_dataparser_outputs()
                    applied_transform = dataparser_outputs.dataparser_transform
                    applied_scale = dataparser_outputs.dataparser_scale

                    # eval infos for Scene
                    scene_camera_eval, _ = get_eval_from_json(camera_meta, image_data_dir, applied_transform, applied_scale)
                    scene_render_output_path = self.render_output_path / f'test{indice}_compo_{baseline_key}'
                    if not scene_render_output_path.exists():
                        scene_render_output_path.mkdir(parents=True, exist_ok=True)

                    scene_metrics_dict = pipeline.get_average_eval_image_metrics_compo(
                        output_path=scene_render_output_path, 
                        get_std=True,
                        cameras=scene_camera_eval,
                        image_paths=image_paths,
                        object_pipelines_list=object_pipelines_list,
                        object_cameras_list=roi_camera_eval_list,
                    )
                    benchmark_info = {"results": scene_metrics_dict}
                    # Save output to output file
                    scene_output_path = self.output_path / f'test{indice}_compo_{baseline_key}.json'
                    scene_output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
                    CONSOLE.print(f"Saved results to: {scene_output_path}")

        # C: Compo full
        if self.eval_mode in ["full_rois", "all"]:
            for indice in range(1, len(self.camera_path_filenames) + 1):
                indice += 2 # remove roi 1,2 begin with roi3
                # Test cams
                test_camera_key = f"test_{indice}"
                camera_meta = test_camera_dict[test_camera_key]

                object_pipelines_list = []
                roi_camera_eval_list = []
                # ROI
                for roi_key in object_pipelines_dict.keys():
                    object_pipeline = object_pipelines_dict[roi_key]
                    object_pipelines_list.append(object_pipeline)
                    # transform of ROI NeRF
                    object_dataparser_outputs =  object_pipeline.datamanager.dataparser.get_dataparser_outputs()
                    object_applied_transform = object_dataparser_outputs.dataparser_transform
                    object_applied_scale = object_dataparser_outputs.dataparser_scale

                    roi_camera_eval, image_paths = get_eval_from_json(camera_meta, image_data_dir, object_applied_transform, object_applied_scale)
                    roi_camera_eval_list.append(roi_camera_eval)
                
                # Scene
                for baseline_key in scene_pipelines_dict:
                    pipeline = scene_pipelines_dict[baseline_key]
                    # transform of scene NeRF
                    dataparser_outputs =  pipeline.datamanager.dataparser.get_dataparser_outputs()
                    applied_transform = dataparser_outputs.dataparser_transform
                    applied_scale = dataparser_outputs.dataparser_scale

                    # eval infos for Scene
                    scene_camera_eval, _ = get_eval_from_json(camera_meta, image_data_dir, applied_transform, applied_scale)
                    scene_render_output_path = self.render_output_path / f'test{indice}_compo{len(self.load_objects_configs)}_{baseline_key}'
                    if not scene_render_output_path.exists():
                        scene_render_output_path.mkdir(parents=True, exist_ok=True)

                    scene_metrics_dict = pipeline.get_average_eval_image_metrics_compo(
                        output_path=scene_render_output_path, 
                        get_std=True,
                        cameras=scene_camera_eval,
                        image_paths=image_paths,
                        object_pipelines_list=object_pipelines_list,
                        object_cameras_list=roi_camera_eval_list,
                    )
                    benchmark_info = {"results": scene_metrics_dict}
                    # Save output to output file
                    scene_output_path = self.output_path / f'test{indice}_compo{len(self.load_objects_configs)}_{baseline_key}.json'
                    scene_output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
                    CONSOLE.print(f"Saved results to: {scene_output_path}")

        # D: Compo single in AABB
        if self.eval_mode in ["single_aabb"]:
            for indice in range(1, len(self.camera_path_filenames) + 1):
                # Test cams
                indice += 2
                test_camera_key = f"test_{indice}"
                camera_meta = test_camera_dict[test_camera_key]

                object_pipelines_list = []
                roi_camera_eval_list = []
                # ROI
                roi_key = f"roi_{indice}"
                object_pipeline = object_pipelines_dict[roi_key]
                object_pipelines_list.append(object_pipeline)
                # transform of ROI NeRF
                object_dataparser_outputs =  object_pipeline.datamanager.dataparser.get_dataparser_outputs()
                object_applied_transform = object_dataparser_outputs.dataparser_transform
                object_applied_scale = object_dataparser_outputs.dataparser_scale

                roi_camera_eval, image_paths = get_eval_from_json(camera_meta, image_data_dir, object_applied_transform, object_applied_scale)
                roi_camera_eval_list.append(roi_camera_eval)
                
                # Scene
                for baseline_key in scene_pipelines_dict:
                    pipeline = scene_pipelines_dict[baseline_key]
                    # transform of scene NeRF
                    dataparser_outputs =  pipeline.datamanager.dataparser.get_dataparser_outputs()
                    applied_transform = dataparser_outputs.dataparser_transform
                    applied_scale = dataparser_outputs.dataparser_scale

                    # eval infos for Scene
                    scene_camera_eval, _ = get_eval_from_json(camera_meta, image_data_dir, applied_transform, applied_scale)
                    # Compute compo
                    compo_render_output_path = self.render_output_path / f'test{indice}_compo_aabb_{baseline_key}'
                    if not compo_render_output_path.exists():
                        compo_render_output_path.mkdir(parents=True, exist_ok=True)

                    compo_metrics_dict, compo_outputs_list = pipeline.get_average_eval_image_metrics_compo_aabb(
                        output_path=compo_render_output_path, 
                        get_std=True,
                        cameras=scene_camera_eval,
                        image_paths=image_paths,
                        object_pipelines_list=object_pipelines_list,
                        object_cameras_list=roi_camera_eval_list,
                        aabb_only=True,
                    )
                    benchmark_info = {"results": compo_metrics_dict}
                    # Save output to output file
                    compo_output_path = self.output_path / f'test{indice}_compo_aabb_{baseline_key}.json'
                    compo_output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
                    CONSOLE.print(f"Saved results compo to: {compo_output_path}")

                    # Compute baselines
                    base_render_output_path = self.render_output_path / f'test{indice}_aabb_{baseline_key}'
                    if not base_render_output_path.exists():
                        base_render_output_path.mkdir(parents=True, exist_ok=True)

                    base_metrics_dict, _ = pipeline.get_average_eval_image_metrics_compo_aabb(
                        output_path=base_render_output_path, 
                        get_std=True,
                        cameras=scene_camera_eval,
                        image_paths=image_paths,
                        aabb_only=True,
                        compo_outputs_list=compo_outputs_list
                    )
                    benchmark_info = {"results": base_metrics_dict}
                    # Save output to output file
                    base_output_path = self.output_path / f'test{indice}_aabb_{baseline_key}.json'
                    base_output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
                    CONSOLE.print(f"Saved results baseline aabb to: {base_output_path}")

                object_render_output_path = self.render_output_path / f'test{indice}_aabb_roi'
                if not object_render_output_path.exists():
                    object_render_output_path.mkdir(parents=True, exist_ok=True)

                object_metrics_dict, _ = object_pipeline.get_average_eval_image_metrics_compo_aabb(
                    output_path=object_render_output_path, 
                    get_std=True,
                    cameras=roi_camera_eval,
                    image_paths=image_paths,
                    aabb_only=True,
                    compo_outputs_list=compo_outputs_list
                )
                benchmark_info = {"results": object_metrics_dict}
                # Save output to output file
                object_output_path = self.output_path / f'test{indice}_aabb_roi.json'
                object_output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
                CONSOLE.print(f"Saved results roi to: {object_output_path}")

        # E: Compo full in AABB
        if self.eval_mode in ["full_aabb"]:
            for indice in range(1, len(self.camera_path_filenames) + 1):
                indice += 2 # remove roi 1,2 begin with roi3
                # Test cams
                test_camera_key = f"test_{indice}"
                camera_meta = test_camera_dict[test_camera_key]

                object_pipelines_list = []
                roi_camera_eval_list = []
                # ROI
                for roi_key in object_pipelines_dict.keys():
                    object_pipeline = object_pipelines_dict[roi_key]
                    object_pipelines_list.append(object_pipeline)
                    # transform of ROI NeRF
                    object_dataparser_outputs =  object_pipeline.datamanager.dataparser.get_dataparser_outputs()
                    object_applied_transform = object_dataparser_outputs.dataparser_transform
                    object_applied_scale = object_dataparser_outputs.dataparser_scale

                    roi_camera_eval, image_paths = get_eval_from_json(camera_meta, image_data_dir, object_applied_transform, object_applied_scale)
                    roi_camera_eval_list.append(roi_camera_eval)
                
                # Scene
                for baseline_key in scene_pipelines_dict:
                    pipeline = scene_pipelines_dict[baseline_key]
                    # transform of scene NeRF
                    dataparser_outputs =  pipeline.datamanager.dataparser.get_dataparser_outputs()
                    applied_transform = dataparser_outputs.dataparser_transform
                    applied_scale = dataparser_outputs.dataparser_scale

                    # eval infos for Scene
                    scene_camera_eval, _ = get_eval_from_json(camera_meta, image_data_dir, applied_transform, applied_scale)
                    # Compute compo
                    compo_render_output_path = self.render_output_path / f'test{indice}_compo{len(self.load_objects_configs)}_aabb_{baseline_key}'
                    if not compo_render_output_path.exists():
                        compo_render_output_path.mkdir(parents=True, exist_ok=True)

                    compo_metrics_dict, compo_outputs_list = pipeline.get_average_eval_image_metrics_compo_aabb(
                        output_path=compo_render_output_path, 
                        get_std=True,
                        cameras=scene_camera_eval,
                        image_paths=image_paths,
                        object_pipelines_list=object_pipelines_list,
                        object_cameras_list=roi_camera_eval_list,
                        aabb_only=True,
                    )
                    benchmark_info = {"results": compo_metrics_dict}
                    # Save output to output file
                    compo_output_path = self.output_path / f'test{indice}_compo{len(self.load_objects_configs)}_aabb_{baseline_key}.json'
                    compo_output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
                    CONSOLE.print(f"Saved results compo to: {compo_output_path}")

                    # Compute baselines
                    base_render_output_path = self.render_output_path / f'test{indice}_{len(self.load_objects_configs)}_aabb_{baseline_key}'
                    if not base_render_output_path.exists():
                        base_render_output_path.mkdir(parents=True, exist_ok=True)

                    base_metrics_dict, _ = pipeline.get_average_eval_image_metrics_compo_aabb(
                        output_path=base_render_output_path, 
                        get_std=True,
                        cameras=scene_camera_eval,
                        image_paths=image_paths,
                        aabb_only=True,
                        compo_outputs_list=compo_outputs_list
                    )
                    benchmark_info = {"results": base_metrics_dict}
                    # Save output to output file
                    base_output_path = self.output_path / f'test{indice}_{len(self.load_objects_configs)}_aabb_{baseline_key}.json'
                    base_output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
                    CONSOLE.print(f"Saved results baseline aabb to: {base_output_path}")

                # object_pipeline = object_pipelines_list[indice-1]
                # roi_camera_eval = roi_camera_eval_list[indice-1]
                object_pipeline = object_pipelines_list[indice-3]
                roi_camera_eval = roi_camera_eval_list[indice-3]
                object_render_output_path = self.render_output_path / f'test{indice}_{len(self.load_objects_configs)}_aabb_roi'
                if not object_render_output_path.exists():
                    object_render_output_path.mkdir(parents=True, exist_ok=True)

                object_metrics_dict, _ = object_pipeline.get_average_eval_image_metrics_compo_aabb(
                    output_path=object_render_output_path, 
                    get_std=True,
                    cameras=roi_camera_eval,
                    image_paths=image_paths,
                    aabb_only=True,
                    compo_outputs_list=compo_outputs_list
                )
                benchmark_info = {"results": object_metrics_dict}
                # Save output to output file
                object_output_path = self.output_path / f'test{indice}_aabb_roi.json'
                object_output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
                CONSOLE.print(f"Saved results roi to: {object_output_path}")


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ComputePSNR, tyro.conf.subcommand(name="scores")],
        Annotated[ComputePSNR_compo, tyro.conf.subcommand(name="compo")],
        Annotated[ComputePSNR_compo_full, tyro.conf.subcommand(name="benchmark")],
    ]
]

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    eval_args = tyro.cli(Commands)

    if eval_args.load_hydra_configs:
        # Clear the existing GlobalHydra instance, if any
        GlobalHydra.instance().clear()

        hydra.initialize(version_base=None, config_path="conf", job_name="eval") # init like @hydra.main()
        cfg = hydra.compose("eval_configs")

        for key in cfg.keys():
            setattr(eval_args, key, cfg[key])
        eval_args.main()

    else:
        tyro.cli(Commands).main()

if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Commands)  # noqa
