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
render.py
"""
from __future__ import annotations

import json
import os
import struct
import shutil
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import mediapy as media
import numpy as np
import torch
import cv2
import tyro
from jaxtyping import Float
from rich import box, style
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch import Tensor
from typing_extensions import Annotated

import open3d as o3d

from nerfstudio.cameras.camera_paths import (
    get_interpolated_camera_path,
    get_path_from_json,
    change_ref_transform_photogrametry,
    get_spiral_path,
    get_camera_from_json,
)
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.model_components import renderers
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import colormaps, install_checks
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from nerfstudio.utils.scripts import run_command
from nerfstudio.utils.io import load_from_json

def get_mean_and_std(x):
    x_mean, x_std = cv2.meanStdDev(x)
    x_mean = np.hstack(x_mean)
    x_std = np.hstack(x_std)
    return x_mean, x_std

def _render_trajectory_video(
    pipeline: Pipeline,
    cameras: Cameras,
    output_filename: Path,
    rendered_output_names: List[str],
    crop_data: Optional[CropData] = None,
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
    output_format: Literal["images", "video"] = "video",
    image_format: Literal["jpeg", "png"] = "jpeg",
    jpeg_quality: int = 100,
    colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions(),
    origin_name: bool = False,
    cameras_filenames: List = None,
    object_pipelines_list: List[Pipeline] = None,
    object_cameras_list: List[Cameras] = None,
    apply_color_transfer: bool = False,
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_names: List of outputs to visualise.
        crop_data: Crop data to apply to the rendered images.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Length of output video.
        output_format: How to save output data.
        colormap_options: Options for colormap.
    """
    CONSOLE.print("[bold green]Creating trajectory " + output_format)
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    cameras = cameras.to(pipeline.device)
    if object_cameras_list:
        for index, object_cameras in enumerate(object_cameras_list):
            object_cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
            object_cameras = object_cameras.to(pipeline.device)
            object_cameras_list[index] = object_cameras

    fps = len(cameras) / seconds

    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(
            text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
            show_speed=True,
        ),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=False, compact=False),
        TimeElapsedColumn(),
    )
    output_image_dir = output_filename.parent / output_filename.stem
    if output_format == "images":
        output_image_dir.mkdir(parents=True, exist_ok=True)
    if output_format == "video":
        # make the folder if it doesn't exist
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        # NOTE:
        # we could use ffmpeg_args "-movflags faststart" for progressive download,
        # which would force moov atom into known position before mdat,
        # but then we would have to move all of mdat to insert metadata atom
        # (unless we reserve enough space to overwrite with our uuid tag,
        # but we don't know how big the video file will be, so it's not certain!)

    if object_pipelines_list:
        dataparser_outputs = pipeline.datamanager.dataparser.get_dataparser_outputs()
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

    with ExitStack() as stack:
        writer = None

        with progress:
            for camera_idx in progress.track(range(cameras.size), description=""):
                aabb_box = None
                if crop_data is not None:
                    bounding_box_min = crop_data.center - crop_data.scale / 2.0
                    bounding_box_max = crop_data.center + crop_data.scale / 2.0
                    aabb_box = SceneBox(torch.stack([bounding_box_min, bounding_box_max]).to(pipeline.device))
                camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx, aabb_box=aabb_box)
                if object_cameras_list:
                    object_camera_ray_bundles_list = []
                    for index, object_cameras in enumerate(object_cameras_list):
                        # object_camera_ray_bundle = object_cameras.generate_rays(camera_indices=camera_idx, aabb_box=scene_object_boxes_list[index])
                        object_camera_ray_bundle = object_cameras.generate_rays(camera_indices=camera_idx, aabb_box=aabb_box)
                        # object_camera_ray_bundle = object_cameras.generate_rays(camera_indices=camera_idx, aabb_box=object_models_list[index].collider_box)
                        # object_camera_ray_bundle = object_cameras.generate_rays(camera_indices=camera_idx, aabb_box=object_models_list[index].scene_box)
                        object_camera_ray_bundles_list.append(object_camera_ray_bundle)

                if crop_data is not None:
                    with renderers.background_color_override_context(
                        crop_data.background_color.to(pipeline.device)
                    ), torch.no_grad():
                        if object_pipelines_list:
                            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(
                                camera_ray_bundle, 
                                object_camera_ray_bundles_list=object_camera_ray_bundles_list,
                                object_models_list=object_models_list, 
                                dataparser_outputs=dataparser_outputs, 
                                object_dataparser_outputs_list=object_dataparser_outputs_list,
                                scene_object_boxes_list=scene_object_boxes_list
                            )
                        else: 
                            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                else:
                    with torch.no_grad():
                        if object_pipelines_list:
                            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(
                                camera_ray_bundle, 
                                object_camera_ray_bundles_list=object_camera_ray_bundles_list,
                                object_models_list=object_models_list,
                                dataparser_outputs=dataparser_outputs, 
                                object_dataparser_outputs_list=object_dataparser_outputs_list,
                                scene_object_boxes_list=scene_object_boxes_list
                            )
                        else:
                            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

                #NOTE: color transfer composition pixels to scene pixels
                if apply_color_transfer and "compo_selector" in outputs:
                    if outputs["compo_selector"].sum().item() > 0 :
                        compo_pixels = outputs["rgb"][outputs["compo_selector"].squeeze(-1)]
                        scene_pixels = outputs["compo_scene_rgb"][outputs["compo_selector"].squeeze(-1)]
                        # scene_pixels = outputs["rgb"][(~outputs["compo_selector"]).squeeze(-1)]

                        compo_pixels = compo_pixels.cpu().numpy()
                        scene_pixels = scene_pixels.cpu().numpy()

                        compo_pixels = compo_pixels * 255
                        scene_pixels = scene_pixels * 255

                        compo_pixels = np.expand_dims(compo_pixels, axis=0)
                        scene_pixels = np.expand_dims(scene_pixels, axis=0)

                        compo_lab = cv2.cvtColor(compo_pixels.astype(np.uint8),cv2.COLOR_RGB2LAB)
                        scene_lab = cv2.cvtColor(scene_pixels.astype(np.uint8),cv2.COLOR_RGB2LAB)

                        s_mean, s_std = get_mean_and_std(compo_lab)
                        t_mean, t_std = get_mean_and_std(scene_lab)

                        compo_lab_ct=((compo_lab-s_mean)*(t_std/s_std))+t_mean
                        compo_rgb_ct = cv2.cvtColor(cv2.convertScaleAbs(compo_lab_ct), cv2.COLOR_LAB2RGB)

                        compo_rgb_ct = np.squeeze(compo_rgb_ct, axis=0)
                        compo_rgb_ct = torch.from_numpy(compo_rgb_ct).to(outputs["rgb"].device).float()

                        compo_rgb_ct = compo_rgb_ct / 255
                        outputs["rgb"][outputs["compo_selector"].squeeze(-1)] = compo_rgb_ct

                render_image = []
                for rendered_output_name in rendered_output_names:
                    if rendered_output_name not in outputs:
                        CONSOLE.rule("Error", style="red")
                        CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                        CONSOLE.print(
                            f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center"
                        )
                        sys.exit(1)
                    output_image = outputs[rendered_output_name]
                    output_image = (
                        colormaps.apply_colormap(
                            image=output_image,
                            colormap_options=colormap_options,
                        )
                        .cpu()
                        .numpy()
                    )
                    render_image.append(output_image)
                render_image = np.concatenate(render_image, axis=1)
                if output_format == "images":
                    if image_format == "png":
                        if origin_name:
                            image_name = os.path.splitext(os.path.basename(pipeline.datamanager.all_dataset.image_filenames[camera_idx]))[0]
                            media.write_image(output_image_dir / f"{image_name}_depth.png", render_image, fmt="png")
                        elif cameras_filenames:
                            media.write_image(output_image_dir / f"{cameras_filenames[camera_idx]}.png", render_image, fmt="png")
                        
                        else:
                            media.write_image(output_image_dir / f"{camera_idx:05d}.png", render_image, fmt="png")
                    if image_format == "jpeg":
                        if origin_name:
                            image_name = os.path.splitext(os.path.basename(pipeline.datamanager.all_dataset.image_filenames[camera_idx]))[0]
                            media.write_image(output_image_dir / f"{image_name}_depth.jpg", render_image, fmt="jpeg", quality=jpeg_quality)
                        elif cameras_filenames:
                            media.write_image(
                                output_image_dir / f"{cameras_filenames[camera_idx]}.jpg", render_image, fmt="jpeg", quality=jpeg_quality
                            )
                        else:
                            media.write_image(
                                output_image_dir / f"{camera_idx:05d}.jpg", render_image, fmt="jpeg", quality=jpeg_quality
                            )
                if output_format == "video":
                    if writer is None:
                        render_width = int(render_image.shape[1])
                        render_height = int(render_image.shape[0])
                        writer = stack.enter_context(
                            media.VideoWriter(
                                path=output_filename,
                                shape=(render_height, render_width),
                                fps=fps,
                            )
                        )
                    writer.add_image(render_image)

    table = Table(
        title=None,
        show_header=False,
        box=box.MINIMAL,
        title_style=style.Style(bold=True),
    )
    if output_format == "video":
        if cameras.camera_type[0] == CameraType.EQUIRECTANGULAR.value:
            CONSOLE.print("Adding spherical camera data")
            insert_spherical_metadata_into_file(output_filename)
        table.add_row("Video", str(output_filename))
    else:
        table.add_row("Images", str(output_image_dir))
    CONSOLE.print(Panel(table, title="[bold][green]:tada: Render Complete :tada:[/bold]", expand=False))


def insert_spherical_metadata_into_file(
    output_filename: Path,
) -> None:
    """Inserts spherical metadata into MP4 video file in-place.
    Args:
        output_filename: Name of the (input and) output file.
    """
    # NOTE:
    # because we didn't use faststart, the moov atom will be at the end;
    # to insert our metadata, we need to find (skip atoms until we get to) moov.
    # we should have 0x00000020 ftyp, then 0x00000008 free, then variable mdat.
    spherical_uuid = b"\xff\xcc\x82\x63\xf8\x55\x4a\x93\x88\x14\x58\x7a\x02\x52\x1f\xdd"
    spherical_metadata = bytes(
        """<rdf:SphericalVideo
xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'
xmlns:GSpherical='http://ns.google.com/videos/1.0/spherical/'>
<GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType>
<GSpherical:Spherical>True</GSpherical:Spherical>
<GSpherical:Stitched>True</GSpherical:Stitched>
<GSpherical:StitchingSoftware>nerfstudio</GSpherical:StitchingSoftware>
</rdf:SphericalVideo>""",
        "utf-8",
    )
    insert_size = len(spherical_metadata) + 8 + 16
    with open(output_filename, mode="r+b") as mp4file:
        try:
            # get file size
            mp4file_size = os.stat(output_filename).st_size

            # find moov container (probably after ftyp, free, mdat)
            while True:
                pos = mp4file.tell()
                size, tag = struct.unpack(">I4s", mp4file.read(8))
                if tag == b"moov":
                    break
                mp4file.seek(pos + size)
            # if moov isn't at end, bail
            if pos + size != mp4file_size:
                # TODO: to support faststart, rewrite all stco offsets
                raise Exception("moov container not at end of file")
            # go back and write inserted size
            mp4file.seek(pos)
            mp4file.write(struct.pack(">I", size + insert_size))
            # go inside moov
            mp4file.seek(pos + 8)
            # find trak container (probably after mvhd)
            while True:
                pos = mp4file.tell()
                size, tag = struct.unpack(">I4s", mp4file.read(8))
                if tag == b"trak":
                    break
                mp4file.seek(pos + size)
            # go back and write inserted size
            mp4file.seek(pos)
            mp4file.write(struct.pack(">I", size + insert_size))
            # we need to read everything from end of trak to end of file in order to insert
            # TODO: to support faststart, make more efficient (may load nearly all data)
            mp4file.seek(pos + size)
            rest_of_file = mp4file.read(mp4file_size - pos - size)
            # go to end of trak (again)
            mp4file.seek(pos + size)
            # insert our uuid atom with spherical metadata
            mp4file.write(struct.pack(">I4s16s", insert_size, b"uuid", spherical_uuid))
            mp4file.write(spherical_metadata)
            # write rest of file
            mp4file.write(rest_of_file)
        finally:
            mp4file.close()


@dataclass
class CropData:
    """Data for cropping an image."""

    background_color: Float[Tensor, "3"] = torch.Tensor([0.0, 0.0, 0.0])
    """background color"""
    center: Float[Tensor, "3"] = torch.Tensor([0.0, 0.0, 0.0])
    """center of the crop"""
    scale: Float[Tensor, "3"] = torch.Tensor([2.0, 2.0, 2.0])
    """scale of the crop"""


def get_crop_from_json(camera_json: Dict[str, Any]) -> Optional[CropData]:
    """Load crop data from a camera path JSON

    args:
        camera_json: camera path data
    returns:
        Crop data
    """
    if "crop" not in camera_json or camera_json["crop"] is None:
        return None

    bg_color = camera_json["crop"]["crop_bg_color"]

    return CropData(
        background_color=torch.Tensor([bg_color["r"] / 255.0, bg_color["g"] / 255.0, bg_color["b"] / 255.0]),
        center=torch.Tensor(camera_json["crop"]["crop_center"]),
        scale=torch.Tensor(camera_json["crop"]["crop_scale"]),
    )


@dataclass
class BaseRender:
    """Base class for rendering."""

    # load_config: Path
    load_config: Optional[Path] = None
    """Path to config YAML file."""
    output_path: Path = Path("renders/output.mp4")
    """Path to output video file."""
    image_format: Literal["jpeg", "png"] = "jpeg"
    """Image format"""
    jpeg_quality: int = 100
    """JPEG quality"""
    downscale_factor: float = 1.0
    """Scaling factor to apply to the camera image resolution."""
    eval_num_rays_per_chunk: Optional[int] = None
    """Specifies number of rays per chunk during eval. If None, use the value in the config file."""
    colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions()
    """Colormap options."""
    load_objects_configs: Optional[List[str]] = None
    """Path to object config YAML file."""
    # load_objects_pcs: Optional[List[str]] = None
    # """Path to objects .ply Point Cloud to take their photogrametry reference AABBs."""
    apply_transforms_to_camera_poses: bool = False
    """Apply dataparser transforms to camera poses."""
    apply_color_transfer: bool = False
    """Apply color transfer."""
    load_hydra_configs: bool = False
    """Load configurations .yaml file using Hydra."""

@dataclass
class RenderCameraPath(BaseRender):
    """Render a camera path generated by the viewer or blender add-on."""

    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    """Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis"""
    camera_path_filename: Path = Path("camera_path.json")
    """Filename of the camera path to render."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""

    def main(self) -> None:
        """Main function."""
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
        )

        if self.load_objects_configs:
            object_pipelines_list = []
            for object_config in self.load_objects_configs:
                object_config_path = Path(object_config)
                _, object_pipeline, _, _ = eval_setup(
                    object_config_path,
                    eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
                    test_mode="inference",
                )
                object_pipelines_list.append(object_pipeline)

        install_checks.check_ffmpeg_installed()

        with open(self.camera_path_filename, "r", encoding="utf-8") as f:
            camera_path_meta = json.load(f)
        crop_data = get_crop_from_json(camera_path_meta)

        if self.apply_transforms_to_camera_poses:
            dataparser_outputs =  pipeline.datamanager.dataparser.get_dataparser_outputs()
            applied_transform = dataparser_outputs.dataparser_transform
            applied_scale = dataparser_outputs.dataparser_scale
            seconds = None
            
            camera_path, image_filenames = get_camera_from_json(camera_path_meta, applied_transform, applied_scale)
            
            if self.load_objects_configs:
                object_camera_path_list = []
                for object_pipeline in object_pipelines_list:
                    object_dataparser_outputs =  object_pipeline.datamanager.dataparser.get_dataparser_outputs()
                    object_applied_transform = object_dataparser_outputs.dataparser_transform
                    object_applied_scale = object_dataparser_outputs.dataparser_scale
                    
                    object_camera_path, _ = get_camera_from_json(camera_path_meta, object_applied_transform, object_applied_scale)
                    object_camera_path_list.append(object_camera_path)
        else:
            seconds = camera_path_meta["seconds"]
            camera_path = get_path_from_json(camera_path_meta)
            image_filenames = None

        if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value:
            # temp folder for writing left and right view renders
            temp_folder_path = self.output_path.parent / (self.output_path.stem + "_temp")

            Path(temp_folder_path).mkdir(parents=True, exist_ok=True)
            left_eye_path = temp_folder_path / "ods_render_Left.mp4"

            self.output_path = left_eye_path

            CONSOLE.print("[bold green]:goggles: Omni-directional Stereo VR :goggles:")
            CONSOLE.print("Rendering left eye view")

        # add mp4 suffix to video output if none is specified
        if self.output_format == "video" and str(self.output_path.suffix) == "":
            self.output_path = self.output_path.with_suffix(".mp4")
        
        #NOTE: recondition more clearly of apply_transforms, load_object_config
        if not self.apply_transforms_to_camera_poses and not self.load_objects_configs:
            _render_trajectory_video(
                pipeline,
                camera_path,
                output_filename=self.output_path,
                rendered_output_names=self.rendered_output_names,
                rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
                crop_data=crop_data,
                seconds=seconds,
                output_format=self.output_format,
                image_format=self.image_format,
                jpeg_quality=self.jpeg_quality,
                colormap_options=self.colormap_options,
            )
        elif self.apply_transforms_to_camera_poses and self.load_objects_configs:
            _render_trajectory_video(
                pipeline,
                camera_path,
                output_filename=self.output_path,
                rendered_output_names=self.rendered_output_names,
                rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
                crop_data=crop_data,
                # seconds=seconds,
                output_format=self.output_format,
                image_format=self.image_format,
                jpeg_quality=self.jpeg_quality,
                colormap_options=self.colormap_options,
                cameras_filenames = image_filenames,
                object_pipelines_list=object_pipelines_list,
                object_cameras_list=object_camera_path_list,
                apply_color_transfer=self.apply_color_transfer,
            )
        elif self.apply_transforms_to_camera_poses:
            _render_trajectory_video(
                pipeline,
                camera_path,
                output_filename=self.output_path,
                rendered_output_names=self.rendered_output_names,
                rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
                crop_data=crop_data,
                # seconds=seconds,
                output_format=self.output_format,
                image_format=self.image_format,
                jpeg_quality=self.jpeg_quality,
                colormap_options=self.colormap_options,
                cameras_filenames = image_filenames
            )
        else:
            _render_trajectory_video(
                pipeline,
                camera_path,
                output_filename=self.output_path,
                rendered_output_names=self.rendered_output_names,
                rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
                crop_data=crop_data,
                seconds=seconds,
                output_format=self.output_format,
                image_format=self.image_format,
                jpeg_quality=self.jpeg_quality,
                colormap_options=self.colormap_options,
                # cameras_filenames = image_filenames,
                object_pipelines_list=object_pipelines_list,
                object_cameras_list=object_camera_path_list,
                apply_color_transfer=self.apply_color_transfer,
            )

        if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value:
            # declare paths for left and right renders

            left_eye_path = self.output_path
            right_eye_path = left_eye_path.parent / "ods_render_Right.mp4"

            self.output_path = right_eye_path
            camera_path.camera_type[0] = CameraType.OMNIDIRECTIONALSTEREO_R.value

            CONSOLE.print("Rendering right eye view")
            _render_trajectory_video(
                pipeline,
                camera_path,
                output_filename=self.output_path,
                rendered_output_names=self.rendered_output_names,
                rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
                crop_data=crop_data,
                seconds=seconds,
                output_format=self.output_format,
                image_format=self.image_format,
                jpeg_quality=self.jpeg_quality,
                colormap_options=self.colormap_options,
            )

            # stack the left and right eye renders for final output
            self.output_path = Path(str(left_eye_path.parent)[:-5] + ".mp4")
            ffmpeg_ods_command = ""
            if self.output_format == "video":
                ffmpeg_ods_command = f'ffmpeg -y -i "{left_eye_path}" -i "{right_eye_path}" -filter_complex "[0:v]pad=iw:2*ih[int];[int][1:v]overlay=0:h" -c:v libx264 -crf 23 -preset veryfast "{self.output_path}"'
                run_command(ffmpeg_ods_command, verbose=False)
            if self.output_format == "images":
                # create a folder for the stacked renders
                self.output_path = Path(str(left_eye_path.parent)[:-5])
                self.output_path.mkdir(parents=True, exist_ok=True)
                if self.image_format == "png":
                    ffmpeg_ods_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.png")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.png")}" -filter_complex vstack -start_number 0 "{str(self.output_path)+"//%05d.png"}"'
                elif self.image_format == "jpeg":
                    ffmpeg_ods_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.jpg")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.jpg")}" -filter_complex vstack -start_number 0 "{str(self.output_path)+"//%05d.jpg"}"'
                run_command(ffmpeg_ods_command, verbose=False)

            # remove the temp files directory
            if str(left_eye_path.parent)[-5:] == "_temp":
                shutil.rmtree(left_eye_path.parent, ignore_errors=True)
            CONSOLE.print("[bold green]Final ODS Render Complete")


@dataclass
class RenderInterpolated(BaseRender):
    """Render a trajectory that interpolates between training or eval dataset images."""

    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    """Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis"""
    pose_source: Literal["eval", "train", "all"] = "eval"
    """Pose source to render."""
    interpolation_steps: int = 10
    """Number of interpolation steps between eval dataset cameras."""
    order_poses: bool = False
    """Whether to order camera poses by proximity."""
    frame_rate: int = 24
    """Frame rate of the output video."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""

    def main(self) -> None:
        """Main function."""
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="test",
        )

        install_checks.check_ffmpeg_installed()
        origin_name = False

        if self.pose_source == "eval":
            assert pipeline.datamanager.eval_dataset is not None
            cameras = pipeline.datamanager.eval_dataset.cameras
        elif self.pose_source == "train":
            assert pipeline.datamanager.train_dataset is not None
            cameras = pipeline.datamanager.train_dataset.cameras
        else:
            assert pipeline.datamanager.all_dataset is not None
            cameras = pipeline.datamanager.all_dataset.cameras
            origin_name = True


        seconds = self.interpolation_steps * len(cameras) / self.frame_rate
        
        if self.pose_source == "all":
            camera_path = cameras
        else:
            camera_path = get_interpolated_camera_path(
                cameras=cameras,
                steps=self.interpolation_steps,
                order_poses=self.order_poses,
            )

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=seconds,
            output_format=self.output_format,
            image_format=self.image_format,
            colormap_options=self.colormap_options,
            origin_name = origin_name,
        )


@dataclass
class SpiralRender(BaseRender):
    """Render a spiral trajectory (often not great)."""

    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    """Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis"""
    seconds: float = 3.0
    """How long the video should be."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""
    frame_rate: int = 24
    """Frame rate of the output video (only for interpolate trajectory)."""
    radius: float = 0.1
    """Radius of the spiral."""

    def main(self) -> None:
        """Main function."""
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="test",
        )

        install_checks.check_ffmpeg_installed()

        assert isinstance(pipeline.datamanager, VanillaDataManager)
        steps = int(self.frame_rate * self.seconds)
        camera_start = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0).flatten()
        camera_path = get_spiral_path(camera_start, steps=steps, radius=self.radius)

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=self.seconds,
            output_format=self.output_format,
            image_format=self.image_format,
            colormap_options=self.colormap_options,
        )

import hydra
from hydra.core.global_hydra import GlobalHydra

Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[RenderCameraPath, tyro.conf.subcommand(name="camera-path")],
        Annotated[RenderInterpolated, tyro.conf.subcommand(name="interpolate")],
        Annotated[SpiralRender, tyro.conf.subcommand(name="spiral")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")

    render_args = tyro.cli(Commands)
    if render_args.load_hydra_configs:
        # Clear the existing GlobalHydra instance, if any
        GlobalHydra.instance().clear()

        hydra.initialize(version_base=None, config_path="conf", job_name="render") # init like @hydra.main()
        cfg = hydra.compose("render_configs")
        if isinstance(render_args, RenderCameraPath):
            # for key, value in cfg.items():
            for key in cfg.keys():
                setattr(render_args, key, cfg[key])
            render_args.main()
        # TODO: write for other render methods
    else:
        tyro.cli(Commands).main()

if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa
