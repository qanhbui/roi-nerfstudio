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
Code for camera paths.
"""

from typing import Any, Dict, Optional, Tuple, List

import torch

import nerfstudio.utils.poses as pose_utils
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.camera_utils import get_interpolated_poses_many
from nerfstudio.cameras.cameras import Cameras, CameraType, CAMERA_MODEL_TO_TYPE
from nerfstudio.viewer.server.utils import three_js_perspective_camera_focal_length


def get_interpolated_camera_path(cameras: Cameras, steps: int, order_poses: bool) -> Cameras:
    """Generate a camera path between two cameras. Uses the camera type of the first camera

    Args:
        cameras: Cameras object containing intrinsics of all cameras.
        steps: The number of steps to interpolate between the two cameras.

    Returns:
        A new set of cameras along a path.
    """
    Ks = cameras.get_intrinsics_matrices()
    poses = cameras.camera_to_worlds
    poses, Ks = get_interpolated_poses_many(poses, Ks, steps_per_transition=steps, order_poses=order_poses)

    cameras = Cameras(
        fx=Ks[:, 0, 0],
        fy=Ks[:, 1, 1],
        cx=Ks[0, 0, 2],
        cy=Ks[0, 1, 2],
        camera_type=cameras.camera_type[0],
        camera_to_worlds=poses,
    )
    return cameras


def get_spiral_path(
    camera: Cameras,
    steps: int = 30,
    radius: Optional[float] = None,
    radiuses: Optional[Tuple[float]] = None,
    rots: int = 2,
    zrate: float = 0.5,
) -> Cameras:
    """
    Returns a list of camera in a spiral trajectory.

    Args:
        camera: The camera to start the spiral from.
        steps: The number of cameras in the generated path.
        radius: The radius of the spiral for all xyz directions.
        radiuses: The list of radii for the spiral in xyz directions.
        rots: The number of rotations to apply to the camera.
        zrate: How much to change the z position of the camera.

    Returns:
        A spiral camera path.
    """

    assert radius is not None or radiuses is not None, "Either radius or radiuses must be specified."
    assert camera.ndim == 1, "We assume only one batch dim here"
    if radius is not None and radiuses is None:
        rad = torch.tensor([radius] * 3, device=camera.device)
    elif radiuses is not None and radius is None:
        rad = torch.tensor(radiuses, device=camera.device)
    else:
        raise ValueError("Only one of radius or radiuses must be specified.")

    up = camera.camera_to_worlds[0, :3, 2]  # scene is z up
    focal = torch.min(camera.fx[0], camera.fy[0])
    target = torch.tensor([0, 0, -focal], device=camera.device)  # camera looking in -z direction

    c2w = camera.camera_to_worlds[0]
    c2wh_global = pose_utils.to4x4(c2w)

    local_c2whs = []
    for theta in torch.linspace(0.0, 2.0 * torch.pi * rots, steps + 1)[:-1]:
        center = (
            torch.tensor([torch.cos(theta), -torch.sin(theta), -torch.sin(theta * zrate)], device=camera.device) * rad
        )
        lookat = center - target
        c2w = camera_utils.viewmatrix(lookat, up, center)
        c2wh = pose_utils.to4x4(c2w)
        local_c2whs.append(c2wh)

    new_c2ws = []
    for local_c2wh in local_c2whs:
        c2wh = torch.matmul(c2wh_global, local_c2wh)
        new_c2ws.append(c2wh[:3, :4])
    new_c2ws = torch.stack(new_c2ws, dim=0)

    times = None
    if camera.times is not None:
        times = torch.linspace(0, 1, steps)[:, None]
    return Cameras(
        fx=camera.fx[0],
        fy=camera.fy[0],
        cx=camera.cx[0],
        cy=camera.cy[0],
        camera_to_worlds=new_c2ws,
        times=times,
    )


def get_path_from_json(camera_path: Dict[str, Any]) -> Cameras:
    """Takes a camera path dictionary and returns a trajectory as a Camera instance.

    Args:
        camera_path: A dictionary of the camera path information coming from the viewer.

    Returns:
        A Cameras instance with the camera path.
    """

    image_height = camera_path["render_height"]
    image_width = camera_path["render_width"]

    if "camera_type" not in camera_path:
        camera_type = CameraType.PERSPECTIVE
    elif camera_path["camera_type"] == "fisheye":
        camera_type = CameraType.FISHEYE
    elif camera_path["camera_type"] == "equirectangular":
        camera_type = CameraType.EQUIRECTANGULAR
    elif camera_path["camera_type"].lower() == "omnidirectional":
        camera_type = CameraType.OMNIDIRECTIONALSTEREO_L
    else:
        camera_type = CameraType.PERSPECTIVE

    c2ws = []
    fxs = []
    fys = []
    for camera in camera_path["camera_path"]:
        # pose
        c2w = torch.tensor(camera["camera_to_world"]).view(4, 4)[:3]
        c2ws.append(c2w)
        if (
            camera_type == CameraType.EQUIRECTANGULAR
            or camera_type == CameraType.OMNIDIRECTIONALSTEREO_L
            or camera_type == CameraType.OMNIDIRECTIONALSTEREO_R
        ):
            fxs.append(image_width / 2)
            fys.append(image_height)
        else:
            # field of view
            fov = camera["fov"]
            focal_length = three_js_perspective_camera_focal_length(fov, image_height)
            fxs.append(focal_length)
            fys.append(focal_length)

    # Iff ALL cameras in the path have a "time" value, construct Cameras with times
    if all("render_time" in camera for camera in camera_path["camera_path"]):
        times = torch.tensor([camera["render_time"] for camera in camera_path["camera_path"]])
    else:
        times = None

    camera_to_worlds = torch.stack(c2ws, dim=0)
    fx = torch.tensor(fxs)
    fy = torch.tensor(fys)
    return Cameras(
        fx=fx,
        fy=fy,
        cx=image_width / 2,
        cy=image_height / 2,
        camera_to_worlds=camera_to_worlds,
        camera_type=camera_type,
        times=times,
    )

from jaxtyping import Float
from torch import Tensor
import numpy as np
from pathlib import Path
import os

def change_ref_transform_photogrametry(camera_path: Dict[str, Any], scene_scale: Float, inv_scene_transform: Float[Tensor, "4 4"]) -> Cameras:
    """Takes a camera path dictionary and returns a trajectory as a Camera instance.

    Args:
        camera_path: A dictionary of the camera path information coming from the viewer.

    Returns:
        A Cameras instance with the camera path.
    """

    image_height = camera_path["render_height"]
    image_width = camera_path["render_width"]

    if "camera_type" not in camera_path:
        camera_type = CameraType.PERSPECTIVE
    elif camera_path["camera_type"] == "fisheye":
        camera_type = CameraType.FISHEYE
    elif camera_path["camera_type"] == "equirectangular":
        camera_type = CameraType.EQUIRECTANGULAR
    elif camera_path["camera_type"].lower() == "omnidirectional":
        camera_type = CameraType.OMNIDIRECTIONALSTEREO_L
    else:
        camera_type = CameraType.PERSPECTIVE

    c2ws = []
    fxs = []
    fys = []
    for camera in camera_path["camera_path"]:
        # pose
        # c2w = torch.tensor(camera["camera_to_world"]).view(4, 4)[:3]
        c2w = torch.tensor(camera["camera_to_world"]).view(4, 4)
        c2ws.append(c2w)
        if (
            camera_type == CameraType.EQUIRECTANGULAR
            or camera_type == CameraType.OMNIDIRECTIONALSTEREO_L
            or camera_type == CameraType.OMNIDIRECTIONALSTEREO_R
        ):
            fxs.append(image_width / 2)
            fys.append(image_height)
        else:
            # field of view
            fov = camera["fov"]
            focal_length = three_js_perspective_camera_focal_length(fov, image_height)
            fxs.append(focal_length)
            fys.append(focal_length)

    # If ALL cameras in the path have a "time" value, construct Cameras with times
    if all("render_time" in camera for camera in camera_path["camera_path"]):
        times = torch.tensor([camera["render_time"] for camera in camera_path["camera_path"]])
    else:
        times = None

    camera_to_worlds = torch.stack(c2ws, dim=0)

    #########
    fx = torch.tensor(fxs)
    fy = torch.tensor(fys)

    new_data = {"fl_x": float(fx[0]), "fl_y": float(fy[0]), "cx": int(image_width/2), "cy": int(image_height/2), "w": image_width, "h":image_height}
    new_data["frames"] = []

    poses = camera_to_worlds

    poses[..., :3] /= scene_scale # n, 4
    # poses = inv_scene_transform @ (poses.T) # 4, 4 @ 4, n = 4, n
    poses = inv_scene_transform @ poses # 4, 4 @ 4, n = 4, n
    # poses = roi_transform @ poses # 4, 4 @ 4, n = 4, n
    # poses = poses.T # n, 4
    # poses[..., :3] *= roi_scale
    # camera_to_worlds = poses[:, :3]

    for c2w in poses:
        new_data["frames"].append({"transform_matrix": c2w.tolist()})

    return new_data

def get_camera_from_json(camera_meta: Dict[str, Any], applied_transform: Float[Tensor, "3 4"]=None, applied_scale: Float=None) -> Tuple[Cameras, List] :
    """Takes a camera path dictionary and returns a trajectory as a Camera instance.

    Args:
        camera_meta: A dictionary of the camera meta information coming from the json file.

    Returns:
        A Cameras instance with the json camera.
    """

    image_filenames = []
    poses = []

    fx_fixed = "fl_x" in camera_meta
    fy_fixed = "fl_y" in camera_meta
    cx_fixed = "cx" in camera_meta
    cy_fixed = "cy" in camera_meta
    height_fixed = "h" in camera_meta
    width_fixed = "w" in camera_meta
    distort_fixed = False
    for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
        if distort_key in camera_meta:
            distort_fixed = True
            break
    fx = []
    fy = []
    cx = []
    cy = []
    height = []
    width = []
    distort = []

    for frame in camera_meta["frames"]:
        filepath = Path(frame["file_path"])
        image_name = filepath.stem
        image_filenames.append(image_name)

        if not fx_fixed:
            assert "fl_x" in frame, "fx not specified in frame"
            fx.append(float(frame["fl_x"]))
        if not fy_fixed:
            assert "fl_y" in frame, "fy not specified in frame"
            fy.append(float(frame["fl_y"]))
        if not cx_fixed:
            assert "cx" in frame, "cx not specified in frame"
            cx.append(float(frame["cx"]))
        if not cy_fixed:
            assert "cy" in frame, "cy not specified in frame"
            cy.append(float(frame["cy"]))
        if not height_fixed:
            assert "h" in frame, "height not specified in frame"
            height.append(int(frame["h"]))
        if not width_fixed:
            assert "w" in frame, "width not specified in frame"
            width.append(int(frame["w"]))
        if not distort_fixed:
            distort.append(
                camera_utils.get_distortion_params(
                    k1=float(frame["k1"]) if "k1" in frame else 0.0,
                    k2=float(frame["k2"]) if "k2" in frame else 0.0,
                    k3=float(frame["k3"]) if "k3" in frame else 0.0,
                    k4=float(frame["k4"]) if "k4" in frame else 0.0,
                    p1=float(frame["p1"]) if "p1" in frame else 0.0,
                    p2=float(frame["p2"]) if "p2" in frame else 0.0,
                )
            )

        
        poses.append(np.array(frame["transform_matrix"]))
    
    poses = torch.from_numpy(np.array(poses).astype(np.float32))
    
    # # Apply transform and scale from dataparser_transforms to camera pose
    # full_poses = torch.cat(
    #     (
    #         poses,
    #         torch.tensor([[[0, 0, 0, 1]]], dtype=poses.dtype, device=poses.device).repeat_interleave(len(poses), 0),
    #     ),
    #     1,
    # )
    if applied_transform is not None:
        full_applied_transform = torch.cat([applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=applied_transform.dtype)], 0)
        poses = torch.einsum("ij,bjk->bik", full_applied_transform, poses)
        poses[..., :3, 3] *= applied_scale
    
    poses = poses[:, :3]

    if "camera_model" in camera_meta:
        camera_type = CAMERA_MODEL_TO_TYPE[camera_meta["camera_model"]]
    else:
        camera_type = CameraType.PERSPECTIVE

    fx = float(camera_meta["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)
    fy = float(camera_meta["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)
    cx = float(camera_meta["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)
    cy = float(camera_meta["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)
    height = int(camera_meta["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)
    width = int(camera_meta["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)
    if distort_fixed:
        distortion_params = camera_utils.get_distortion_params(
            k1=float(camera_meta["k1"]) if "k1" in camera_meta else 0.0,
            k2=float(camera_meta["k2"]) if "k2" in camera_meta else 0.0,
            k3=float(camera_meta["k3"]) if "k3" in camera_meta else 0.0,
            k4=float(camera_meta["k4"]) if "k4" in camera_meta else 0.0,
            p1=float(camera_meta["p1"]) if "p1" in camera_meta else 0.0,
            p2=float(camera_meta["p2"]) if "p2" in camera_meta else 0.0,
        )
    else:
        distortion_params = torch.stack(distort, dim=0)

    cameras = Cameras(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        distortion_params=distortion_params,
        height=height,
        width=width,
        camera_to_worlds=poses[:, :3, :4],
        camera_type=camera_type,
    )

    return cameras, image_filenames
    # return cameras

def get_eval_from_json(camera_meta: Dict[str, Any], data_dir: Optional[Path]=None, applied_transform: Optional[Float[Tensor, "3 4"]]=None, applied_scale: Optional[Float]=None ) -> Tuple[Cameras, List] :
    """Takes a camera path dictionary and returns a trajectory as a Camera instance.

    Args:
        camera_meta: A dictionary of the camera meta information coming from the json file.

    Returns:
        A Cameras instance with the json camera.
    """

    # image_filenames = []
    image_paths = []
    poses = []

    fx_fixed = "fl_x" in camera_meta
    fy_fixed = "fl_y" in camera_meta
    cx_fixed = "cx" in camera_meta
    cy_fixed = "cy" in camera_meta
    height_fixed = "h" in camera_meta
    width_fixed = "w" in camera_meta
    distort_fixed = False
    for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
        if distort_key in camera_meta:
            distort_fixed = True
            break
    fx = []
    fy = []
    cx = []
    cy = []
    height = []
    width = []
    distort = []

    for frame in camera_meta["frames"]:
        filepath = Path(frame["file_path"])
        fname = data_dir / filepath
        # image_name = filepath.stem

        if not fx_fixed:
            assert "fl_x" in frame, "fx not specified in frame"
            fx.append(float(frame["fl_x"]))
        if not fy_fixed:
            assert "fl_y" in frame, "fy not specified in frame"
            fy.append(float(frame["fl_y"]))
        if not cx_fixed:
            assert "cx" in frame, "cx not specified in frame"
            cx.append(float(frame["cx"]))
        if not cy_fixed:
            assert "cy" in frame, "cy not specified in frame"
            cy.append(float(frame["cy"]))
        if not height_fixed:
            assert "h" in frame, "height not specified in frame"
            height.append(int(frame["h"]))
        if not width_fixed:
            assert "w" in frame, "width not specified in frame"
            width.append(int(frame["w"]))
        if not distort_fixed:
            distort.append(
                camera_utils.get_distortion_params(
                    k1=float(frame["k1"]) if "k1" in frame else 0.0,
                    k2=float(frame["k2"]) if "k2" in frame else 0.0,
                    k3=float(frame["k3"]) if "k3" in frame else 0.0,
                    k4=float(frame["k4"]) if "k4" in frame else 0.0,
                    p1=float(frame["p1"]) if "p1" in frame else 0.0,
                    p2=float(frame["p2"]) if "p2" in frame else 0.0,
                )
            )

        # image_filenames.append(image_name)
        image_paths.append(fname)
        poses.append(np.array(frame["transform_matrix"]))
    
    poses = torch.from_numpy(np.array(poses).astype(np.float32))

    if applied_transform is not None and applied_scale is not None:
        full_applied_transform = torch.cat([applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=applied_transform.dtype)], 0)
        poses = torch.einsum("ij,bjk->bik", full_applied_transform, poses)
        poses[..., :3, 3] *= applied_scale
        poses = poses[:, :3]

    if "camera_model" in camera_meta:
        camera_type = CAMERA_MODEL_TO_TYPE[camera_meta["camera_model"]]
    else:
        camera_type = CameraType.PERSPECTIVE

    fx = float(camera_meta["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)
    fy = float(camera_meta["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)
    cx = float(camera_meta["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)
    cy = float(camera_meta["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)
    height = int(camera_meta["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)
    width = int(camera_meta["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)
    if distort_fixed:
        distortion_params = camera_utils.get_distortion_params(
            k1=float(camera_meta["k1"]) if "k1" in camera_meta else 0.0,
            k2=float(camera_meta["k2"]) if "k2" in camera_meta else 0.0,
            k3=float(camera_meta["k3"]) if "k3" in camera_meta else 0.0,
            k4=float(camera_meta["k4"]) if "k4" in camera_meta else 0.0,
            p1=float(camera_meta["p1"]) if "p1" in camera_meta else 0.0,
            p2=float(camera_meta["p2"]) if "p2" in camera_meta else 0.0,
        )
    else:
        distortion_params = torch.stack(distort, dim=0)

    cameras = Cameras(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        distortion_params=distortion_params,
        height=height,
        width=width,
        camera_to_worlds=poses[:, :3, :4],
        camera_type=camera_type,
    )

    return cameras, image_paths