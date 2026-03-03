import numpy as np
from PIL import Image
import open3d as o3d
import math

import os
from pathlib import Path

import copy
import pandas as pd
import json

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
import time

import matplotlib.cm as cm

# read .out bundler file
def read_out_file(out_file_path):
    with open(out_file_path, 'r') as file:
        # Read all lines from the file and store them in a list
        lines = file.readlines()
    
    # Remove the first line if it starts with '#'
    if lines and lines[0].startswith('#'):
        lines = lines[1:]

    # <num_cameras> <num_points>   [two integers]
    # first_line_numbers = [int(num) for num in lines[0].strip().split(' ')]
    num_cameras = int(lines[0].strip().split(' ')[0])
    num_points = int(lines[0].strip().split(' ')[1])
    lines = lines[1:]
    
    cam_intrinsics = []
    cam_rotations = []
    cam_translations = []
    cam_transforms = []
    cam_positions = []
    for camera_idx in range(num_cameras):
        # <f> <k1> <k2>   [the focal length, followed by two radial distortion coeffs]
        cam_intrinsics.append([float(intrinsic) for intrinsic in lines[camera_idx*5].strip().split(' ')])
        # <R>             [a 3x3 matrix representing the camera rotation]
        matrix = []
        for rot_line in lines[camera_idx*5+1:camera_idx*5+4]:
            row = list(map(float, rot_line.strip().split(' ')))
            matrix.append(row)
        matrix = np.array(matrix)
        cam_rotations.append(matrix)
        # <t>             [a 3-vector describing the camera translation]
        translation = np.array([float(translation) for translation in lines[camera_idx*5+4].strip().split(' ')])
        cam_translations.append(translation)
        # Extrinsic matrix from .out, transformation matrix
        transform_matrix = np.eye(4)  # 4x4 identity matrix
        transform_matrix[:3, :3] = matrix
        transform_matrix[:3, 3] = translation
        cam_transforms.append(transform_matrix)
        # Camera position, inverse from transform matrix, to get pos like translation from transform matrix of .xml
        inverse_transform = np.linalg.inv(transform_matrix)  
        cam_pos = inverse_transform.transpose()[3][:3]
        cam_positions.append(cam_pos)
        # if i % 100 == 0:
        #     print(i)
    start_point_line = num_cameras * 5
    point_positions = []
    point_colors = []
    cam_lists_idx = []
    view_lists = []

    for point_idx in range(num_points):
        # <position>      [a 3-vector describing the 3D position of the point]
        point_positions.append([float(position) for position in lines[start_point_line+point_idx*3].strip().split(' ')])
        # <color>         [a 3-vector describing the RGB color of the point]
        point_colors.append([int(color) for color in lines[start_point_line+point_idx*3+1].strip().split(' ')])
        # <view list>     [a list of views the point is visible in]
        # <camera> <key> <x> <y>: 
        # <camera> is a camera index, 
        # <key> the index of the SIFT keypoint where the point was detected in that camera, 
        # and <x> and <y> are the detected positions of that keypoint. 
        cams_idx = []
        views = []
        num_view = int(lines[start_point_line+point_idx*3+2].strip().split(' ')[0])
        view_line = lines[start_point_line+point_idx*3+2].strip().split(' ')[1:]
        for view_idx in range(num_view):
            cams_idx.append(int(view_line[view_idx*4]))
            view_info = [int(view) for view in view_line[view_idx*4:view_idx*4+2]]
            view_info.extend([float(view) for view in view_line[view_idx*4+2:view_idx*4+4]])
            views.append(view_info)
        cam_lists_idx.append(cams_idx)
        view_lists.append(views)

    # return num_cameras, num_points, cam_intrinsics, cam_rotations, cam_translations, cam_transform, cam_position, point_positions, point_colors, cam_lists_idx, view_lists
    return num_cameras, num_points, cam_intrinsics, cam_transforms, cam_positions, point_positions, point_colors, cam_lists_idx, view_lists

# Take sparse pc points inside AABB of object of interest
def points_in_AABB(sparse_pcd, object_pcd):
    points = np.asarray(sparse_pcd.points)
    
    object_aabb = object_pcd.get_axis_aligned_bounding_box()
    object_min_bound = object_aabb.min_bound
    object_max_bound = object_aabb.max_bound
    
    inside_aabb_indices = np.where((points[..., 0] >= object_min_bound[0]) & (points[..., 0] <= object_max_bound[0])
                                   & (points[..., 1] >= object_min_bound[1]) & (points[..., 1] <= object_max_bound[1])
                                   & (points[..., 2] >= object_min_bound[2]) & (points[..., 2] <= object_max_bound[2]))[0]
    return inside_aabb_indices

# Process and write jsons of train and test split cameras
def write_split_jsons(json_path, output_json, output_test_json, train_indices, val_indices, test_indices, object_pc_aabb = None):
    with open(json_path, 'r') as f:
        data = json.load(f)
    src_dir = json_path.parent
    des_dir = output_json.parent
    relative_path = os.path.relpath(src_dir, des_dir)

    # write test json
    frames = data["frames"]
    cam_names = []
    for frame in frames:
        frame["file_path"] = f'{relative_path}/' + frame["file_path"] 
        cam_names.append(frame["file_path"])
    
    frame_test = [frames[i] for i in test_indices]
    new_data_test = {"frames": frame_test}
    with open(output_test_json, 'w') as outfile:
        json.dump(new_data_test, outfile, indent=4)

    # write train/eval split json
    cam_names = np.array(cam_names)
    data["train_filenames"] = list(cam_names[train_indices])
    data["val_filenames"] = list(cam_names[val_indices])
    data["test_filenames"] = list(cam_names[test_indices])
    if object_pc_aabb:
        data["object_pc_aabb"] = object_pc_aabb
    with open(output_json, 'w') as outfile:
        json.dump(data, outfile, indent=4)

#TODO: copy test images to gt folder
# Iterate multiple objects, output json transforms of train and test for render
def multi_objects_cameras_partition(sparse_full_pcd, object_pcds_dict, json_path, cam_lists_idx, num_cameras, des_dir, write_json=True):
    objects_camera_indices_all = set()
    scene_keeping_camera_indices_all = set()
    train_camera_indices_all = set()
    val_camera_indices_all = set()
    test_camera_indices_all = set()

    objects_cam_indices_dict = {}
    for idx, key in enumerate(object_pcds_dict):
        # Get points inidices inside Bbox
        object_aabb = object_pcds_dict[key]['pcd'].get_axis_aligned_bounding_box()
        point_min = np.expand_dims(np.hstack((object_aabb.min_bound[[2, 0, 1]], 1)), axis=-1)
        point_max = np.expand_dims(np.hstack((object_aabb.max_bound[[2, 0, 1]], 1)), axis=-1)
        ns_object_aabb = np.concatenate((point_min.T, point_max.T), axis=0).tolist()

        inside_aabb_points_indices = points_in_AABB(sparse_full_pcd, object_pcds_dict[key]['pcd'])
        # List of all cameras viewing points in the aabb
        object_camera_indices_full = set()
        for point_idx in inside_aabb_points_indices:
            cam_idx = set(cam_lists_idx[point_idx])
            object_camera_indices_full = object_camera_indices_full.union(cam_idx)

        # camera_indices_all = set(range(num_cameras))
        # camera_indices_else = list(camera_indices_all.difference(object_camera_indices_full))
        object_camera_indices_full = list(object_camera_indices_full)
        # camera_indices_all = list(camera_indices_all)

        # Filter cam by number of tie points or points of interest
        filtered_camera_indices = []
        # filtered_camera_indices_else = []
        max_num = 0
        for cam_idx in object_camera_indices_full:
            num_pts = 0
            for point_idx in inside_aabb_points_indices:
                if cam_idx in cam_lists_idx[point_idx]:
                    num_pts += 1
            max_num = max(max_num, num_pts)
            # Define threshhold by num points max
            if num_pts >= (max_num*0.1): # 0.05
                filtered_camera_indices.append(cam_idx)
            # else:
            #     filtered_camera_indices_else.append(cam_idx)

        # Keep 20% images for scene training
        object_split_fraction = 0.8
        num_filterd_cams = len(filtered_camera_indices)
        num_object_cams = math.ceil(num_filterd_cams * object_split_fraction)
        # num_scene_cams = num_filterd_cams - num_object_cams
        i_all = np.arange(num_filterd_cams)
        i_object = np.linspace(
            0, num_filterd_cams - 1, num_object_cams, dtype=int
        )  # equally spaced object images starting and ending at 0 and num_images-1
        i_scene_keeping = np.setdiff1d(i_all, i_object)  # scene keeping images are the remaining images

        object_camera_indices = [filtered_camera_indices[i] for i in i_object]
        scene_keeping_camera_indices = [filtered_camera_indices[i] for i in i_scene_keeping]

        # Store indices to dict
        indices_dict = {}
        indices_dict['object_full'] = object_camera_indices_full
        indices_dict['filtered_object_full'] = object_camera_indices
        indices_dict['scene_keeping'] = scene_keeping_camera_indices

        objects_camera_indices_all = objects_camera_indices_all.union(set(object_camera_indices))
        scene_keeping_camera_indices_all = scene_keeping_camera_indices_all.union(set(scene_keeping_camera_indices))

        # filter image_filenames and poses based on train/eval split percentage
        train_split_fraction = 0.7
        test_val_split_fraction = 0.6
        num_train_cams = math.ceil(num_object_cams * train_split_fraction)
        num_eval_cams = num_object_cams - num_train_cams

        i_object_all = np.arange(num_object_cams)
        i_train = np.linspace(
            0, num_object_cams - 1, num_train_cams, dtype=int
        )  # equally spaced training images starting and ending at 0 and num_images-1
        i_eval = np.setdiff1d(i_object_all, i_train)  # eval images are the remaining images

        num_test_cams = math.ceil(num_eval_cams * test_val_split_fraction)
        i_eval_all = np.arange(num_eval_cams)
        i_test = np.linspace(
            0, num_eval_cams - 1, num_test_cams, dtype=int
        )
        i_val = np.setdiff1d(i_eval_all, i_test)

        train_camera_indices = [object_camera_indices[i] for i in i_train]
        # eval_camera_indices = [object_camera_indices[i] for i in i_eval]
        val_camera_indices = [object_camera_indices[i_eval[i]] for i in i_val]
        test_camera_indices = [object_camera_indices[i_eval[i]] for i in i_test]

        indices_dict['train'] = train_camera_indices
        indices_dict['val'] = val_camera_indices
        indices_dict['test'] = test_camera_indices

        train_camera_indices_all = train_camera_indices_all.union(set(train_camera_indices))
        val_camera_indices_all = val_camera_indices_all.union(set(val_camera_indices))
        test_camera_indices_all = test_camera_indices_all.union(set(test_camera_indices))

        # Write split jsons file for train object
        output_test_json_object = object_pcds_dict[key]['output_test_json']
        output_json = object_pcds_dict[key]['output_json']
        if write_json:
            write_split_jsons(json_path, output_json, output_test_json_object, train_camera_indices, val_camera_indices, test_camera_indices, object_pc_aabb=ns_object_aabb)
        
        objects_cam_indices_dict[f'object_indices_{idx+1}'] = indices_dict

    #NOTE: Build scene images
    idx_all = np.arange(num_cameras)
    scene_camera_indices = (set(idx_all).difference(objects_camera_indices_all)).union(scene_keeping_camera_indices_all)
    scene_camera_indices = list(scene_camera_indices)
    scene_camera_indices.sort()

    # Keep 10% images for validation
    train_split_fraction = 0.9
    num_scene_images = len(scene_camera_indices)
    num_train_cams = math.ceil(num_scene_images * train_split_fraction)
    # num_val_cams = num_scene_images - num_train_cams
    i_all = np.arange(num_scene_images)
    i_train = np.linspace(
        0, num_scene_images - 1, num_train_cams, dtype=int
    )  # equally spaced object images starting and ending at 0 and num_images-1
    i_val = np.setdiff1d(i_all, i_train)

    train_scene_camera_indices = [scene_camera_indices[i] for i in i_train]
    val_scene_camera_indices = [scene_camera_indices[i] for i in i_val]
    test_scene_camera_indices = list(test_camera_indices_all)
    test_scene_camera_indices.sort()

    output_test_json_scene = des_dir / f'transforms_test_scene.json'
    output_json_scene = des_dir / f'transforms_scene.json'
    if write_json:
        write_split_jsons(json_path, output_json_scene, output_test_json_scene, train_scene_camera_indices, val_scene_camera_indices, test_scene_camera_indices)
    scene_cam_indices_dict = {
        "train": train_scene_camera_indices,
        "val": val_scene_camera_indices,
        "test": test_scene_camera_indices,
    }

    #NOTE: Build full images
    full_camera_indices = set(idx_all).difference(test_camera_indices_all)
    full_camera_indices = list(full_camera_indices)
    full_camera_indices.sort()

     # Keep 10% images for validation
    train_split_fraction = 0.9
    num_full_images = len(full_camera_indices)
    num_train_cams = math.ceil(num_full_images * train_split_fraction)
    # num_val_cams = num_full_images - num_train_cams
    i_all = np.arange(num_full_images)
    i_train = np.linspace(
        0, num_full_images - 1, num_train_cams, dtype=int
    )  # equally spaced object images starting and ending at 0 and num_images-1
    i_val = np.setdiff1d(i_all, i_train)

    train_full_camera_indices = [full_camera_indices[i] for i in i_train]
    val_full_camera_indices = [full_camera_indices[i] for i in i_val]
    test_full_camera_indices = test_scene_camera_indices

    # write full images transforms json
    output_test_json_full = des_dir / f'transforms_test_full.json'
    output_json_full = des_dir / f'transforms_full.json'
    if write_json:
        write_split_jsons(json_path, output_json_full, output_test_json_full, train_full_camera_indices, val_full_camera_indices, test_full_camera_indices)
    full_cam_indices_dict = {
        "train": train_full_camera_indices,
        "val": val_full_camera_indices,
        "test": test_full_camera_indices,
    }

    return objects_cam_indices_dict, scene_cam_indices_dict, full_cam_indices_dict

#NOTE: Draw multiples cams
# Draw multiples cameras -> return list of cameraLines different colors
def cam_visualisation(cam_transforms, camera_indices, color):
    # Define intrinsic matrix 
    WIDTH = 1920
    HEIGHT = 1080

    num = 2
    fl = 1676*num
    fx = fy = fl
    cx = WIDTH / 2
    cy = HEIGHT / 2
    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    scale = -0.1

    cameraLines = []
    for obj_cam_idx in camera_indices:
        extrinsics_matrix = cam_transforms[obj_cam_idx]
        cameraLine = o3d.geometry.LineSet.create_camera_visualization(view_width_px=WIDTH, view_height_px=HEIGHT, intrinsic=intrinsic_matrix, extrinsic=extrinsics_matrix, scale=scale)
        cameraLine.colors = color
        cameraLines.append(cameraLine)
    return cameraLines

def single_cam_visualisation(cam_transforms, camera_indice, color):
    # Define intrinsic matrix 
    WIDTH = 1920
    HEIGHT = 1080

    num = 2
    fl = 1676*num
    fx = fy = fl
    cx = WIDTH / 2
    cy = HEIGHT / 2
    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    scale = -0.1

    extrinsics_matrix = cam_transforms[camera_indice]
    cameraLine = o3d.geometry.LineSet.create_camera_visualization(view_width_px=WIDTH, view_height_px=HEIGHT, intrinsic=intrinsic_matrix, extrinsic=extrinsics_matrix, scale=scale)
    vector_color = o3d.utility.Vector3dVector(np.array([color] * 8))
    cameraLine.colors = vector_color
    return cameraLine

# Stack Point Cloud and cameras -> return list of cameraLines different colors and PC
def multi_objects_cameras_visualisation(objects_indices_dict, scene_indices_dict, object_pcds_dict, cam_transforms, colors_list):
    visu = []
    # Objects cams view
    for idx, (key_pcd, key_indice) in enumerate(zip(object_pcds_dict, objects_indices_dict)):
        object_pcd = object_pcds_dict[key_pcd]['pcd']
        object_aabb = object_pcd.get_axis_aligned_bounding_box()
        object_aabb.color = colors_list[idx]

        # object_indices = objects_indices_dict[key_indice]["object_full"]
        object_train_val_indices = objects_indices_dict[key_indice]["train"] + objects_indices_dict[key_indice]["val"]
        vector_color = o3d.utility.Vector3dVector(np.array([colors_list[idx]] * 8))
        # object_camLine = cam_visualisation(cam_transform, object_indices, vector_color)
        object_camLine = cam_visualisation(cam_transforms, object_train_val_indices, vector_color)
        
        visu += [object_pcd, object_aabb] + object_camLine

    # Remaining scene cams view
    scene_indices = scene_indices_dict['train'] + scene_indices_dict['val']
    vector_color = o3d.utility.Vector3dVector(np.array([colors_list[idx+1]] * 8))
    scene_camLine = cam_visualisation(cam_transforms, scene_indices, vector_color)
    visu += scene_camLine
    # # Test cams view ?
    # test_indices = scene_indices_dict['test']
    # vector_color = o3d.utility.Vector3dVector(np.array([colors_list[idx+2]] * 8))
    # test_camLine = cam_visualisation(cam_transform, test_indices, vector_color)
    # visu += test_camLine

    return visu

def normalize_list(lst):
    min_val = min(lst)
    max_val = max(lst)
    normalized = [(x - min_val) / (max_val - min_val) for x in lst]
    return normalized

# Color heatmap the cameras corresponding to the number of points seen by each camera
def num_points_cameras_heatmap_visualisation(sparse_full_pcd, object_pcds_dict, cam_lists_idx, num_cameras, cam_transforms, object_indice_visu=0, all_visu=False):
    black_color = [0, 0, 0]
    key_name = list(object_pcds_dict.keys())[object_indice_visu]

    object_pcd = object_pcds_dict[key_name]['pcd']
    object_aabb = object_pcd.get_axis_aligned_bounding_box()
    object_aabb.color = black_color

    pcd_visu = [object_pcd, object_aabb]

    inside_aabb_points_indices = points_in_AABB(sparse_full_pcd, object_pcd)

    # Num inside AABB points of All cameras
    num_points_per_cam_all = []
    object_camera_indices_full = set()
    for cam_idx in range(num_cameras):
        num_pts = 0
        for point_idx in inside_aabb_points_indices:
            if cam_idx in cam_lists_idx[point_idx]:
                num_pts += 1
            cam_indices_set = set(cam_lists_idx[point_idx])
            object_camera_indices_full = object_camera_indices_full.union(cam_indices_set)
        num_points_per_cam_all.append(num_pts)

    normalized_points_per_cam_all = normalize_list(num_points_per_cam_all)

    # Num inside AABB points of object cameras
    num_points_per_cam = [num_points_per_cam_all[i] for i in object_camera_indices_full]
    normalized_points_per_cam = normalize_list(num_points_per_cam)

    if all_visu:
        cams_all_visu = []
        colormap = cm.get_cmap('jet')
        for cam_idx in range(num_cameras):
            color_normalized = colormap(normalized_points_per_cam_all[cam_idx])
            cams_all_visu += [single_cam_visualisation(cam_transforms, cam_idx, color_normalized[:3])]

        visu = pcd_visu + cams_all_visu

    else:
        camera_indices_all = set(range(num_cameras))
        camera_indices_else = list(camera_indices_all.difference(object_camera_indices_full))
        object_camera_indices_full = list(object_camera_indices_full)

        vector_color = o3d.utility.Vector3dVector(np.array([black_color] * 8))
        else_camLines = cam_visualisation(cam_transforms, camera_indices_else, vector_color)

        object_cams_visu = []
        colormap = cm.get_cmap('jet')
        for idx, cam_idx in enumerate(object_camera_indices_full):
            color_normalized = colormap(normalized_points_per_cam[idx])
            object_cams_visu += [single_cam_visualisation(cam_transforms, cam_idx, color_normalized[:3])]

        visu = pcd_visu + else_camLines + object_cams_visu

    return visu

@hydra.main(version_base=None, config_path=".", config_name="config_pc")
def main(cfg : DictConfig) -> None:
    machine_name = cfg.db.machine[cfg.db.machine_indice]
    print("Working in machine:", machine_name)
    machine = cfg.db[machine_name]
    out_file_path = Path(machine.out_file_path)
    start_time = time.time()
    num_cameras, num_points, cam_intrinsics, cam_transforms, cam_positions, point_positions, point_colors, cam_lists_idx, view_lists = read_out_file(out_file_path)
    end_time = time.time()
    print(f"Read .out file: Finished. Runtime: {end_time - start_time} seconds")
    sparse_big_pc_path = machine.sparse_full_pcd
    sparse_pc_path = machine.sparse_pcd

    sparse_big_pcd = o3d.io.read_point_cloud(sparse_big_pc_path)
    sparse_pcd = o3d.io.read_point_cloud(sparse_pc_path)

    object_pc_path_1 = machine.object_pc_path_1
    object_pc_path_2 = machine.object_pc_path_2
    object_pc_path_3 = machine.object_pc_path_3
    object_pc_path_4 = machine.object_pc_path_4
    
    object_pcd_1 = o3d.io.read_point_cloud(object_pc_path_1)
    object_pcd_2 = o3d.io.read_point_cloud(object_pc_path_2)
    object_pcd_3 = o3d.io.read_point_cloud(object_pc_path_3)
    object_pcd_4 = o3d.io.read_point_cloud(object_pc_path_4)

    object_pcds_list = [object_pcd_1, object_pcd_2, object_pcd_3, object_pcd_4]

    start_time = time.time()
    json_path = Path(machine.json_path)

    object_pcds_indices = cfg.db.objects_indices
    object_pcds_list = [object_pcds_list[i] for i in object_pcds_indices]
    print("Number of objects: ", len(object_pcds_list))
    object_pcd_name = 'object_pcd'
    output_object_name = 'object'
    src_dir = json_path.parent
    des_dir = src_dir / f'{cfg.db.out_dir_prename}_{len(object_pcds_list)}_objects'
    if not des_dir.exists():
        # Create directory
        des_dir.mkdir(parents=True, exist_ok=True)

    object_pcds_dict = {}
    for idx, pcd in enumerate(object_pcds_list):
        pcd_dict = {}
        pcd_dict['pcd'] = pcd
        pcd_dict['output_json'] = des_dir / f'transforms_{output_object_name}_{idx+1}.json'
        pcd_dict['output_test_json'] = des_dir / f'transforms_test_{output_object_name}_{idx+1}.json'
        object_pcds_dict[f'{object_pcd_name}_{idx+1}'] = pcd_dict

    end_time = time.time()
    print(f"Create objects PC dict: Finished. Runtime: {end_time - start_time} seconds")

    start_time = time.time()
    write_json = cfg.db.write_json
    objects_indices_dict, scene_indices_dict, full_indices_dict = multi_objects_cameras_partition(sparse_big_pcd, object_pcds_dict, json_path, cam_lists_idx, num_cameras, des_dir, write_json)
    end_time = time.time()
    print(f"Multi-Object Cameras partition: Finished. Runtime: {end_time - start_time} seconds")

    if cfg.db.cam_visu:
        red_color = [1, 0, 0]
        green_color = [0, 1, 0]
        blue_color = [0, 0, 1]
        yellow_color = [1, 1, 0]
        cyan_color = [0, 1, 1]
        violet_color = [1, 0, 1]
        black_color = [0, 0, 0]
        colors_list = [red_color, green_color, blue_color, yellow_color, cyan_color, violet_color, black_color]

        visu = multi_objects_cameras_visualisation(objects_indices_dict, scene_indices_dict, object_pcds_dict, cam_transforms, colors_list)
        o3d.visualization.draw_geometries(visu)

    if cfg.db.cam_heatmap_visu:
        object_indice_visu = cfg.db.object_indice_visu
        all_visu = cfg.db.all_visu
        visu = num_points_cameras_heatmap_visualisation(sparse_big_pcd, object_pcds_dict, cam_lists_idx, num_cameras, cam_transforms, object_indice_visu, all_visu)
        o3d.visualization.draw_geometries(visu)

    return objects_indices_dict, scene_indices_dict, full_indices_dict, object_pcds_dict, cam_transforms

if __name__ == "__main__":
    main()


# # Clear the existing GlobalHydra instance, if any
# GlobalHydra.instance().clear()

# hydra.initialize(version_base=None, config_path="conf", job_name="read_config") # init like @hydra.main()
# cfg = hydra.compose("config_pc")
# cfg.db.machine_indice = 1
# cfg.db.objects_indices = [0,1]
# cfg.db.write_json = False
# cfg.db.cam_visu = False
# objects_indices, scene_indices, full_indices, object_pcds, cam_transform = main(cfg)

# red_color = [1, 0, 0]
# green_color = [0, 1, 0]
# blue_color = [0, 0, 1]
# yellow_color = [1, 1, 0]
# cyan_color = [0, 1, 1]
# violet_color = [1, 0, 1]
# black_color = [0, 0, 0]
# colors_list = [red_color, green_color, blue_color, yellow_color, cyan_color, violet_color, black_color]
# visu = multi_objects_cameras_visualisation(objects_indices, scene_indices, object_pcds, cam_transform, colors_list)

# o3d.visualization.draw_geometries(visu)


# o3d.visualization.draw_plotly(visu)

# # Relative path 
# pwd_path =  Path(os.getcwd())
# config_path = Path(r'C:\Users\qb273560\Documents\PhDcode\data-docker\code\tools\config_pc.yaml')
# relative_path = os.path.relpath(config_path.parent, pwd_path)

# # Verify points
# inside_sparse_points = np.asarray(sparse_big_pcd.points)[inside_object_aabb_points_indices]
# inside_sparse_colors = np.asarray(sparse_big_pcd.colors)[inside_object_aabb_points_indices]
# new_inside_pcd = o3d.geometry.PointCloud()
# new_inside_pcd.points = o3d.utility.Vector3dVector(inside_sparse_points)
# new_inside_pcd.colors = o3d.utility.Vector3dVector(inside_sparse_colors)

# o3d.visualization.draw_geometries([new_inside_pcd, object_aabb_1])
