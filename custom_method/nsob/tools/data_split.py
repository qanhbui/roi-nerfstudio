import numpy as np
from nerfstudio.utils.io import load_from_json
from pathlib import Path
import json

scene_transform_path = Path('../transforms_scene.json')
object_transform_path = Path('../transforms_roi.json')
all_transform_path = Path('../transforms.json')

scene_transform_meta = load_from_json(scene_transform_path)
object_transform_meta = load_from_json(object_transform_path)

num_images_scene = len(scene_transform_meta["frames"])
num_images_object = len(object_transform_meta["frames"])
num_test_scene = 10
num_test_object = 5

i_all = np.arange(num_images_scene)
i_train = np.linspace(0, num_images_scene - 1, num_images_scene-num_test_scene, dtype=int)
i_eval = np.setdiff1d(i_all, i_train)

image_filenames_scene = []
for frame in scene_transform_meta["frames"]:
    image_filenames_scene.append(frame["file_path"])

eval_filenames_scene = np.array(image_filenames_scene)[i_eval]
