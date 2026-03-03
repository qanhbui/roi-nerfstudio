import numpy as np
import numpy.typing as npt 
import torch
from torch import Tensor
from typing import List, Dict, Tuple
from jaxtyping import Float

from pathlib import Path
import json

from PIL import Image
import os

from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def get_numpy_image(image_filenames, image_idx: int) -> npt.NDArray[np.uint8]:
    """Returns the image of shape (H, W, 3 or 4).

    Args:
        image_idx: The image index in the dataset.
    """
    image_filename = image_filenames[image_idx]
    pil_image = Image.open(image_filename)
    image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
    if len(image.shape) == 2:
        image = image[:, :, None].repeat(3, axis=2)
    assert len(image.shape) == 3
    assert image.dtype == np.uint8
    assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
    return image

def get_image(image_filenames, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
    """Returns a 3 channel image.

    Args:
        image_idx: The image index in the dataset.
    """
    image = torch.from_numpy(get_numpy_image(image_filenames, image_idx).astype("float32") / 255.0)
    return image

def get_one_image(image_filename) -> Float[Tensor, "image_height image_width num_channels"]:
    """Returns the image of shape (H, W, 3 or 4).

    Args:
        image_idx: The image index in the dataset.
    """
    pil_image = Image.open(image_filename)
    image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
    if len(image.shape) == 2:
        image = image[:, :, None].repeat(3, axis=2)
    assert len(image.shape) == 3
    assert image.dtype == np.uint8
    assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
    image = torch.from_numpy(image.astype("float32") / 255.0)
    return image

def get_image_filenames(data_dir) -> List[Path]:
    entries = os.listdir(data_dir)
    entries.sort()
    image_filenames = [data_dir / Path(entry) for entry in entries]
    return image_filenames

def compute_image_metrics(
    rendered_image_dir: Path, gt_filenames: List[Path], get_std: bool = False
) -> Tuple[List[Dict], Dict[str, float]]:
    psnr_computation = PeakSignalNoiseRatio(data_range=1.0)
    ssim_computation = structural_similarity_index_measure
    lpips_computation = LearnedPerceptualImagePatchSimilarity(normalize=True)

    # gt_filenames = get_image_filenames(gt_dir)
    predicted_filenames = get_image_filenames(rendered_image_dir)
    predicted_suffix_fname = predicted_filenames[2].suffix

    metrics_dict_list = []
    for idx, gt_filename in enumerate(gt_filenames):
        gt_base_fname = gt_filename.stem        
        gt_rgb = get_image(gt_filenames, idx)
        # rendered_rgb = get_image(predicted_filenames, idx)
        rendered_rgb = get_one_image(rendered_image_dir / Path(gt_base_fname + predicted_suffix_fname))
        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        rendered_rgb = torch.moveaxis(rendered_rgb, -1, 0)[None, ...]

        psnr = psnr_computation(gt_rgb, rendered_rgb)
        ssim = ssim_computation(gt_rgb, rendered_rgb)
        lpips = lpips_computation(gt_rgb, rendered_rgb)
        
        # all of these metrics will be logged as scalars
        metrics_dict = {"image_fname": gt_base_fname}
        metrics_dict["psnr"] = float(psnr.item()) 
        metrics_dict["ssim"] = float(ssim)
        metrics_dict["lpips"] = float(lpips)
        metrics_dict_list.append(metrics_dict)

    print("Num eval images: ", len(metrics_dict_list))
    # average the metrics list
    metrics_dict = {}
    for key in metrics_dict_list[0].keys():
        if get_std and key != "image_fname":
            key_std, key_mean = torch.std_mean(
                torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
            )
            metrics_dict[key] = float(key_mean)
            metrics_dict[f"{key}_std"] = float(key_std)
        elif key != "image_fname":
            metrics_dict[key] = float(
                torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
            )

    return metrics_dict_list, metrics_dict

def save_metrics_json(
    rendered_image_dirs: Dict[str, Path], gt_dir: Path, save_path: Path, save_path_2: Path
):
    gt_filenames = get_image_filenames(gt_dir)
    psnr_dict = {}
    psnr_images_dict = {}
    for key in rendered_image_dirs:
        rendered_image_dir = rendered_image_dirs[key]
        print(key)
        metrics_dict_list, metrics_dict = compute_image_metrics(rendered_image_dir, gt_filenames)
        psnr_dict[key] = metrics_dict
        psnr_images_dict[key] = metrics_dict_list
    
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)
    with open(save_path, "w", encoding="UTF-8") as file:
        json.dump(psnr_dict, file, indent=4)

    with open(save_path_2, "w", encoding="UTF-8") as file2:
        json.dump(psnr_images_dict, file2, indent=4)


full_huge = Path('../test_full_huge')
full_big = Path('../test_full_big')
scene_huge = Path('../test_scene_huge')
scene_big = Path('../test_scene_big')
compo_fh = Path('../test_comp_fh')
compo_fb = Path('../test_comp_fb')
compo_sh = Path('../test_comp_sh')
compo_sb = Path('../test_comp_sb')
compo_fh_full = Path('../test_comp_fh_full')
compo_fb_full = Path('../test_comp_fb_full')
compo_sh_full = Path('../test_comp_sh_full')
compo_sb_full = Path('../test_comp_sb_full')
roi = Path('../test_roi')
gt_dir = Path('../test_gt')

rendered_image_dirs = {
    "full_huge": full_huge,
    "compo_fh": compo_fh,
    # "compo_fh_full": compo_fh_full,
    # "compo_fh_ct": compo_fh_ct,
    "scene_huge": scene_huge,
    "compo_sh": compo_sh,
    # "compo_sh_full": compo_sh_full,

    "full_big": full_big,
    "compo_fb": compo_fb,
    # "compo_fb_full": compo_fb_full,

    "scene_big": scene_big,
    "compo_sb": compo_sb,
    # "compo_sb_full": compo_sb_full,
    # "compo_sb_ct": compo_sb_ct,
    
    "roi": roi,
}

save_path = Path('../psnr_mean.json')
save_path_2 = Path('../psnr_detail.json')

save_metrics_json(rendered_image_dirs, gt_dir, save_path, save_path_2)