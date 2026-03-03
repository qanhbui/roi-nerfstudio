import numpy as np
import numpy.typing as npt 
import torch
from torch import Tensor
from typing import List, Dict, Tuple
from jaxtyping import Float

from pathlib import Path
import json

import os

from nerfstudio.utils.io import load_from_json

json_path = Path('../psnr_detail.json')
des_path = Path('../psnr_process.json')

methods = load_from_json(json_path)

baselines = ['full_huge', 'scene_huge', 'full_big', 'scene_big']
compos = ['compo_fh', 'compo_sh', 'compo_fb', 'compo_sb']

psnr_processed_dict = {}

for base, compo in zip(baselines, compos):
    print(base)
    print(compo)
    psnr_enh = []
    ssim_enh = []
    lpips_enh = []
    min_psnr = 50 
    max_psnr = 0
    min_ssim = 50 
    max_ssim = 0
    min_lpips = 50 
    max_lpips = 0
    score_dict = {}
    n = 0.0022

    for frame1, frame2 in zip(methods[base], methods[compo]):
        fname  = frame1["image_fname"]
        psnr1 = float(frame1["psnr"])
        psnr2 = float(frame2["psnr"])
        ssim1 = float(frame1["ssim"])
        ssim2 = float(frame2["ssim"])
        lpips1 = float(frame1["lpips"])
        lpips2 = float(frame2["lpips"])

        dif_psnr = psnr2 - psnr1
        dif_ssim = ssim2 - ssim1
        dif_lpips = lpips1 - lpips2

        if dif_psnr < min_psnr:
            if dif_psnr > n:
                min_psnr = dif_psnr
                min_frame_psnr = fname
        
        if dif_psnr > max_psnr:
            max_psnr = dif_psnr
            max_frame_psnr = fname

        if dif_ssim < min_ssim:
            if dif_ssim > n:
                min_ssim = dif_ssim
                min_frame_ssim = fname
        
        if dif_ssim > max_ssim:
            max_ssim = dif_ssim
            max_frame_ssim = fname

        if dif_lpips < min_lpips:
            if dif_lpips > n:
                min_lpips = dif_lpips
                min_frame_lpips = fname
        
        if dif_lpips > max_lpips:
            max_lpips = dif_lpips
            max_frame_lpips = fname

    psnr_enh.append((round(min_psnr, 2), round(max_psnr, 2)))
    score_dict["psnr_min"] = min_frame_psnr
    score_dict["psnr_max"] = max_frame_psnr
    score_dict["psnr_enh"] = psnr_enh

    ssim_enh.append((round(min_ssim, 3), round(max_ssim, 3)))
    score_dict["ssim_min"] = min_frame_ssim
    score_dict["ssim_max"] = max_frame_ssim
    score_dict["ssim_enh"] = ssim_enh

    lpips_enh.append((round(min_lpips, 3), round(max_lpips, 3)))
    score_dict["lpips_min"] = min_frame_lpips
    score_dict["lpips_max"] = max_frame_lpips
    score_dict["lpips_enh"] = lpips_enh

    psnr_processed_dict[f"{base}_proscessed"] = score_dict

# if not save_path.parent.exists():
#         save_path.parent.mkdir(parents=True)
with open(des_path, "w", encoding="UTF-8") as file:
    json.dump(psnr_processed_dict, file, indent=4)
