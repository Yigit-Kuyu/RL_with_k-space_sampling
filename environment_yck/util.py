# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import json
import pathlib

from typing import Dict, Tuple

import numpy as np
import skimage.metrics
import torch


def get_user_dir() -> pathlib.Path:
    return pathlib.Path.home() / ".activemri"


def maybe_create_datacache_dir() -> pathlib.Path:
    datacache_dir = get_user_dir() / "__datacache__"
    if not datacache_dir.is_dir():
        datacache_dir.mkdir()
    return datacache_dir


def get_defaults_json() -> Tuple[Dict[str, str], str]:
    defaults_path = get_user_dir() / "defaults.json"
    if not pathlib.Path.exists(defaults_path):
        parent = defaults_path.parents[0]
        parent.mkdir(exist_ok=True)
        content = {"data_location": "", "saved_models_dir": ""}
        with defaults_path.open("w", encoding="utf-8") as f:
            json.dump(content, f)
    else:
        with defaults_path.open("r", encoding="utf-8") as f:
            content = json.load(f)
    return content, str(defaults_path)


def import_object_from_str(classname: str, path_num: int):
    import environment_yck as masks


    the_module, the_object = classname.rsplit(".", 1)
    the_object = classname.split(".")[-1]

    if path_num == 1:
        the_module = 'environment_yck.masks'
        the_object = 'sample_low_frequency_mask'
        '''
        the_module:
        'activemri.envs.masks'
        the_object:
        'sample_low_frequency_mask'

        '''
    elif path_num == 2:
        the_module = 'models.cvpr19_reconstructor'
        the_object = 'CVPR19Reconstructor'
        '''
        the_module:
        'activemri.models.cvpr19_reconstructor'
        the_object:
        'CVPR19Reconstructor
        
        '''
    elif path_num == 3:
        the_module = 'data.transforms'
        the_object = 'raw_transform_miccai2020'
        '''
        the_module:
        'activemri.data.transforms'
        the_object:
        'raw_transform_miccai2020'
        '''
    
    
    
    module = importlib.import_module(the_module)
    return getattr(module, the_object)


def compute_ssim(xs: torch.Tensor, ys: torch.Tensor) -> np.ndarray:
    ssims = []
    for i in range(xs.shape[0]):
        #ssim = skimage.metrics.structural_similarity(
        #    xs[i].cpu().numpy(),
        #    ys[i].cpu().numpy(),
        #    data_range=ys[i].cpu().numpy().max(),
        #)
        
        if np.iscomplexobj(xs[i]) or np.iscomplexobj(ys[i]): # new added
            # Convert complex to magnitude
            xs_real = np.abs(xs[i].cpu().numpy())
            ys_real = np.abs(ys[i].cpu().numpy())
        else:
            xs_real = xs[i].cpu().numpy()
            ys_real = ys[i].cpu().numpy()

        ssim = skimage.metrics.structural_similarity(
            xs_real,
            ys_real,
            data_range=ys_real.max(),
        )
        ssims.append(ssim)
    return np.array(ssims, dtype=np.float32)


def compute_psnr(xs: torch.Tensor, ys: torch.Tensor) -> np.ndarray:
    psnrs = []
    for i in range(xs.shape[0]):
        #psnr = skimage.metrics.peak_signal_noise_ratio(
        #    xs[i].cpu().numpy(),
        #    ys[i].cpu().numpy(),
        #    data_range=ys[i].cpu().numpy().max(),
        #)
        
        if np.iscomplexobj(xs[i]) or np.iscomplexobj(ys[i]): # new added
            xs_real = np.abs(xs[i].cpu().numpy())
            ys_real = np.abs(ys[i].cpu().numpy())
        else:
            xs_real = xs[i].cpu().numpy()
            ys_real = ys[i].cpu().numpy()

        psnr = skimage.metrics.peak_signal_noise_ratio(xs_real, ys_real, data_range=ys_real.max())
        psnrs.append(psnr)
    return np.array(psnrs, dtype=np.float32)


def compute_mse(xs: torch.Tensor, ys: torch.Tensor) -> np.ndarray:
    dims = tuple(range(1, len(xs.shape)))
    return np.mean((ys.cpu().numpy() - xs.cpu().numpy()) ** 2, axis=dims)


def compute_nmse(xs: torch.Tensor, ys: torch.Tensor) -> np.ndarray:
    ys_numpy = ys.cpu().numpy()
    nmses = []
    for i in range(xs.shape[0]):
        x = xs[i].cpu().numpy()
        y = ys_numpy[i]
        nmse = np.linalg.norm(y - x) ** 2 / np.linalg.norm(y) ** 2
        nmses.append(nmse)
    return np.array(nmses, dtype=np.float32)
