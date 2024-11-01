# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pathlib
from typing import Callable, List, Optional, Tuple

import fastmri
import h5py
import numpy as np
import torch.utils.data
import os

# -----------------------------------------------------------------------------
#                Single coil knee dataset (as used in MICCAI'20)
# -----------------------------------------------------------------------------
class MICCAI2020Data(torch.utils.data.Dataset):
    # This is the same as fastMRI singlecoil_knee, except we provide a custom test split
    # and also normalize images by the mean norm of the k-space over training data
    KSPACE_WIDTH = 368
    KSPACE_HEIGHT = 640
    START_PADDING = 166
    END_PADDING = 202
    CENTER_CROP_SIZE = 320

    def __init__(
        self,
        root: pathlib.Path,
        transform: Callable,
        num_cols: Optional[int] = None,
        num_volumes: Optional[int] = None,
        num_rand_slices: Optional[int] = None,
        custom_split: Optional[str] = None,
    ):
        self.transform = transform
        self.examples: List[Tuple[pathlib.PurePath, int]] = []

        self.num_rand_slices = num_rand_slices
        self.rng = np.random.RandomState(1234)

        # Remove the last directory from the path
        root = pathlib.Path(os.path.dirname(root))  # New added, will remove "/knee_singlecoil_train"
        
        files = []
        for fname in list(pathlib.Path(root).iterdir()):
            data = h5py.File(fname, "r")
            if num_cols is not None and data["kspace"].shape[2] != num_cols:
                continue
            files.append(fname)

        if custom_split is not None:
            split_info = []
            with open(f"/home/yck/Desktop/GITHUB/Bayesian Reinforcement Learning/active-mri-acquisition_yck/activemri/data/splits/knee_singlecoil/{custom_split}.txt") as f:
                for line in f:
                    split_info.append(line.rsplit("\n")[0])
            files = [f for f in files if f.name in split_info]

        if num_volumes is not None:
            self.rng.shuffle(files)
            files = files[:num_volumes]

        for volume_i, fname in enumerate(sorted(files)):
            data = h5py.File(fname, "r")
            kspace = data["kspace"]

            if num_rand_slices is None:
                num_slices = kspace.shape[0]
                self.examples += [(fname, slice_id) for slice_id in range(num_slices)]
            else:
                slice_ids = list(range(kspace.shape[0]))
                self.rng.seed(seed=volume_i)
                self.rng.shuffle(slice_ids)
                self.examples += [
                    (fname, slice_id) for slice_id in slice_ids[:num_rand_slices]
                ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice_id = self.examples[i]
        with h5py.File(fname, "r") as data:
            kspace = data["kspace"][slice_id]
            kspace = torch.from_numpy(np.stack([kspace.real, kspace.imag], axis=-1))
            kspace = fastmri.ifftshift(kspace, dim=(0, 1))
            target = torch.fft.ifft2(kspace, dim=(0, 1), norm='backward') # new added
            target = fastmri.ifftshift(target, dim=(0, 1))
            # Normalize using mean of k-space in training data
            target /= 7.072103529760345e-07
            kspace /= 7.072103529760345e-07

            # Environment expects numpy arrays. The code above was used with an older
            # version of the environment to generate the results of the MICCAI'20 paper.
            # So, to keep this consistent with the version in the paper, we convert
            # the tensors back to numpy rather than changing the original code.
            kspace = kspace.numpy()
            target = target.numpy()
            return self.transform(
                kspace,
                torch.zeros(kspace.shape[1]),
                target,
                dict(data.attrs),
                fname.name,
                slice_id,
            )
