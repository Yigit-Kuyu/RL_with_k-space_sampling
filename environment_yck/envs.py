# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
activemri.envs.envs.py
====================================
Gym-like environment for active MRI acquisition.
"""
import functools
import json
import os
import pathlib
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Sized,
    Tuple,
    Union,
)

import fastmri.data
import gym
import numpy as np
import torch
import torch.utils.data


import h5py
from typing import Tuple, Union
import fastmri.data.transforms as fastmri_transforms


#from environment_yck import util, masks   




'''
import activemri.data.singlecoil_knee_data as scknee_data
import activemri.data.transforms
import activemri.envs.masks
import activemri.envs.util
import activemri.models
'''

DataInitFnReturnType = Tuple[
    torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset
]





######## Transform functions to process fastMRI data for reconstruction models. ########


TensorType = Union[np.ndarray, torch.Tensor]


def to_magnitude(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    return (tensor ** 2).sum(dim=dim) ** 0.5


def center_crop(x: TensorType, shape: Tuple[int, int]) -> TensorType:
    """Center crops a tensor to the desired 2D shape.

    Args:
        x(union(``torch.Tensor``, ``np.ndarray``)): The tensor to crop.
            Shape should be ``(batch_size, height, width)``.
        shape(tuple(int,int)): The desired shape to crop to.

    Returns:
        (union(``torch.Tensor``, ``np.ndarray``)): The cropped tensor.
    """
    assert len(x.shape) == 3
    assert 0 < shape[0] <= x.shape[1]
    assert 0 < shape[1] <= x.shape[2]
    h_from = (x.shape[1] - shape[0]) // 2
    w_from = (x.shape[2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    x = x[:, h_from:h_to, w_from:w_to]
    return x


def ifft_permute_maybe_shift(
    x: torch.Tensor, normalized: bool = False, ifft_shift: bool = False
) -> torch.Tensor:
    x = x.permute(0, 2, 3, 1)
    y = torch.ifft(x, 2, normalized=normalized)
    if ifft_shift:
        y = fastmri.ifftshift(y, dim=(1, 2))
    return y.permute(0, 3, 1, 2)


def raw_transform_miccai2020(kspace=None, mask=None, **_kwargs):
    """Transform to produce input for reconstructor used in `Pineda et al. MICCAI'20 <https://arxiv.org/pdf/2007.10469.pdf>`_.

    Produces a zero-filled reconstruction and a mask that serve as a input to models of type
    :class:`activemri.models.cvpr10_reconstructor.CVPR19Reconstructor`. The mask is almost
    equal to the mask passed as argument, except that high-frequency padding columns are set
    to 1, and the mask is reshaped to be compatible with the reconstructor.

    Args:
        kspace(``np.ndarray``): The array containing the k-space data returned by the dataset.
        mask(``torch.Tensor``): The masks to apply to the k-space.

    Returns:
        tuple: A tuple containing:
            - ``torch.Tensor``: The zero-filled reconstructor that will be passed to the
              reconstructor.
            - ``torch.Tensor``: The mask to use as input to the reconstructor.
    """
    # alter mask to always include the highest frequencies that include padding
    mask[
        :,
        :,
        MICCAI2020Data.START_PADDING : MICCAI2020Data.END_PADDING,
    ] = 1
    mask = mask.unsqueeze(1)

    all_kspace = []
    for ksp in kspace:
        all_kspace.append(torch.from_numpy(ksp).permute(2, 0, 1))
    k_space = torch.stack(all_kspace)

    masked_true_k_space = torch.where(
        mask.byte(),
        k_space,
        torch.tensor(0.0).to(mask.device),
    )
    reconstructor_input = ifft_permute_maybe_shift(masked_true_k_space, ifft_shift=True)
    return reconstructor_input, mask


# Based on
# https://github.com/facebookresearch/fastMRI/blob/master/experimental/unet/unet_module.py
def _base_fastmri_unet_transform(
    kspace,
    mask,
    ground_truth,
    attrs,
    which_challenge="singlecoil",
):
    kspace = fastmri_transforms.to_tensor(kspace)

    mask = mask[..., : kspace.shape[-2]]  # accounting for variable size masks
    masked_kspace = kspace * mask.unsqueeze(-1) + 0.0

    # inverse Fourier transform to get zero filled solution
    image = fastmri.ifft2c(masked_kspace)

    # crop input to correct size
    if ground_truth is not None:
        crop_size = (ground_truth.shape[-2], ground_truth.shape[-1])
    else:
        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

    # check for FLAIR 203
    if image.shape[-2] < crop_size[1]:
        crop_size = (image.shape[-2], image.shape[-2])

    # noinspection PyTypeChecker
    image = fastmri_transforms.complex_center_crop(image, crop_size)

    # absolute value
    image = fastmri.complex_abs(image)

    # apply Root-Sum-of-Squares if multicoil data
    if which_challenge == "multicoil":
        image = fastmri.rss(image)

    # normalize input
    image, mean, std = fastmri_transforms.normalize_instance(image, eps=1e-11)
    image = image.clamp(-6, 6)

    return image.unsqueeze(0), mean, std


def _batched_fastmri_unet_transform(
    kspace, mask, ground_truth, attrs, which_challenge="singlecoil"
):
    batch_size = len(kspace)
    images, means, stds = [], [], []
    for i in range(batch_size):
        image, mean, std = _base_fastmri_unet_transform(
            kspace[i],
            mask[i],
            ground_truth[i],
            attrs[i],
            which_challenge=which_challenge,
        )
        images.append(image)
        means.append(mean)
        stds.append(std)
    return torch.stack(images), torch.stack(means), torch.stack(stds)


# noinspection PyUnusedLocal
def fastmri_unet_transform_singlecoil(
    kspace=None, mask=None, ground_truth=None, attrs=None, fname=None, slice_id=None
):
    """
    Transform to use as input to fastMRI's Unet model for singlecoil data.

    This is an adapted version of the code found in
    `fastMRI <https://github.com/facebookresearch/fastMRI/blob/master/experimental/unet/unet_module.py#L190>`_.
    """
    return _batched_fastmri_unet_transform(
        kspace, mask, ground_truth, attrs, "singlecoil"
    )


# noinspection PyUnusedLocal
def fastmri_unet_transform_multicoil(
    kspace=None, mask=None, ground_truth=None, attrs=None, fname=None, slice_id=None
):
    """Transform to use as input to fastMRI's Unet model for multicoil data.

    This is an adapted version of the code found in
    `fastMRI <https://github.com/facebookresearch/fastMRI/blob/master/experimental/unet/unet_module.py#L190>`_.
    """
    return _batched_fastmri_unet_transform(
        kspace, mask, ground_truth, attrs, "multicoil"
    )






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
            with open(f"splits/knee_singlecoil/{custom_split}.txt") as f:
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
            target = torch.ifft(kspace, 2, normalized=False)
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





# -----------------------------------------------------------------------------
#                               DATA HANDLING
# -----------------------------------------------------------------------------
class CyclicSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        data_source: Sized,
        order: Optional[Sized] = None,
        loops: int = 1,
    ):
        torch.utils.data.Sampler.__init__(self, data_source)
        assert loops > 0
        assert order is None or len(order) == len(data_source)
        self.data_source = data_source
        self.order = order if order is not None else range(len(self.data_source))
        self.loops = loops

    def _iterator(self):
        for _ in range(self.loops):
            for j in self.order:
                yield j

    def __iter__(self):
        return iter(self._iterator())

    def __len__(self):
        return len(self.data_source) * self.loops


def _env_collate_fn(
    batch: Tuple[Union[np.array, list], ...]
) -> Tuple[Union[np.array, list], ...]:
    ret = []
    for i in range(6):  # kspace, mask, target, attrs, fname, slice_id
        ret.append([item[i] for item in batch])
    return tuple(ret)


class DataHandler:
    def __init__(
        self,
        data_source: torch.utils.data.Dataset,
        seed: Optional[int],
        batch_size: int = 1,
        loops: int = 1,
        collate_fn: Optional[Callable] = None,
    ):
        self._iter = None  # type: Iterator[Any]
        self._collate_fn = collate_fn
        self._batch_size = batch_size
        self._loops = loops
        self._init_impl(data_source, seed, batch_size, loops, collate_fn)

    def _init_impl(
        self,
        data_source: torch.utils.data.Dataset,
        seed: Optional[int],
        batch_size: int = 1,
        loops: int = 1,
        collate_fn: Optional[Callable] = None,
    ):
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(data_source))
        sampler = CyclicSampler(data_source, order, loops=loops)
        if collate_fn:
            self._data_loader = torch.utils.data.DataLoader(
                data_source,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=collate_fn,
            )
        else:
            self._data_loader = torch.utils.data.DataLoader(
                data_source, batch_size=batch_size, sampler=sampler
            )
        self._iter = iter(self._data_loader)

    def reset(self):
        self._iter = iter(self._data_loader)

    def __iter__(self):
        return self._iter

    def __next__(self):
        return next(self._iter)

    def seed(self, seed: int):
        self._init_impl(
            self._data_loader.dataset,
            seed,
            self._batch_size,
            self._loops,
            self._collate_fn,
        )


# -----------------------------------------------------------------------------
#                           BASE ACTIVE MRI ENV
# -----------------------------------------------------------------------------


class ActiveMRIEnv(gym.Env):
    """Base class for all active MRI acquisition environments.

    This class provides the core logic implementation of the k-space acquisition process.
    The class is not to be used directly, but rather one of its subclasses should be
    instantiated. Subclasses of `ActiveMRIEnv` are responsible for data initialization
    and specifying configuration options for the environment.

    Args:
        kspace_shape(tuple(int,int)): Shape of the k-space slices for the dataset.
        num_parallel_episodes(int): Determines the number images that will be processed
                                    simultaneously by :meth:`reset()` and :meth:`step()`.
                                    Defaults to 1.
        budget(optional(int)): The length of an acquisition episode. Defaults to ``None``,
                               which indicates that episode will continue until all k-space
                               columns have been acquired.
        seed(optional(int)): The seed for the environment's random number generator, which is
                             an instance of ``numpy.random.RandomState``. Defaults to ``None``.
        no_checkpoint(optional(bool)): Set to ``True`` if you want to run your reconstructor
                                       model without loading anything from a checkpoint.

    """

    _num_loops_train_data = 100000

    metadata = {"render.modes": ["human"], "video.frames_per_second": None}

    def __init__(
        self,
        kspace_shape: Tuple[int, int],
        num_parallel_episodes: int = 1,
        budget: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        # Default initialization
        self._cfg: Mapping[str, Any] = None
        self._data_location: str = None
        self._reconstructor: Optional[Any] = None # new added
        self._transform: Callable = None
        self._train_data_handler: DataHandler = None
        self._val_data_handler: DataHandler = None
        self._test_data_handler: DataHandler = None
        self._device = torch.device("cpu")
        self._has_setup = False
        self.num_parallel_episodes = num_parallel_episodes
        self.budget = budget

        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self.reward_metric = "mse"

        # Init from provided configuration
        self.kspace_height, self.kspace_width = kspace_shape

        # Gym init
        # Observation is a dictionary
        self.observation_space = None
        self.action_space = gym.spaces.Discrete(self.kspace_width)

        # This is changed by `set_training()`, `set_val()`, `set_test()`
        self._current_data_handler: DataHandler = None

        # These are changed every call to `reset()`
        self._current_ground_truth: torch.Tensor = None
        self._transform_wrapper: Callable = None
        self._current_k_space: torch.Tensor = None
        self._did_reset = False
        self._steps_since_reset = 0
        # These three are changed every call to `reset()` and every call to `step()`
        self._current_reconstruction_numpy: np.ndarray = None
        self._current_score: Dict[str, np.ndarray] = None
        self._current_mask: torch.Tensor = None

    # -------------------------------------------------------------------------
    # Protected methods
    # -------------------------------------------------------------------------
    def _setup(
        self,
        cfg_filename: str,
        data_init_func: Callable[[], DataInitFnReturnType],
    ):
        self._has_setup = True
        self._init_from_config_file(cfg_filename)
        self._setup_data_handlers(data_init_func)

    def _setup_data_handlers(
        self,
        data_init_func: Callable[[], DataInitFnReturnType],
    ):
        train_data, val_data, test_data = data_init_func()
        self._train_data_handler = DataHandler(
            train_data,
            self._seed,
            batch_size=self.num_parallel_episodes,
            loops=self._num_loops_train_data,
            collate_fn=_env_collate_fn,
        )
        self._val_data_handler = DataHandler(
            val_data,
            self._seed + 1 if self._seed else None,
            batch_size=self.num_parallel_episodes,
            loops=1,
            collate_fn=_env_collate_fn,
        )
        self._test_data_handler = DataHandler(
            test_data,
            self._seed + 2 if self._seed else None,
            batch_size=self.num_parallel_episodes,
            loops=1,
            collate_fn=_env_collate_fn,
        )
        self._current_data_handler = self._train_data_handler

    def _init_from_config_dict(self, cfg: Mapping[str, Any]):
        from environment_yck import util  # Local import
        
        self._cfg = cfg
        self._data_location = cfg["data_location"]
        if not os.path.isdir(self._data_location):
            default_cfg, defaults_fname = util.get_defaults_json()
            self._data_location = default_cfg["data_location"]
            if not os.path.isdir(self._data_location) and self._has_setup:
                raise RuntimeError(
                    f"No 'data_location' key found in the given config. Please "
                    f"write dataset location in your JSON config, or in file {defaults_fname} "
                    f"(to use as a default)."
                )
        self._device = torch.device(cfg["device"])
        self.reward_metric = cfg["reward_metric"]
        if self.reward_metric not in ["mse", "ssim", "psnr", "nmse"]:
            raise ValueError("Reward metric must be one of mse, nmse, ssim, or psnr.")
        #cfg["mask"]["function"]='masks.sample_low_frequency_mask'  # new added
        mask_func = util.import_object_from_str(cfg["mask"]["function"], 1) 
        self._mask_func = functools.partial(mask_func, cfg["mask"]["args"])

        # Instantiating reconstructor
        reconstructor_cfg = cfg["reconstructor"]
        reconstructor_cls = util.import_object_from_str(reconstructor_cfg["cls"], 2)

        checkpoint_fname = pathlib.Path(reconstructor_cfg["checkpoint_fname"])
        default_cfg, defaults_fname = util.get_defaults_json()
        saved_models_dir = default_cfg["saved_models_dir"]
        checkpoint_path = pathlib.Path(saved_models_dir) / checkpoint_fname
        if self._has_setup and not checkpoint_path.is_file():
            raise RuntimeError(
                f"No checkpoint was found at {str(checkpoint_path)}. "
                f"Please make sure that both 'checkpoint_fname' (in your JSON config) "
                f"and 'saved_models_dir' (in {defaults_fname}) are configured correctly."
            )

        checkpoint = (
            torch.load(str(checkpoint_path)) if checkpoint_path.is_file() else None
        )
        options = reconstructor_cfg["options"]
        if checkpoint and "options" in checkpoint:
            msg = (
                f"Checkpoint at {checkpoint_path.name} has an 'options' key. "
                f"This will override the options defined in configuration file."
            )
            warnings.warn(msg)
            options = checkpoint["options"]
            assert isinstance(options, dict)
        self._reconstructor = reconstructor_cls(**options)
        self._reconstructor.init_from_checkpoint(checkpoint)
        self._reconstructor.eval()
        self._reconstructor.to(self._device)
        self._transform = util.import_object_from_str(reconstructor_cfg["transform"], 3)

    def _init_from_config_file(self, config_filename: str):
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))  # new added
        
        # Go up one level to the RL_with_k-space_sampling directory
        project_root = os.path.dirname(current_dir) # new added
        
        # Construct the correct path to the config file
        config_path = os.path.join(project_root, "configs", config_filename) # new added
        
        with open(config_filename, "rb") as f:
            self._init_from_config_dict(json.load(f))

    @staticmethod
    def _void_transform(
        kspace: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
        attrs: List[Dict[str, Any]],
        fname: List[str],
        slice_id: List[int],
    ) -> Tuple:
        return kspace, mask, target, attrs, fname, slice_id

    def _send_tuple_to_device(self, the_tuple: Tuple[Union[Any, torch.Tensor]]):
        the_tuple_device = []
        for i in range(len(the_tuple)):
            if isinstance(the_tuple[i], torch.Tensor):
                the_tuple_device.append(the_tuple[i].to(self._device))
            else:
                the_tuple_device.append(the_tuple[i])
        return tuple(the_tuple_device)

    @staticmethod
    def _send_dict_to_cpu_and_detach(the_dict: Dict[str, Union[Any, torch.Tensor]]):
        the_dict_cpu = {}
        for key in the_dict:
            if isinstance(the_dict[key], torch.Tensor):
                the_dict_cpu[key] = the_dict[key].detach().cpu()
            else:
                the_dict_cpu[key] = the_dict[key]
        return the_dict_cpu

    def _compute_obs_and_score(
        self, override_current_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        mask_to_use = (
            override_current_mask
            if override_current_mask is not None
            else self._current_mask
        )
        reconstructor_input = self._transform_wrapper(
            kspace=self._current_k_space,
            mask=mask_to_use,
            ground_truth=self._current_ground_truth,
        )

        reconstructor_input = self._send_tuple_to_device(reconstructor_input)
        with torch.no_grad():
            extra_outputs = self._reconstructor(*reconstructor_input)

        extra_outputs = self._send_dict_to_cpu_and_detach(extra_outputs)
        reconstruction = extra_outputs["reconstruction"]

        # this dict is only for storing the other outputs
        del extra_outputs["reconstruction"]

        # noinspection PyUnusedLocal
        reconstructor_input = None  # de-referencing GPU tensors

        score = self._compute_score_given_tensors(
            *self._process_tensors_for_score_fns(
                reconstruction, self._current_ground_truth
            )
        )

        obs = {
            "reconstruction": reconstruction,
            "extra_outputs": extra_outputs,
            "mask": self._current_mask.clone().view(self._current_mask.shape[0], -1),
        }

        return obs, score

    def _clear_cache_and_unset_did_reset(self):
        self._current_mask = None
        self._current_ground_truth = None
        self._current_reconstruction_numpy = None
        self._transform_wrapper = None
        self._current_k_space = None
        self._current_score = None
        self._steps_since_reset = 0
        self._did_reset = False

    # noinspection PyMethodMayBeStatic
    def _process_tensors_for_score_fns(
        self, reconstruction: torch.Tensor, ground_truth: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return reconstruction, ground_truth

    @staticmethod
    def _compute_score_given_tensors(reconstruction: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, np.ndarray]:
        from environment_yck import util  # Local import

        mse = util.compute_mse(reconstruction, ground_truth)
        nmse =util.compute_nmse(reconstruction, ground_truth)
        ssim = util.compute_ssim(reconstruction, ground_truth)
        psnr = util.compute_psnr(reconstruction, ground_truth)

        return {"mse": mse, "nmse": nmse, "ssim": ssim, "psnr": psnr}

    @staticmethod
    def _convert_to_gray(array: np.ndarray) -> np.ndarray:
        M = np.max(array)
        m = np.min(array)
        return (255 * (array - m) / (M - m)).astype(np.uint8)

    @staticmethod
    def _render_arrays(
        ground_truth: np.ndarray, reconstruction: np.ndarray, mask: np.ndarray
    ) -> List[np.ndarray]:
        batch_size, img_height, img_width = ground_truth.shape
        frames = []
        for i in range(batch_size):
            mask_i = np.tile(mask[i], (1, img_height, 1))

            pad = 32
            mask_begin = pad
            mask_end = mask_begin + mask.shape[-1]
            gt_begin = mask_end + pad
            gt_end = gt_begin + img_width
            rec_begin = gt_end + pad
            rec_end = rec_begin + img_width
            error_begin = rec_end + pad
            error_end = error_begin + img_width
            frame = 128 * np.ones((img_height, error_end + pad), dtype=np.uint8)
            frame[:, mask_begin:mask_end] = 255 * mask_i
            frame[:, gt_begin:gt_end] = ActiveMRIEnv._convert_to_gray(ground_truth[i])
            frame[:, rec_begin:rec_end] = ActiveMRIEnv._convert_to_gray(
                reconstruction[i]
            )
            rel_error = np.abs((ground_truth[i] - reconstruction[i]) / ground_truth[i])
            frame[:, error_begin:error_end] = 255 * rel_error.astype(np.uint8)

            frames.append(frame)
        return frames

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------
    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Starts a new acquisition episode with a batch of images.

        This methods performs the following steps:

            1. Reads a batch of images from the environment's dataset.
            2. Creates an initial acquisition mask for each image.
            3. Passes the loaded data and the initial masks to the transform function,
               producing a batch of inputs for the environment's reconstructor model.
            4. Calls the reconstructor model on this input and returns its output
               as an observation.

        The observation returned is a dictionary with the following keys:
            - *"reconstruction"(torch.Tensor):* The reconstruction produced by the
              environment's reconstructor model, using the current
              acquisition mask.
            - *"extra_outputs"(dict(str,Any)):* A dictionary with any additional
              outputs produced by the reconstructor  (e.g., uncertainty maps).
            - *"mask"(torch.Tensor):* The current acquisition mask.

        Returns:
            tuple: tuple containing:
                   - obs(dict(str,any): Observation dictionary.
                   - metadata(dict(str,any): Metadata information containing the following keys:

                        - *"fname"(list(str)):* the filenames of the image read from the dataset.
                        - *"slice_id"(list(int)):* slice indices for each image within the volume.
                        - *"current_score"(dict(str,float):* A dictionary with the error measures
                          for the reconstruction (e.g., "mse", "nmse", "ssim", "psnr"). The measures
                          considered can be obtained with :meth:`score_keys()`.
        """
        self._did_reset = True
        try:
            kspace, _, ground_truth, attrs, fname, slice_id = next(
                self._current_data_handler
            )
        except StopIteration:
            return {}, {}

        self._current_ground_truth = torch.from_numpy(np.stack(ground_truth))

        # Converting k-space to torch is better handled by transform,
        # since we have both complex and non-complex versions
        self._current_k_space = kspace

        self._transform_wrapper = functools.partial(
            self._transform, attrs=attrs, fname=fname, slice_id=slice_id
        )
        kspace_shapes = [tuple(k.shape) for k in kspace]
        self._current_mask = self._mask_func(kspace_shapes, self._rng, attrs=attrs)
        obs, self._current_score = self._compute_obs_and_score()
        self._current_reconstruction_numpy = obs["reconstruction"].cpu().numpy()
        self._steps_since_reset = 0

        meta = {
            "fname": fname,
            "slice_id": slice_id,
            "current_score": self._current_score,
        }
        return obs, meta

    def step(self, action: Union[int, Sequence[int]]) -> Tuple[Dict[str, Any], np.ndarray, List[bool], Dict]:
        
        from environment_yck import masks  # Local import
        
        """Performs a step of active MRI acquisition.

        Given a set of indices for k-space columns to acquire, updates the current batch
        of masks with their corresponding indices, creates a new batch of reconstructions,
        and returns the corresponding observations and rewards (for the observation format
        see :meth:`reset()`). The reward is the improvement in score with
        respect to the reconstruction before adding the indices. The specific score metric
        used is determined by ``env.reward_metric``.

        The method also returns a list of booleans, indicating whether any episodes in the
        batch have already concluded.

        The last return value is a metadata dictionary. It contains a single key
        "current_score", which contains a dictionary with the error measures for the
        reconstruction (e.g., ``"mse", "nmse", "ssim", "psnr"``). The measures
        considered can be obtained with :meth:`score_keys()`.

        Args:
            action(union(int, sequence(int))): Indices for k-space columns to acquire. The
                                               length of the sequence must be equal to the
                                               current number of parallel episodes
                                               (i.e., ``obs["reconstruction"].shape[0]``).
                                               If only an ``int`` is passed, the index will
                                               be replicated for the whole batch of episodes.

        Returns:
            tuple: The transition information in the order
            ``(next_observation, reward, done, meta)``. The types and shapes are:

              - ``next_observation(dict):`` Dictionary format (see :meth:`reset()`).
              - ``reward(np.ndarray)``: length equal to current number of parallel
                episodes.
              - ``done(list(bool))``: same length as ``reward``.
              - ``meta(dict)``: see description above.

        """
        if not self._did_reset:
            raise RuntimeError(
                "Attempting to call env.step() before calling env.reset()."
            )
        if isinstance(action, int):
            action = [action for _ in range(self.num_parallel_episodes)]
        self._current_mask = masks.update_masks_from_indices(
            self._current_mask, action
        )
        obs, new_score = self._compute_obs_and_score()
        self._current_reconstruction_numpy = obs["reconstruction"].cpu().numpy()

        reward = new_score[self.reward_metric] - self._current_score[self.reward_metric]
        if self.reward_metric in ["mse", "nmse"]:
            reward *= -1
        else:
            assert self.reward_metric in ["ssim", "psnr"]
        self._current_score = new_score
        self._steps_since_reset += 1

        done = masks.check_masks_complete(self._current_mask)
        if self.budget and self._steps_since_reset >= self.budget:
            done = [True] * len(done)
        return obs, reward, done, {"current_score": self._current_score}

    def try_action(self, action: Union[int, Sequence[int]]) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        from environment_yck import masks  # Local import
        
        """Simulates the effects of actions without changing the environment's state.

        This method operates almost exactly as :meth:`step()`, with the exception that
        the environment's state is not altered. The method returns the next observation
        and the resulting reconstruction score after applying the give k-space columns to
        each image in the current batch of episodes.

        Args:
            action(union(int, sequence(int))): Indices for k-space columns to acquire. The
                                               length of the sequence must be equal to the
                                               current number of parallel episodes
                                               (i.e., ``obs["reconstruction"].shape[0]``).
                                               If only an ``int`` is passed, the index will
                                               be replicated for the whole batch of episodes.

        Returns:
            tuple: The reconstruction information in the order
            ``(next_observation, current_score)``. The types and shapes are:

              - ``next_observation(dict):`` Dictionary format (see :meth:`reset()`).
              - ``current_score(dict(str, float))``: A dictionary with the error measures
                  for the reconstruction (e.g., "mse", "nmse", "ssim", "psnr"). The measures
                  considered can be obtained with `ActiveMRIEnv.score_keys()`.

        """
        if not self._did_reset:
            raise RuntimeError(
                "Attempting to call env.try_action() before calling env.reset()."
            )
        if isinstance(action, int):
            action = [action for _ in range(self.num_parallel_episodes)]
        new_mask = masks.update_masks_from_indices(
            self._current_mask, action
        )
        obs, new_score = self._compute_obs_and_score(override_current_mask=new_mask)

        return obs, new_score

    def render(self, mode="human"):
        """Renders information about the environment's current state.

        Returns:
            ``np.ndarray``: An image frame containing, from left to right: current
                            acquisition mask, current ground image, current reconstruction,
                            and current relative reconstruction error.
        """
        pass

    def seed(self, seed: Optional[int] = None):
        """Sets the seed for the internal number generator.

        This seeds affects the order of the data loader for all loop modalities (i.e.,
        training, validation, test).

        Args:
            seed(optional(int)): The seed for the environment's random number generator.
        """
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._train_data_handler.seed(seed)
        self._val_data_handler.seed(seed)
        self._test_data_handler.seed(seed)

    def set_training(self, reset: bool = False):
        """Sets the environment to use the training data loader.

        Args:
            reset(bool): If ``True``, also resets the data loader so that it starts again
                         from the first image in the loop order.

        Warning:
            After this method is called the ``env.reset()`` needs to be called again, otherwise
            an exception will be thrown.
        """
        if reset:
            self._train_data_handler.reset()
        self._current_data_handler = self._train_data_handler
        self._clear_cache_and_unset_did_reset()

    def set_val(self, reset: bool = True):
        """Sets the environment to use the validation data loader.

        Args:
            reset(bool): If ``True``, also resets the data loader so that it starts again
                         from the first image in the loop order.

        Warning:
            After this method is called the ``env.reset()`` needs to be called again, otherwise
            an exception will be thrown.
        """
        if reset:
            self._val_data_handler.reset()
        self._current_data_handler = self._val_data_handler
        self._clear_cache_and_unset_did_reset()

    def set_test(self, reset: bool = True):
        """Sets the environment to use the test data loader.

        Args:
            reset(bool): If ``True``, also resets the data loader so that it starts again
                         from the first image in the loop order.

        Warning:
            After this method is called the ``env.reset()`` needs to be called again, otherwise
            an exception will be thrown.
        """
        if reset:
            self._test_data_handler.reset()
        self._current_data_handler = self._test_data_handler
        self._clear_cache_and_unset_did_reset()

    @staticmethod
    def score_keys() -> List[str]:
        """ Returns the list of score metric names used by this environment. """
        return ["mse", "nmse", "ssim", "psnr"]


# -----------------------------------------------------------------------------
#                             CUSTOM ENVIRONMENTS
# -----------------------------------------------------------------------------
class MICCAI2020Env(ActiveMRIEnv):
    """Implementation of environment used for *Pineda et al., MICCAI 2020*.

    This environment is provided to facilitate replication of the experiments performed
    in *Luis Pineda, Sumana Basu, Adriana Romero, Roberto Calandra, Michal Drozdzal,
    "Active MR k-space Sampling with Reinforcement Learning". MICCAI 2020.*

    The dataset is the same as that of :class:`SingleCoilKneeEnv`, except that we provide
    a custom validation/test split of the original validation data. The environment's
    configuration file is set to use the reconstruction model used in the paper
    (see :class:`activemri.models.cvpr19_reconstructor.CVPR19Reconstructor`), as well
    as the proper transform to generate inputs for this model.

    The k-space shape of this environment is set to ``(640, 368)``.

    Args:
        num_parallel_episodes(int): Determines the number images that will be processed
                                    simultaneously by :meth:`reset()` and :meth:`step()`.
                                    Defaults to 1.
        budget(optional(int)): The length of an acquisition episode. Defaults to ``None``,
                               which indicates that episode will continue until all k-space
                               columns have been acquired.
        seed(optional(int)): The seed for the environment's random number generator, which is
                             an instance of ``numpy.random.RandomState``. Defaults to ``None``.
        extreme(bool): ``True`` or ``False`` for running extreme acceleration or normal
                       acceleration scenarios described in the paper, respectively.
    """

    KSPACE_WIDTH = MICCAI2020Data.KSPACE_WIDTH
    START_PADDING = MICCAI2020Data.START_PADDING
    END_PADDING = MICCAI2020Data.END_PADDING
    CENTER_CROP_SIZE = MICCAI2020Data.CENTER_CROP_SIZE

    def __init__(
        self,
        num_parallel_episodes: int = 1,
        budget: Optional[int] = None,
        seed: Optional[int] = None,
        extreme: bool = False,
        obs_includes_padding: bool = True,
    ):
        super().__init__(
            (640, self.KSPACE_WIDTH),
            num_parallel_episodes=num_parallel_episodes,
            budget=budget,
            seed=seed,
        )
        import os
        # Get the directory of the activemri package
        activemri_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # new added
        
        # Construct the absolute path to the config file
        config_path1 = os.path.join(activemri_dir, "configs", "miccai-2020-extreme-acc.json") # new added
        config_path2 = os.path.join(activemri_dir, "configs", "miccai-2020-normal-acc.json") # new added
        if extreme: # False
            self._setup("configs/miccai-2020-extreme-acc.json", self._create_dataset) # orj
            #self._setup(config_path1, self._create_dataset)
        else: # True
            self._setup("configs/miccai-2020-normal-acc.json", self._create_dataset) # orj
            #self._setup(config_path2, self._create_dataset)
        self.obs_includes_padding = obs_includes_padding

    # -------------------------------------------------------------------------
    # Protected methods
    # -------------------------------------------------------------------------
    def _create_dataset(self) -> DataInitFnReturnType:
        root_path = pathlib.Path(self._data_location)
        train_path = root_path / "knee_singlecoil_train"
        val_and_test_path = root_path / "knee_singlecoil_val"

        train_data = MICCAI2020Data(
            train_path,
            ActiveMRIEnv._void_transform,
            num_cols=self.KSPACE_WIDTH,
        )
        val_data = MICCAI2020Data(
            val_and_test_path,
            ActiveMRIEnv._void_transform,
            custom_split="val",
            num_cols=self.KSPACE_WIDTH,
        )
        test_data = MICCAI2020Data(
            val_and_test_path,
            ActiveMRIEnv._void_transform,
            custom_split="test",
            num_cols=self.KSPACE_WIDTH,
        )
        return train_data, val_data, test_data

    def _process_tensors_for_score_fns(
        self, reconstruction: torch.Tensor, ground_truth: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute magnitude (for metrics)
        reconstruction = to_magnitude(reconstruction, dim=3)
        ground_truth = to_magnitude(ground_truth, dim=3)

        reconstruction = center_crop(
            reconstruction, (self.CENTER_CROP_SIZE, self.CENTER_CROP_SIZE)
        )
        ground_truth = center_crop(
            ground_truth, (self.CENTER_CROP_SIZE, self.CENTER_CROP_SIZE)
        )
        return reconstruction, ground_truth

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------
    def reset(
        self,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        obs, meta = super().reset()
        if not obs:
            return obs, meta
        if self.obs_includes_padding:
            obs["mask"][:, self.START_PADDING : self.END_PADDING] = 1
        return obs, meta

    def step(
        self, action: Union[int, Sequence[int]]
    ) -> Tuple[Dict[str, Any], np.ndarray, List[bool], Dict]:
        obs, reward, done, meta = super().step(action)
        if self.obs_includes_padding:
            obs["mask"][:, self.START_PADDING : self.END_PADDING] = 1
        return obs, reward, done, meta

    def render(self, mode="human"):
        gt = self._current_ground_truth.cpu().numpy()
        rec = self._current_reconstruction_numpy

        gt = center_crop(
            (gt ** 2).sum(axis=3) ** 0.5, (self.CENTER_CROP_SIZE, self.CENTER_CROP_SIZE)
        )
        rec = center_crop(
            (rec ** 2).sum(axis=3) ** 0.5,
            (self.CENTER_CROP_SIZE, self.CENTER_CROP_SIZE),
        )
        return ActiveMRIEnv._render_arrays(gt, rec, self._current_mask.cpu().numpy())


class FastMRIEnv(ActiveMRIEnv):
    """Base class for all fastMRI environments.

    This class can be used to instantiate active acquisition environments using fastMRI
    data. However, for convenience we provide subclasses of ``FastMRIEnv`` with
    default configuration options for each dataset:

        - :class:`SingleCoilKneeEnv`
        - :class:`MultiCoilKneeEnv`
        - :class:`SingleCoilBrainEnv`
        - :class:`MultiCoilKneeEnv`

    Args:
        config_path(str): The path to the JSON configuration file.
        dataset_name(str): One of "knee_singlecoil", "multicoil" (for knee),
                           "brain_multicoil". Primarily used to locate the fastMRI
                           dataset in the user's fastMRI data root folder.
        num_parallel_episodes(int): Determines the number images that will be processed
                                    simultaneously by :meth:`reset()` and :meth:`step()`.
                                    Defaults to 1.
        budget(optional(int)): The length of an acquisition episode. Defaults to ``None``,
                               which indicates that episode will continue until all k-space
                               columns have been acquired.
        seed(optional(int)): The seed for the environment's random number generator, which is
                             an instance of ``numpy.random.RandomState``. Defaults to ``None``.
        num_cols(sequence(int)): Used to filter k-space data to only use images whose k-space
                                 width is in this tuple. Defaults to ``(368, 372)``.
    """

    def __init__(
        self,
        config_path: str,
        dataset_name: str,
        num_parallel_episodes: int = 1,
        budget: Optional[int] = None,
        seed: Optional[int] = None,
        num_cols: Sequence[int] = (368, 372),
    ):
        assert dataset_name in ["knee_singlecoil", "multicoil", "brain_multicoil"]
        challenge = "singlecoil" if dataset_name == "knee_singlecoil" else "multicoil"
        super().__init__(
            (640, np.max(num_cols)),
            num_parallel_episodes=num_parallel_episodes,
            budget=budget,
            seed=seed,
        )
        self.num_cols = num_cols
        self.dataset_name = dataset_name
        self.challenge = challenge
        self._setup(config_path, self._create_dataset)

    def _create_dataset(self) -> DataInitFnReturnType:
        from environment_yck import util  # Local import

        root_path = pathlib.Path(self._data_location)
        datacache_dir = util.maybe_create_datacache_dir()

        train_path = root_path / f"{self.dataset_name}_train"
        val_path = root_path / f"{self.dataset_name}_val"
        val_cache_file = datacache_dir / f"val_{self.dataset_name}_cache.pkl"
        test_path = root_path / f"{self.dataset_name}_test"
        test_cache_file = datacache_dir / f"test_{self.dataset_name}_cache.pkl"

        if not test_path.is_dir():
            warnings.warn(
                f"No test directory found for {self.dataset_name}. "
                f"I will use val directory for test model (env.set_test())."
            )
            test_path = val_path
            test_cache_file = val_cache_file

        train_data = fastmri.data.SliceDataset(
            train_path,
            ActiveMRIEnv._void_transform,
            challenge=self.challenge,
            num_cols=self.num_cols,
            dataset_cache_file=datacache_dir / f"train_{self.dataset_name}_cache.pkl",
        )
        val_data = fastmri.data.SliceDataset(
            val_path,
            ActiveMRIEnv._void_transform,
            challenge=self.challenge,
            num_cols=self.num_cols,
            dataset_cache_file=val_cache_file,
        )
        test_data = fastmri.data.SliceDataset(
            test_path,
            ActiveMRIEnv._void_transform,
            challenge=self.challenge,
            num_cols=self.num_cols,
            dataset_cache_file=test_cache_file,
        )
        return train_data, val_data, test_data

    def render(self, mode="human"):
        return ActiveMRIEnv._render_arrays(
            self._current_ground_truth.cpu().numpy(),
            self._current_reconstruction_numpy,
            self._current_mask.cpu().numpy(),
        )


class SingleCoilKneeEnv(FastMRIEnv):
    """Convenience class to access single-coil knee data.

    Loads the configuration from ``configs/single-coil-knee.json``.
    Looks for datasets named "knee_singlecoil_{train/val/test}" under the ``data_location`` dir.
    If "test" is not found, it uses "val" folder for test mode.

    Args:
        num_parallel_episodes(int): Determines the number images that will be processed
                                    simultaneously by :meth:`reset()` and :meth:`step()`.
                                    Defaults to 1.
        budget(optional(int)): The length of an acquisition episode. Defaults to ``None``,
                               which indicates that episode will continue until all k-space
                               columns have been acquired.
        seed(optional(int)): The seed for the environment's random number generator, which is
                             an instance of ``numpy.random.RandomState``. Defaults to ``None``.
        num_cols(sequence(int)): Used to filter k-space data to only use images whose k-space
                                 width is in this tuple. Defaults to ``(368, 372)``.
    """

    def __init__(
        self,
        num_parallel_episodes: int = 1,
        budget: Optional[int] = None,
        seed: Optional[int] = None,
        num_cols: Sequence[int] = (368, 372),
    ):
        super().__init__(
            "configs/single-coil-knee.json",
            "knee_singlecoil",
            num_parallel_episodes=num_parallel_episodes,
            budget=budget,
            seed=seed,
            num_cols=num_cols,
        )


class MultiCoilKneeEnv(FastMRIEnv):
    """Convenience class to access multi-coil knee data.

    Loads the configuration from ``configs/multi-coil-knee.json``.
    Looks for datasets named "multicoil_{train/val/test}" under default ``data_location`` dir.
    If "test" is not found, it uses "val" folder for test mode.

    Args:
        num_parallel_episodes(int): Determines the number images that will be processed
                                    simultaneously by :meth:`reset()` and :meth:`step()`.
                                    Defaults to 1.
        budget(optional(int)): The length of an acquisition episode. Defaults to ``None``,
                               which indicates that episode will continue until all k-space
                               columns have been acquired.
        seed(optional(int)): The seed for the environment's random number generator, which is
                             an instance of ``numpy.random.RandomState``. Defaults to ``None``.
        num_cols(sequence(int)): Used to filter k-space data to only use images whose k-space
                                 width is in this tuple. Defaults to ``(368, 372)``.
    """

    def __init__(
        self,
        num_parallel_episodes: int = 1,
        budget: Optional[int] = None,
        seed: Optional[int] = None,
        num_cols: Sequence[int] = (368, 372),
    ):

        super().__init__(
            "configs/multi-coil-knee.json",
            "multicoil",
            num_parallel_episodes=num_parallel_episodes,
            budget=budget,
            seed=seed,
            num_cols=num_cols,
        )
