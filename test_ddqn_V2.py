# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
import os
import sys

#import activemri.baselines.ddqn as ddqn
#import activemri.envs as envs
#Add the parent directory of environment_yck to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import environment_yck as envs
import environment_yck.envs as mri_envs


import argparse
import torch
import torch.nn as nn
import numpy as np
import sys
import logging
import tempfile
import pickle
from typing import Optional, Dict, Any, List, Tuple
import filelock
import random
import math
import torch.optim as optim
import torch.nn.functional as F
import functools
import time



###################### Interface Classs for Policy ######################

import abc
class Policy:
    """ A basic policy interface. """

    def __init__(self, *args, **kwargs):
        pass


    @abc.abstractmethod
    def get_action(self, obs: Dict[str, Any], **kwargs: Any) -> List[int]:
        """ Returns a list of actions for a batch of observations. """
        pass

    def __call__(self, obs: Dict[str, Any], **kwargs: Any) -> List[int]:
        return self.get_action(obs, **kwargs)


###################### Replay Memory ######################

class replay_buffer:
    """Replay memory of transitions (ot, at, o_t+1, r_t+1).

    Args:
        capacity(int): How many transitions can be stored. After capacity is reached early
                transitions are overwritten in FIFO fashion.
        obs_shape(np.array): The shape of the numpy arrays representing observations.
        batch_size(int): The size of batches returned by the replay buffer.
        burn_in(int): While the replay buffer has lesser entries than this number,
                :meth:`sample()` will return ``None``. Indicates a burn-in period before
                training.
        use_normalization(bool): If ``True``, the replay buffer will keep running mean
                and standard deviation for the observations. Defaults to ``False``.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: np.array,
        batch_size: int,
        burn_in: int,
        use_normalization: bool = False,
    ):
        assert burn_in >= batch_size
        self.batch_size = batch_size
        self.burn_in = burn_in
        self.observations = torch.zeros(capacity, *obs_shape, dtype=torch.float32)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.next_observations = torch.zeros(capacity, *obs_shape, dtype=torch.float32)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.dones = torch.zeros(capacity, dtype=torch.bool)

        self.position = 0
        self.mean_obs = torch.zeros(obs_shape, dtype=torch.float32)
        self.std_obs = torch.ones(obs_shape, dtype=torch.float32)
        self._m2_obs = torch.ones(obs_shape, dtype=torch.float32)
        self.count_seen = 1

        if not use_normalization:
            self._normalize = lambda x: x  # type: ignore
            self._denormalize = lambda x: x  # type: ignore

    def _normalize(self, observation: torch.Tensor) -> Optional[torch.Tensor]:
        if observation is None:
            return None
        return (observation - self.mean_obs) / self.std_obs

    def _denormalize(self, observation: torch.Tensor) -> Optional[torch.Tensor]:
        if observation is None:
            return None
        return self.std_obs * observation + self.mean_obs

    def _update_stats(self, observation: torch.Tensor):
        self.count_seen += 1
        delta = observation - self.mean_obs
        self.mean_obs = self.mean_obs + delta / self.count_seen
        delta2 = observation - self.mean_obs
        self._m2_obs = self._m2_obs + (delta * delta2)
        self.std_obs = np.sqrt(self._m2_obs / (self.count_seen - 1))

    def push(
        self,
        observation: np.array,
        action: int,
        next_observation: np.array,
        reward: float,
        done: bool,
    ):
        """ Pushes a transition into the replay buffer. """
        self.observations[self.position] = observation.clone()
        self.actions[self.position] = torch.tensor([action], dtype=torch.long)
        self.next_observations[self.position] = next_observation.clone()
        self.rewards[self.position] = torch.tensor([reward], dtype=torch.float32)
        self.dones[self.position] = torch.tensor([done], dtype=torch.bool)

        self._update_stats(self.observations[self.position])
        self.position = (self.position + 1) % len(self)

    def sample(self) -> Optional[Dict[str, Optional[torch.Tensor]]]:
        """Samples a batch of transitions from the replay buffer.


        Returns:
            Dictionary(str, torch.Tensor): Contains keys for "observations",
            "next_observations", "actions", "rewards", "dones". If the number of entries
            in the buffer is less than ``self.burn_in``, then returns ``None`` instead.
        """
        if self.count_seen - 1 < self.burn_in:
            return None
        indices = np.random.choice(min(self.count_seen - 1, len(self)), self.batch_size)
        return {
            "observations": self._normalize(self.observations[indices]),
            "next_observations": self._normalize(self.next_observations[indices]),
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
        }

    def save(self, directory: str, name: str):
        """ Saves all tensors and normalization info to file `directory/name` """
        data = {
            "observations": self.observations,
            "actions": self.actions,
            "next_observations": self.next_observations,
            "rewards": self.rewards,
            "dones": self.dones,
            "position": self.position,
            "mean_obs": self.mean_obs,
            "std_obs": self.std_obs,
            "m2_obs": self._m2_obs,
            "count_seen": self.count_seen,
        }

        tmp_filename = tempfile.NamedTemporaryFile(delete=False, dir=directory)
        try:
            torch.save(data, tmp_filename)
        except BaseException:
            tmp_filename.close()
            os.remove(tmp_filename.name)
            raise
        else:
            tmp_filename.close()
            full_path = os.path.join(directory, name)
            os.rename(tmp_filename.name, full_path)
            return full_path

    def load(self, path: str, capacity: Optional[int] = None):
        """Loads the replay buffer from the specified path.

        Args:
            path(str): The path from where the memory will be loaded from.
            capacity(int): If provided, the buffer is created with this much capacity. This
                    value must be larger than the length of the stored tensors.
        """
        data = torch.load(path)
        self.position = data["position"]
        self.mean_obs = data["mean_obs"]
        self.std_obs = data["std_obs"]
        self._m2_obs = data["m2_obs"]
        self.count_seen = data["count_seen"]

        old_len = data["observations"].shape[0]
        if capacity is None:
            self.observations = data["observations"]
            self.actions = data["actions"]
            self.next_observations = data["next_observations"]
            self.rewards = data["rewards"]
            self.dones = data["dones"]
        else:
            assert capacity >= len(data["observations"])
            obs_shape = data["observations"].shape[1:]
            self.observations = torch.zeros(capacity, *obs_shape, dtype=torch.float32)
            self.actions = torch.zeros(capacity, dtype=torch.long)
            self.next_observations = torch.zeros(
                capacity, *obs_shape, dtype=torch.float32
            )
            self.rewards = torch.zeros(capacity, dtype=torch.float32)
            self.dones = torch.zeros(capacity, dtype=torch.bool)
            self.observations[:old_len] = data["observations"]
            self.actions[:old_len] = data["actions"]
            self.next_observations[:old_len] = data["next_observations"]
            self.rewards[:old_len] = data["rewards"]
            self.dones[:old_len] = data["dones"]

        return old_len

    def __len__(self):
        return len(self.observations)


###################### cvpr19_models evaluator network ######################

class SimpleSequential(nn.Module):
    def __init__(self, net1, net2):
        super(SimpleSequential, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x, mask):
        output = self.net1(x, mask)
        return self.net2(output, mask)


class SpectralMapDecomposition(nn.Module):
    def __init__(self):
        super(SpectralMapDecomposition, self).__init__()

    def forward(self, reconstructed_image, mask_embedding, mask):
        batch_size = reconstructed_image.shape[0]
        height = reconstructed_image.shape[2]
        width = reconstructed_image.shape[3]

        # create spectral maps in kspace
        kspace = fft(reconstructed_image)
        kspace = kspace.unsqueeze(1).repeat(1, width, 1, 1, 1)

        # separate image into spectral maps
        separate_mask = torch.zeros([1, width, 1, 1, width], dtype=torch.float32)
        for i in range(width):
            separate_mask[0, i, 0, 0, i] = 1

        separate_mask = separate_mask.to(reconstructed_image.device)

        masked_kspace = torch.where(
            separate_mask.byte(), kspace, torch.tensor(0.0).to(kspace.device)
        )
        masked_kspace = masked_kspace.view(batch_size * width, 2, height, width)

        # convert spectral maps to image space
        separate_images = ifft(masked_kspace)
        # result is (batch, [real_M0, img_M0, real_M1, img_M1, ...],  height, width]
        separate_images = separate_images.contiguous().view(
            batch_size, 2, width, height, width
        )

        # add mask information as a summation -- might not be optimal
        if mask is not None:
            separate_images = (
                separate_images + mask.permute(0, 3, 1, 2).unsqueeze(1).detach()
            )

        separate_images = separate_images.contiguous().view(
            batch_size, 2 * width, height, width
        )
        # concatenate mask embedding
        if mask_embedding is not None:
            spectral_map = torch.cat([separate_images, mask_embedding], dim=1)
        else:
            spectral_map = separate_images

        return spectral_map


class EvaluatorNetwork(nn.Module):
    """Evaluator network used in Zhang et al., CVPR'19.

    Args:
        number_of_filters(int): Number of filters used in convolutions. Defaults to 256. \n
        number_of_conv_layers(int): Depth of the model defined as a number of
                convolutional layers. Defaults to 4.
        use_sigmoid(bool): Whether the sigmoid non-linearity is applied to the
                output of the network. Defaults to False.
        width(int): The width of the image. Defaults to 128 (corresponds to DICOM).
        height(Optional[int]): The height of the image. If ``None`` the value of ``width``.
            is used. Defaults to ``None``.
        mask_embed_dim(int): Dimensionality of the mask embedding.
        num_output_channels(Optional[int]): The dimensionality of the output. If ``None``,
            the value of ``width`` is used. Defaults to ``None``.
    """

    def __init__(
        self,
        number_of_filters: int = 256,
        number_of_conv_layers: int = 4,
        use_sigmoid: bool = False,
        width: int = 128,
        height: Optional[int] = None,
        mask_embed_dim: int = 6,
        num_output_channels: Optional[int] = None,
    ):
        print(f"[EvaluatorNetwork] -> n_layers = {number_of_conv_layers}")
        super(EvaluatorNetwork, self).__init__()

        self.spectral_map = SpectralMapDecomposition()
        self.mask_embed_dim = mask_embed_dim

        if height is None:
            height = width

        number_of_input_channels = 2 * width + mask_embed_dim

        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        sequence = [
            nn.Conv2d(
                number_of_input_channels,
                number_of_filters,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2, True),
        ]

        in_channels = number_of_filters

        for n in range(1, number_of_conv_layers):
            if n < number_of_conv_layers - 1:
                if n <= 4:
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels // 2

            else:
                out_channels = in_channels

            sequence += [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(out_channels),
                nn.LeakyReLU(0.2, True),
            ]

            in_channels = out_channels
        kernel_size_width = width // 2 ** number_of_conv_layers
        kernel_size_height = height // 2 ** number_of_conv_layers
        sequence += [nn.AvgPool2d(kernel_size=(kernel_size_height, kernel_size_width))]

        if num_output_channels is None:
            num_output_channels = width
        sequence += [
            nn.Conv2d(
                in_channels, num_output_channels, kernel_size=1, stride=1, padding=0
            )
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)
        self.apply(init_func)

    def forward(
        self,
        input_tensor: torch.Tensor,
        mask_embedding: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """Computes scores for each k-space column.

        Args:
            input_tensor(torch.Tensor): Batch of reconstructed images,
                    as produced by :class:`models.reconstruction.ReconstructorNetwork`.
            mask_embedding(Optional[torch.Tensor]): Corresponding batch of mask embeddings
                    produced by :class:`models.reconstruction.ReconstructorNetwork`, if needed.
            mask(Optional[torch.Tensor]): Corresponding masks arrays, if needed.

        Returns:
            torch.Tensor: Evaluator score for each k-space column in each image in the batch.
        """
        spectral_map_and_mask_embedding = self.spectral_map(
            input_tensor, mask_embedding, mask
        )
        # Convert complex tensor to real tensor by taking magnitude
        if torch.is_complex(spectral_map_and_mask_embedding): # new added
            spectral_map_and_mask_embedding = torch.abs(spectral_map_and_mask_embedding)
        
        return self.model(spectral_map_and_mask_embedding).squeeze(3).squeeze(2)



###################### DDQN Model ######################


class DDQN(nn.Module, Policy):
    """Implementation of Double DQN value network.

    The configuration is given by the ``opts`` argument, which must contain the following
    fields:

        - mask_embedding_dim(int): See
          :class:`cvpr19_models.models.evaluator.EvaluatorNetwork`.
        - gamma(float): Discount factor for target updates.
        - dqn_model_type(str): Describes the architecture of the neural net. Options
          are "simple_mlp" and "evaluator", to use :class:`SimpleMLP` and
          :class:`EvaluatorBasedValueNetwork`, respectively.
        - budget(int): The environment's budget.
        - image_width(int): The width of the input images.

    Args:
        device(``torch.device``): Device to use.
        memory(optional(``replay_buffer.ReplayMemory``)): Replay buffer to sample transitions
            from. Can be ``None``, for example, if this is a target network.
        opts(``argparse.Namespace``): Options for the algorithm as explained above.
    """

    def __init__(
        self,
        device: torch.device,
        memory: Optional[replay_buffer],
        opts: argparse.Namespace,
    ):
        super().__init__()
        self.model = _get_model(opts)
        self.memory = memory
        self.optimizer = optim.Adam(self.parameters(), lr=opts.dqn_learning_rate)
        self.opts = opts
        self.device = device
        self.random_sampler = RandomPolicy()
        self.to(device)

    def add_experience(
        self,
        observation: np.array,
        action: int,
        next_observation: np.array,
        reward: float,
        done: bool,
    ):
        self.memory.push(observation, action, next_observation, reward, done)

    def update_parameters(self, target_net: nn.Module) -> Optional[Dict[str, Any]]:
        self.model.train()
        batch = self.memory.sample()
        if batch is None:
            return None
        observations = batch["observations"].to(self.device)
        next_observations = batch["next_observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device).squeeze()
        dones = batch["dones"].to(self.device)

        not_done_mask = dones.logical_not().squeeze()

        # Compute Q-values and get best action according to online network
        output_cur_step = self.forward(observations)
        all_q_values_cur = output_cur_step
        q_values = all_q_values_cur.gather(1, actions.unsqueeze(1))

        # Compute target values using the best action found
        if self.opts.gamma == 0.0:
            target_values = rewards
        else:
            with torch.no_grad():
                all_q_values_next = self.forward(next_observations)
                target_values = torch.zeros(observations.shape[0], device=self.device)
                del observations
                if not_done_mask.any().item() != 0:
                    best_actions = all_q_values_next.detach().max(1)[1]
                    target_values[not_done_mask] = (
                        target_net.forward(next_observations)
                        .gather(1, best_actions.unsqueeze(1))[not_done_mask]
                        .squeeze()
                        .detach()
                    )

                target_values = self.opts.gamma * target_values + rewards

        # loss = F.mse_loss(q_values, target_values.unsqueeze(1))
        loss = F.smooth_l1_loss(q_values, target_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        # Compute total gradient norm (for logging purposes) and then clip gradients
        grad_norm: torch.Tensor = 0  # type: ignore
        for p in list(filter(lambda p: p.grad is not None, self.parameters())):
            grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        torch.nn.utils.clip_grad_value_(self.parameters(), 1)

        self.optimizer.step()

        torch.cuda.empty_cache()

        return {
            "loss": loss,
            "grad_norm": grad_norm,
            "q_values_mean": q_values.detach().mean().cpu().numpy(),
            "q_values_std": q_values.detach().std().cpu().numpy(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts action values.

        Args:
            x(torch.Tensor): The observation tensor.

        Returns:
            Dictionary(torch.Tensor): The predicted Q-values.

        Note:
            Values corresponding to active k-space columns in the observation are manually
            set to ``1e-10``.
        """
        return self.model(x)

    def get_action(  # type: ignore
        self, obs: Dict[str, Any], eps_threshold: float = 0.0
    ) -> List[int]:
        """Returns an action sampled from an epsilon-greedy policy.

        With probability epsilon sample a random k-space column (ignoring active columns),
        otherwise return the column with the highest estimated Q-value for the observation.

        Args:
            obs(torch.Tensor): The observation for which an action is required.
            eps_threshold(float): The probability of sampling a random action instead of using
                a greedy action.
        """
        sample = random.random()
        if sample < eps_threshold:
            return self.random_sampler.get_action(obs)
        with torch.no_grad():
            self.model.eval()
            obs_tensor = _encode_obs_dict(obs)
            q_values = self(obs_tensor.to(self.device))
        actions = torch.argmax(q_values, dim=1) + getattr(self.opts, "legacy_offset", 0)
        return actions.tolist()


def _get_folder_lock(path):
    return filelock.FileLock(path, timeout=-1)




###################### Policy Generation ######################


class RandomPolicy(Policy):
    """A policy representing random k-space selection.

    Returns one of the valid actions uniformly at random.

    Args:
        seed(optional(int)): The seed to use for the random number generator, which is
            based on ``torch.Generator()``.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.rng = torch.Generator()
        if seed:
            self.rng.manual_seed(seed)

    def get_action(self, obs: Dict[str, Any], **_kwargs) -> List[int]:
        """Returns a random action without replacement.

        Args:
            obs(dict(str, any)): As returned by :class:`activemri.envs.ActiveMRIEnv`.

        Returns:
            list(int): A list of random k-space column indices, one per batch element in
                the observation. The indices are sampled from the set of inactive (0) columns
                on each batch element.
        """
        return (
            (obs["mask"].logical_not().float() + 1e-6)
            .multinomial(1, generator=self.rng)
            .squeeze()
            .tolist()
        )


class RandomLowBiasPolicy(Policy):
    def __init__(
        self, acceleration: float, centered: bool = True, seed: Optional[int] = None
    ):
        super().__init__()
        self.acceleration = acceleration
        self.centered = centered
        self.rng = np.random.RandomState(seed)

    def get_action(self, obs: Dict[str, Any], **_kwargs) -> List[int]:
        mask = obs["mask"].squeeze().cpu().numpy()
        new_mask = self._cartesian_mask(mask)
        action = (new_mask - mask).argmax(axis=1)
        return action.tolist()

    @staticmethod
    def _normal_pdf(length: int, sensitivity: float):
        return np.exp(-sensitivity * (np.arange(length) - length / 2) ** 2)

    def _cartesian_mask(self, current_mask: np.ndarray) -> np.ndarray:
        batch_size, image_width = current_mask.shape
        pdf_x = RandomLowBiasPolicy._normal_pdf(
            image_width, 0.5 / (image_width / 10.0) ** 2
        )
        pdf_x = np.expand_dims(pdf_x, axis=0)
        lmda = image_width / (2.0 * self.acceleration)
        # add uniform distribution
        pdf_x += lmda * 1.0 / image_width
        # remove previously chosen columns
        # note that pdf_x designed for centered masks
        new_mask = (
            np.fft.ifftshift(current_mask, axes=1)
            if not self.centered
            else current_mask.copy()
        )
        pdf_x = pdf_x * np.logical_not(new_mask)
        # normalize probabilities and choose accordingly
        pdf_x /= pdf_x.sum(axis=1, keepdims=True)
        indices = [
            self.rng.choice(image_width, 1, False, pdf_x[i]).item()
            for i in range(batch_size)
        ]
        new_mask[range(batch_size), indices] = 1
        if not self.centered:
            new_mask = np.fft.ifftshift(new_mask, axes=1)
        return new_mask


class LowestIndexPolicy(Policy):
    """A policy that represents low-to-high frequency k-space selection.

    Args:
        alternate_sides(bool): If ``True`` the indices of selected actions will alternate
            between the sides of the mask. For example, for an image with 100
            columns, and non-centered k-space, the order will be 0, 99, 1, 98, 2, 97, ..., etc.
            For the same size and centered, the order will be 49, 50, 48, 51, 47, 52, ..., etc.

        centered(bool): If ``True`` (default), low frequencies are in the center of the mask.
            Otherwise, they are in the edges of the mask.
    """

    def __init__(
        self,
        alternate_sides: bool,
        centered: bool = True,
    ):
        super().__init__()
        self.alternate_sides = alternate_sides
        self.centered = centered
        self.bottom_side = True

    def get_action(self, obs: Dict[str, Any], **_kwargs) -> List[int]:
        """Returns a random action without replacement.

        Args:
            obs(dict(str, any)): As returned by :class:`activemri.envs.ActiveMRIEnv`.

        Returns:
            list(int): A list of k-space column indices, one per batch element in
                the observation, equal to the lowest non-active k-space column in their
                corresponding observation masks.
        """
        mask = obs["mask"].squeeze().cpu().numpy()
        new_mask = self._get_new_mask(mask)
        action = (new_mask - mask).argmax(axis=1)
        return action.tolist()

    def _get_new_mask(self, current_mask: np.ndarray) -> np.ndarray:
        # The code below assumes mask in non centered
        new_mask = (
            np.fft.ifftshift(current_mask, axes=1)
            if self.centered
            else current_mask.copy()
        )
        if self.bottom_side:
            idx = np.arange(new_mask.shape[1], 0, -1)
        else:
            idx = np.arange(new_mask.shape[1])
        if self.alternate_sides:
            self.bottom_side = not self.bottom_side
        # Next line finds the first non-zero index (from edge to center) and returns it
        indices = (np.logical_not(new_mask) * idx).argmax(axis=1)
        indices = np.expand_dims(indices, axis=1)
        new_mask[range(new_mask.shape[0]), indices] = 1
        if self.centered:
            new_mask = np.fft.ifftshift(new_mask, axes=1)
        return new_mask




########################## Action Value Network ##########################

class SimpleMLP(nn.Module):
    """ Value network used for dataset specific DDQN model. """

    def __init__(
        self,
        budget: int,
        image_width: int,
        num_hidden_layers: int = 2,
        hidden_size: int = 32,
        ignore_mask: bool = True,
    ):
        super().__init__()
        self.ignore_mask = ignore_mask
        self.num_inputs = budget if self.ignore_mask else image_width
        num_actions = image_width
        self.linear1 = nn.Sequential(nn.Linear(self.num_inputs, hidden_size), nn.ReLU())
        hidden_layers = []
        for i in range(num_hidden_layers):
            hidden_layers.append(
                nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
            )
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Linear(hidden_size, num_actions)
        self.model = nn.Sequential(self.linear1, self.hidden, self.output)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Predicts action values.

        Args:
            obs(torch.Tensor): The observation tensor. Once decoded, it only uses the mask
                               information. If ``__init__(..., ignore_mask=True)``, it will
                               additionally use the mask only to deduce the time step.

        Returns:
            torch.Tensor: Q-values for all actions at the given observation.

        Note:
            Values corresponding to active k-space columns in the observation are manually
            set to ``1e-10``.
        """
        _, mask, _ = _decode_obs_tensor(obs, 0)
        previous_actions = mask.squeeze()

        if self.ignore_mask:
            input_tensor = torch.zeros(obs.shape[0], self.num_inputs).to(obs.device)
            time_steps = previous_actions.sum(1).unsqueeze(1)
            # We allow the model to receive observations that are over budget during test
            # Code below randomizes the input to the model for these observations
            index_over_budget = (time_steps >= self.num_inputs).squeeze()
            time_steps = time_steps.clamp(0, self.num_inputs - 1)
            input_tensor.scatter_(1, time_steps.long(), 1)
            input_tensor[index_over_budget] = torch.randn_like(
                input_tensor[index_over_budget]
            )
        else:
            input_tensor = mask

        value = self.model(input_tensor)
        return value - 1e10 * previous_actions


########################## Action Value Network ##########################

class EvaluatorBasedValueNetwork(nn.Module):
    """ Value network based on Zhang et al., CVPR'19 evaluator architecture. """

    def __init__(
        self, image_width: int, mask_embed_dim: int, legacy_offset: Optional[int] = None
    ):
        super().__init__()
        num_actions = image_width
        if legacy_offset:
            num_actions -= 2 * legacy_offset
        self.legacy_offset = legacy_offset
        self.evaluator = EvaluatorNetwork(
            number_of_filters=128,
            number_of_conv_layers=4,
            use_sigmoid=False,
            width=image_width,
            mask_embed_dim=mask_embed_dim,
            num_output_channels=num_actions,
        )
        self.mask_embed_dim = mask_embed_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Predicts action values.

        Args:
            obs(torch.Tensor): The observation tensor.

        Returns:
            torch.Tensor: Q-values for all actions at the given observation.

        Note:
            Values corresponding to active k-space columns in the observation are manually
            set to ``1e-10``.
        """
        reconstruction, mask, mask_embedding = _decode_obs_tensor(
            obs, self.evaluator.mask_embed_dim
        )
        qvalue = self.evaluator(reconstruction, mask_embedding)
        if self.legacy_offset:
            mask = mask[..., self.legacy_offset : -self.legacy_offset]
        return qvalue - 1e10 * mask.squeeze()


###################### Evaluate DDQN ######################

def evaluation(
    env: envs.envs.ActiveMRIEnv,
    policy: Policy,
    num_episodes: int,
    seed: int,
    split: str,
    verbose: Optional[bool] = False,
) -> Tuple[Dict[str, np.ndarray], List[Tuple[Any, Any]]]:
    env.seed(seed)
    if split == "test":
        env.set_test()
    elif split == "val":
        env.set_val()
    else:
        raise ValueError(f"Invalid evaluation split: {split}.")

    score_keys = env.score_keys()
    all_scores = dict(
        (k, np.zeros((num_episodes * env.num_parallel_episodes, env.budget + 1)))
        for k in score_keys
    )
    all_img_ids = []
    trajectories_written = 0
    for episode in range(num_episodes):
        step = 0
        obs, meta = env.reset()
        if not obs:
            break  # no more images
        # in case the last batch is smaller
        actual_batch_size = len(obs["reconstruction"])
        if verbose:
            msg = ", ".join(
                [
                    f"({meta['fname'][i]}, {meta['slice_id'][i]})"
                    for i in range(actual_batch_size)
                ]
            )
            print(f"Read images: {msg}")
        for i in range(actual_batch_size):
            all_img_ids.append((meta["fname"][i], meta["slice_id"][i]))
        batch_idx = slice(
            trajectories_written, trajectories_written + actual_batch_size
        )
        for k in score_keys:
            all_scores[k][batch_idx, step] = meta["current_score"][k]
        trajectories_written += actual_batch_size
        all_done = False
        while not all_done:
            step += 1
            action = policy.get_action(obs)
            obs, reward, done, meta = env.step(action)
            for k in score_keys:
                all_scores[k][batch_idx, step] = meta["current_score"][k]
            all_done = all(done)

    for k in score_keys:
        all_scores[k] = all_scores[k][: len(all_img_ids), :]
    return all_scores, all_img_ids




########################## Test DDQN ##########################


class DDQNTester:
    def __init__(
        self, env: mri_envs.ActiveMRIEnv, training_dir: str, device: torch.device
    ):
        self.env = env
        self.device = device

        self.training_dir = training_dir
        self.evaluation_dir = os.path.join(training_dir, "evaluation")
        os.makedirs(self.evaluation_dir, exist_ok=True)

        #self.folder_lock_path = DDQNTrainer.get_lock_filename(training_dir)
        self.folder_lock_path = '/home/yck/Desktop/GITHUB/Bayesian Reinforcement Learning/active-mri-acquisition_yck/Save_Training_Checkpoints_yck/.LOCK'

        #self.latest_policy_path = DDQNTrainer.get_name_latest_checkpoint(self.training_dir
        self.latest_policy_path = '/home/yck/Desktop/GITHUB/Bayesian Reinforcement Learning/active-mri-acquisition_yck/Save_Training_Checkpoints_yck/policy_checkpoint.pth'
        
        
        self.best_test_score = -np.inf
        self.last_time_stamp = -np.inf

        self.options = None

        # Initialize writer and logger
        '''
        self.writer = tensorboardX.SummaryWriter(os.path.join(self.evaluation_dir))
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(threadName)s - %(levelname)s: %(message)s"
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        ch.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(self.evaluation_dir, "evaluation.log"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Read the options used for training
        options_file_found = False
        while not options_file_found:
            options_filename = DDQNTrainer.get_options_filename(self.training_dir)
            with _get_folder_lock(self.folder_lock_path):
                if os.path.isfile(options_filename):
                    self.logger.info(f"Options file found at {options_filename}.")
                    with open(options_filename, "rb") as f:
                        self.options = pickle.load(f)
                    options_file_found = True
            if not options_file_found:
                self.logger.info(f"No options file found at {options_filename}.")
                self.logger.info("I will wait for five minutes before trying again.")
                time.sleep(300)
        # This change is needed so that util.test_policy writes results to correct directory
        self.options.checkpoints_dir = self.evaluation_dir
        os.makedirs(self.evaluation_dir, exist_ok=True)

        # Initialize environment
        self.options.image_width = self.env.action_space.n
        self.logger.info(f"Created environment with {self.env.action_space.n} actions")

        self.logger.info(f"Checkpoint dir for this job is {self.evaluation_dir}")
        self.logger.info(
            f"Evaluation will be done for model saved at {self.training_dir}"
        )
        '''

        # Initialize policy
        self.policy = DDQN(device, None, self.options)

        # Load info about best checkpoint tested and timestamp
        self.load_tester_checkpoint_if_present()

    def __call__(self):
        training_done = False
        while not training_done:
            training_done = self.check_if_train_done()
            self.logger.info(f"Is training done? {training_done}.")
            checkpoint_episode, timestamp = self.load_latest_policy()

            if timestamp is None or timestamp <= self.last_time_stamp:
                # No new policy checkpoint to evaluate
                self.logger.info(
                    "No new policy to evaluate. "
                    "I will wait for 10 minutes before trying again."
                )
                time.sleep(600)
                continue

            self.logger.info(
                f"Found a new checkpoint with timestamp {timestamp}, "
                f"I will start evaluation now."
            )
            test_scores, _ = evaluation.evaluate(
                self.env,
                self.policy,
                self.options.num_test_episodes,
                self.options.seed,
                "val",
                verbose=True,
            )
            auc_score = test_scores[self.options.reward_metric].sum(axis=1).mean()
            if "mse" in self.options.reward_metric:
                auc_score *= -1
            self.logger.info(f"The test score for the model was {auc_score}.")
            self.last_time_stamp = timestamp
            if auc_score > self.best_test_score:
                self.save_tester_checkpoint()
                policy_path = os.path.join(self.evaluation_dir, "policy_best.pt")
                self.save_policy(policy_path, checkpoint_episode)
                self.best_test_score = auc_score
                self.logger.info(
                    f"Saved DQN model with score {self.best_test_score} to {policy_path}, "
                    f"corresponding to episode {checkpoint_episode}."
                )

    def check_if_train_done(self):
        with _get_folder_lock(self.folder_lock_path):
            #done_file_path = DDQNTrainer.get_done_filename(self.training_dir)
            done_file_path= '/home/yck/Desktop/GITHUB/Bayesian Reinforcement Learning/active-mri-acquisition/Save_Training_Checkpoints_yck/DONE'
            return os.path.isfile(done_file_path)

    def checkpoint(self):
        self.save_tester_checkpoint()

    def save_tester_checkpoint(self):
        path = os.path.join(self.evaluation_dir, "tester_checkpoint.pickle")
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "best_test_score": self.best_test_score,
                    "last_time_stamp": self.last_time_stamp,
                },
                f,
            )

    def load_tester_checkpoint_if_present(self):
        path = os.path.join(self.evaluation_dir, "tester_checkpoint.pickle")
        if os.path.isfile(path):
            with open(path, "rb") as f:
                checkpoint = pickle.load(f)
            self.best_test_score = checkpoint["best_test_score"]
            self.last_time_stamp = checkpoint["last_time_stamp"]
            self.logger.info(
                f"Found checkpoint from previous evaluation run. "
                f"Best Score set to {self.best_test_score}. "
                f"Last Time Stamp set to {self.last_time_stamp}"
            )

    # noinspection PyProtectedMember
    def load_latest_policy(self):
        with _get_folder_lock(self.folder_lock_path):
            if not os.path.isfile(self.latest_policy_path):
                return None, None
            timestamp = os.path.getmtime(self.latest_policy_path)
            checkpoint = torch.load(self.latest_policy_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["dqn_weights"])
        return checkpoint["episode"], timestamp

    def save_policy(self, path, episode):
        torch.save(
            {
                "dqn_weights": self.policy.state_dict(),
                "episode": episode,
                "options": self.options,
            },
            path,
        )




###################### Reconstructor Network ######################



def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def init_func(m):
    init_type = "normal"
    gain = 0.02
    classname = m.__class__.__name__
    if hasattr(m, "weight") and (
        classname.find("Conv") != -1 or classname.find("Linear") != -1
    ):
        if init_type == "normal":
            torch.nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == "xavier":
            torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == "kaiming":
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        elif init_type == "orthogonal":
            torch.nn.init.orthogonal_(m.weight.data, gain=gain)
        else:
            raise NotImplementedError(
                "initialization method [%s] is not implemented" % init_type
            )
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, gain)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, dropout_probability, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, dropout_probability, use_bias
        )

    def build_conv_block(
        self, dim, padding_type, norm_layer, dropout_probability, use_bias
    ):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if dropout_probability > 0:
            conv_block += [nn.Dropout(dropout_probability)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ReconstructorNetwork(nn.Module):
    """Reconstructor network used in Zhang et al., CVPR'19.

    Args:
        number_of_encoder_input_channels(int): Number of input channels to the
                reconstruction model.
        number_of_decoder_output_channels(int): Number of output channels
                of the reconstruction model.
        number_of_filters(int): Number of convolutional filters.\n
        dropout_probability(float): Dropout probability.
        number_of_layers_residual_bottleneck (int): Number of residual
                blocks in each model between two consecutive down-
                or up-sampling operations.
        number_of_cascade_blocks (int): Number of times the entire architecture is
                replicated.
        mask_embed_dim(int): Dimensionality of the mask embedding.
        padding_type(str): Convolution operation padding type.
        n_downsampling(int): Number of down-sampling operations.
        img_width(int): The width of the image.
        use_deconv(binary): Whether to use deconvolution in the up-sampling.
    """

    def __init__(
        self,
        number_of_encoder_input_channels=2,
        number_of_decoder_output_channels=3,
        number_of_filters=128,
        dropout_probability=0.0,
        number_of_layers_residual_bottleneck=6,
        number_of_cascade_blocks=3,
        mask_embed_dim=6,
        padding_type="reflect",
        n_downsampling=3,
        img_width=128,
        use_deconv=True,
    ):
        super(ReconstructorNetwork, self).__init__()
        self.number_of_encoder_input_channels = number_of_encoder_input_channels
        self.number_of_decoder_output_channels = number_of_decoder_output_channels
        self.number_of_filters = number_of_filters
        self.use_deconv = use_deconv
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.number_of_cascade_blocks = number_of_cascade_blocks
        self.use_mask_embedding = True if mask_embed_dim > 0 else False

        if self.use_mask_embedding:
            number_of_encoder_input_channels += mask_embed_dim
            print("[Reconstructor Network] -> use masked embedding condition")

        # Lists of encoder, residual bottleneck and decoder blocks for all cascade blocks
        self.encoders_all_cascade_blocks = nn.ModuleList()
        self.residual_bottlenecks_all_cascade_blocks = nn.ModuleList()
        self.decoders_all_cascade_blocks = nn.ModuleList()

        # Architecture for the Cascade Blocks
        for iii in range(1, self.number_of_cascade_blocks + 1):

            # Encoder for iii_th cascade block
            encoder = [
                nn.ReflectionPad2d(1),
                nn.Conv2d(
                    number_of_encoder_input_channels,
                    number_of_filters,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    bias=use_bias,
                ),
                norm_layer(number_of_filters),
                nn.ReLU(True),
            ]

            for i in range(1, n_downsampling):
                mult = 2 ** i
                encoder += [
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(
                        number_of_filters * mult // 2,
                        number_of_filters * mult,
                        kernel_size=3,
                        stride=2,
                        padding=0,
                        bias=use_bias,
                    ),
                    norm_layer(number_of_filters * mult),
                    nn.ReLU(True),
                ]

            self.encoders_all_cascade_blocks.append(nn.Sequential(*encoder))

            # Bottleneck for iii_th cascade block
            residual_bottleneck = []
            mult = 2 ** (n_downsampling - 1)
            for i in range(number_of_layers_residual_bottleneck):
                residual_bottleneck += [
                    ResnetBlock(
                        number_of_filters * mult,
                        padding_type=padding_type,
                        norm_layer=norm_layer,
                        dropout_probability=dropout_probability,
                        use_bias=use_bias,
                    )
                ]

            self.residual_bottlenecks_all_cascade_blocks.append(
                nn.Sequential(*residual_bottleneck)
            )

            # Decoder for iii_th cascade block
            decoder = []
            for i in range(n_downsampling):
                mult = 2 ** (n_downsampling - 1 - i)
                if self.use_deconv:
                    decoder += [
                        nn.ConvTranspose2d(
                            number_of_filters * mult,
                            int(number_of_filters * mult / 2),
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            bias=use_bias,
                        ),
                        norm_layer(int(number_of_filters * mult / 2)),
                        nn.ReLU(True),
                    ]
                else:
                    decoder += [nn.Upsample(scale_factor=2), nn.ReflectionPad2d(1)] + [
                        nn.Conv2d(
                            number_of_filters * mult,
                            int(number_of_filters * mult / 2),
                            kernel_size=3,
                            stride=1,
                            padding=0,
                            bias=use_bias,
                        ),
                        norm_layer(int(number_of_filters * mult / 2)),
                        nn.ReLU(True),
                    ]
            decoder += [
                nn.Conv2d(
                    number_of_filters // 2,
                    number_of_decoder_output_channels,
                    kernel_size=1,
                    padding=0,
                    bias=False,
                )
            ]  # better

            self.decoders_all_cascade_blocks.append(nn.Sequential(*decoder))

        if self.use_mask_embedding:
            self.mask_embedding_layer = nn.Sequential(
                nn.Conv2d(img_width, mask_embed_dim, 1, 1)
            )

        self.apply(init_func)

    def data_consistency(self, x, input, mask):
        ft_x = fft(x)
        fuse = (
            ifft(
                torch.where((1 - mask).byte(), ft_x, torch.tensor(0.0).to(ft_x.device))
            )
            + input
        )
        return fuse

    def embed_mask(self, mask):
        b, c, h, w = mask.shape
        mask = mask.view(b, w, 1, 1)
        cond_embed = self.mask_embedding_layer(mask)
        return cond_embed

    # noinspection PyUnboundLocalVariable
    def forward(self, zero_filled_input, mask):
        """Generates reconstructions given images with partial k-space info.

        Args:
            zero_filled_input(torch.Tensor): Image obtained from zero-filled reconstruction
                of partial k-space scans.
            mask(torch.Tensor): Mask used in creating the zero filled image from ground truth
                image.

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor): Contains:\n
                * Reconstructed high resolution image.
                * Uncertainty map.
                * Mask_embedding.
        """
        if self.use_mask_embedding:
            mask_embedding = self.embed_mask(mask)
            mask_embedding = mask_embedding.repeat(
                1, 1, zero_filled_input.shape[2], zero_filled_input.shape[3]
            )
            encoder_input = torch.cat([zero_filled_input, mask_embedding], 1)
        else:
            encoder_input = zero_filled_input
            mask_embedding = None

        residual_bottleneck_output = None
        for cascade_block, (encoder, residual_bottleneck, decoder) in enumerate(
            zip(
                self.encoders_all_cascade_blocks,
                self.residual_bottlenecks_all_cascade_blocks,
                self.decoders_all_cascade_blocks,
            )
        ):
            encoder_output = encoder(encoder_input)
            if cascade_block > 0:
                # Skip connection from previous residual block
                encoder_output = encoder_output + residual_bottleneck_output

            residual_bottleneck_output = residual_bottleneck(encoder_output)

            decoder_output = decoder(residual_bottleneck_output)

            reconstructed_image = self.data_consistency(
                decoder_output[:, :-1, ...], zero_filled_input, mask
            )
            uncertainty_map = decoder_output[:, -1:, :, :]

            if self.use_mask_embedding:
                encoder_input = torch.cat([reconstructed_image, mask_embedding], 1)
            else:
                encoder_input = reconstructed_image

        return reconstructed_image, uncertainty_map, mask_embedding

    def init_from_checkpoint(self, checkpoint):

        if not isinstance(self, nn.DataParallel):
            self.load_state_dict(
                {
                    # This assumes that environment code runs in a single GPU
                    key.replace("module.", ""): val
                    for key, val in checkpoint["reconstructor"].items()
                }
            )
        else:
            self.load_state_dict(checkpoint["reconstructor"])
        return checkpoint["options"]



###################### Fourier Transform ######################

def roll(x, shift, dim):
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


# note that for IFFT we do not use irfft
# this function returns two channels where the first one (real part) is in image space
def ifftshift(x, dim=None):
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def fftshift(x, dim=None):
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifft(x, normalized=False, ifft_shift=False):
    x = x.permute(0, 2, 3, 1)
    #y = torch.ifft(x, 2, normalized=normalized)
    y = torch.fft.ifftn(x, dim=(1, 2), norm='backward' if not normalized else 'ortho') # new added
    if ifft_shift:
        y = ifftshift(y, dim=(1, 2))
    return y.permute(0, 3, 1, 2)


def rfft(x, normalized=False):
    # x is in gray scale and has 1-d in the 1st dimension
    x = x.squeeze(1)
    y = torch.rfft(x, 2, onesided=False, normalized=normalized)
    return y.permute(0, 3, 1, 2)


def fft(x, normalized=False, shift=False):
    x = x.permute(0, 2, 3, 1)
    if shift:
        x = fftshift(x, dim=(1, 2))
    #y = torch.fft(x, 2, normalized=normalized)
    y = torch.fft.fftn(x, dim=(1, 2), norm='backward' if not normalized else 'ortho') # new added
    
    return y.permute(0, 3, 1, 2)


def center_crop(x, shape):
    assert 0 < shape[0] <= x.shape[-2]
    assert 0 < shape[1] <= x.shape[-1]
    w_from = (x.shape[-1] - shape[0]) // 2
    h_from = (x.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    x = x[..., h_from:h_to, w_from:w_to]
    return x


def to_magnitude(tensor):
    tensor = (tensor[:, 0, :, :] ** 2 + tensor[:, 1, :, :] ** 2) ** 0.5
    return tensor.unsqueeze(1)


def dicom_to_0_1_range(tensor):
    return (tensor.clamp(-3, 3) + 3) / 6


def gaussian_nll_loss(reconstruction, target, logvar, options):
    reconstruction = to_magnitude(reconstruction)
    target = to_magnitude(target)
    if options.dataroot == "KNEE_RAW":
        reconstruction = center_crop(reconstruction, [320, 320])
        target = center_crop(target, [320, 320])
        logvar = center_crop(logvar, [320, 320])
    l2 = F.mse_loss(reconstruction, target, reduce=False)
    # Clip logvar to make variance in [0.0001, 5], for numerical stability
    logvar = logvar.clamp(-9.2, 1.609)
    one_over_var = torch.exp(-logvar)

    assert len(l2) == len(logvar)
    return 0.5 * (one_over_var * l2 + logvar)


def preprocess_inputs(batch, dataroot, device, prev_reconstruction=None):
    mask = batch[0].to(device)
    target = batch[1].to(device)
    if dataroot == "KNEE_RAW":
        k_space = batch[2].permute(0, 3, 1, 2).to(device)
        # alter mask to always include the highest frequencies that include padding
        mask = torch.where(
            to_magnitude(k_space).sum(2).unsqueeze(2) == 0.0,
            torch.tensor(1.0).to(device),
            mask,
        )
        if prev_reconstruction is None:
            masked_true_k_space = torch.where(
                mask.byte(), k_space, torch.tensor(0.0).to(device)
            )
        else:
            prev_reconstruction = prev_reconstruction.clone()
            prev_reconstruction[:, :, :160, :] = 0
            prev_reconstruction[:, :, -160:, :] = 0
            prev_reconstruction[:, :, :, :24] = 0
            prev_reconstruction[:, :, :, -24:] = 0
            ft_x = fft(prev_reconstruction, shift=True)
            masked_true_k_space = torch.where(mask.byte(), k_space, ft_x)
        reconstructor_input = ifft(masked_true_k_space, ifft_shift=True)
        target = target.permute(0, 3, 1, 2)
    else:
        fft_target = fft(target)
        if prev_reconstruction is None:
            masked_true_k_space = torch.where(
                mask.byte(), fft_target, torch.tensor(0.0).to(device)
            )
        else:
            ft_x = fft(prev_reconstruction)
            masked_true_k_space = torch.where(mask.byte(), fft_target, ft_x)

        reconstructor_input = ifft(masked_true_k_space)

    return reconstructor_input, target, mask





###################### Helper Functions ######################

def _get_epsilon(steps_done, opts):
    return opts.epsilon_end + (opts.epsilon_start - opts.epsilon_end) * math.exp(
        -1.0 * steps_done / opts.epsilon_decay
    )

def _encode_obs_dict(obs: Dict[str, Any]) -> torch.Tensor:
    reconstruction = obs["reconstruction"].permute(0, 3, 1, 2)
    mask_embedding = obs["extra_outputs"]["mask_embedding"]
    mask = obs["mask"]

    batch_size, num_channels, img_height, img_width = reconstruction.shape
    transformed_obs = torch.zeros(
        batch_size, num_channels, img_height + 2, img_width
    ).float()
    transformed_obs[..., :img_height, :] = reconstruction
    # The second to last row is the mask
    transformed_obs[..., img_height, :] = mask.unsqueeze(1)
    # The last row is the mask embedding (padded with 0s if necessary)
    if mask_embedding:
        mask_embed_dim = len(mask_embedding[0])
        transformed_obs[..., img_height + 1, :mask_embed_dim] = mask_embedding[
            :, :, 0, 0
        ]
    else:
        transformed_obs[:, :, img_height + 1, 0] = np.nan
    return transformed_obs



def _get_model(options):
    if options.dqn_model_type == "simple_mlp":
        return SimpleMLP(options.budget, options.image_width)
    if options.dqn_model_type == "evaluator":
        return EvaluatorBasedValueNetwork(
            options.image_width,
            options.mask_embedding_dim,
            legacy_offset=getattr(options, "legacy_offset", None),
        )
    raise ValueError("Unknown model specified for DQN.")


def _decode_obs_tensor(
    obs_tensor: torch.Tensor, mask_embed_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    reconstruction = obs_tensor[..., :-2, :]
    bs = obs_tensor.shape[0]
    if torch.isnan(obs_tensor[0, 0, -1, 0]).item() == 1:
        assert mask_embed_dim == 0
        mask_embedding = None
    else:
        mask_embedding = obs_tensor[:, 0, -1, :mask_embed_dim].view(bs, -1, 1, 1)
        mask_embedding = mask_embedding.repeat(
            1, 1, reconstruction.shape[2], reconstruction.shape[3]
        )

    mask = obs_tensor[:, 0, -2, :]
    mask = mask.contiguous().view(bs, 1, 1, -1)

    return reconstruction, mask, mask_embedding

# This is just a wrapper for the model in cvpr19_models folder
class CVPR19Evaluator(Policy):
    def __init__(
        self,
        evaluator_path: str,
        device: torch.device,
        add_mask: bool = False,
    ):
        super().__init__()
        evaluator_checkpoint = torch.load(evaluator_path)
        assert (
            evaluator_checkpoint is not None
            and evaluator_checkpoint["evaluator"] is not None
        )
        self.evaluator = EvaluatorNetwork(
            number_of_filters=evaluator_checkpoint[
                "options"
            ].number_of_evaluator_filters,
            number_of_conv_layers=evaluator_checkpoint[
                "options"
            ].number_of_evaluator_convolution_layers,
            use_sigmoid=False,
            width=evaluator_checkpoint["options"].image_width,
            height=640,
            mask_embed_dim=evaluator_checkpoint["options"].mask_embed_dim,
        )
        self.evaluator.load_state_dict(
            {
                key.replace("module.", ""): val
                for key, val in evaluator_checkpoint["evaluator"].items()
            }
        )
        self.evaluator.eval()
        self.evaluator.to(device)
        self.add_mask = add_mask
        self.device = device

    def get_action(self, obs: Dict[str, Any], **_kwargs) -> List[int]:
        with torch.no_grad():
            mask_embedding = (
                None
                if obs["extra_outputs"]["mask_embedding"] is None
                else obs["extra_outputs"]["mask_embedding"].to(self.device)
            )
            mask = obs["mask"].bool().to(self.device)
            mask = mask.view(mask.shape[0], 1, 1, -1)
            k_space_scores = self.evaluator(
                obs["reconstruction"].permute(0, 3, 1, 2).to(self.device),
                mask_embedding,
                mask if self.add_mask else None,
            )
            # Just fill chosen actions with some very large number to prevent from selecting again.
            k_space_scores.masked_fill_(mask.squeeze(), 100000)
            return torch.argmin(k_space_scores, dim=1).tolist()





if __name__ == "__main__":
    args = SimpleNamespace(
        budget=10,
        num_parallel_episodes=4,
        training_dir='/home/yck/Desktop/GITHUB/Bayesian Reinforcement Learning/active-mri-acquisition_yck/Save_Training_Checkpoints_yck',
        device=None,
        extreme_acc=False,
        seed=0
    )


    env = envs.MICCAI2020Env(
        args.num_parallel_episodes,
        args.budget,
        extreme=args.extreme_acc,
        seed=args.seed,
    )
    #tester = ddqn.DDQNTester(env, args.training_dir, args.device)
    tester = DDQNTester(env, args.training_dir, args.device)
    tester()



