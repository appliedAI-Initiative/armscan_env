from collections.abc import Callable
from typing import Any, Generic

import numpy as np
import torch
from tianshou.data.batch import BatchProtocol
from tianshou.utils.net.common import TRecurrentState
from torch import nn


# KEEP IN SYNC WITH ChanneledLabelmapsObsWithActReward
# This batch will be created automatically from these observations
class LabelmapsObsBatchProtocol(BatchProtocol):
    channelled_slice: np.ndarray
    action: np.ndarray
    reward: np.ndarray


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize a layer with the given standard deviation and bias constant."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DQN_MLP_Concat(nn.Module, Generic[TRecurrentState]):
    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_dim: int,
        device: str | int | torch.device = "cpu",
        mlp_output_dim: int = 512,
        layer_init: Callable[[nn.Module], nn.Module] = layer_init,
    ) -> None:
        super().__init__()
        self.device = device

        self.channeled_slice_cnn_CHW = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        mlp_input_dim = action_dim + 1  # action concatenated with reward
        self.action_reward_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, mlp_output_dim),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            base_cnn_output_dim = int(
                np.prod(self.channeled_slice_cnn_CHW(torch.zeros(1, c, h, w)).shape[1:]),
            )
        base_mlp_output_dim = mlp_output_dim
        concat_dim = base_cnn_output_dim + base_mlp_output_dim

        self.final_processing_mlp = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, action_dim),
        )
        self.output_dim = action_dim

    def forward(
        self,
        obs: LabelmapsObsBatchProtocol,
        state: Any | None = None,
    ) -> tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        # obs_batch = cast(
        #     LabelmapsObsBatchProtocol,
        #     Batch(
        #         channelled_labelmap_BCWH=obs["channeled_slice"],
        #         action_BA=obs["action"],
        #         reward_B=obs["reward"],
        #     ),
        # )

        channeled_slice = torch.as_tensor(obs.channeled_slice)
        image_output = self.channeled_slice_cnn_CHW(channeled_slice)

        action_reward = torch.concat(
            [torch.as_tensor(obs.action), torch.as_tensor(obs.reward)],
            dim=1,
        )
        action_reward_output = self.action_reward_mlp(action_reward)

        concat = torch.cat([image_output, action_reward_output], dim=1)
        return self.final_processing_mlp(concat), state
